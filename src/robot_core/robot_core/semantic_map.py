"""Semantic map — the bridge between "the kitchen" and (x, y, yaw).

Nav2 navigates to coordinates. People navigate to *places*. This module holds
the layer between the two, and it is the piece that turns a navigation stack
into something you can talk to.

A landmark is a named region with a pose the robot should actually stop at.
Those are different things, and conflating them is the first bug people hit:
the centroid of "the kitchen" is in the middle of the floor, which is fine, but
the centroid of "the desk" is *inside the desk*, and the robot cannot stand
there. So every landmark carries an explicit `approach` pose that may sit
outside its own footprint.

No ROS imports. Persisted as YAML so a map is editable by hand.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from robot_core.geometry import Pose2D


@dataclass
class Landmark:
    name: str
    category: str                      # room | furniture | object | waypoint
    x: float
    y: float
    yaw: float = 0.0
    aliases: List[str] = field(default_factory=list)
    radius_m: float = 0.5              # how big the thing is
    approach_x: Optional[float] = None
    approach_y: Optional[float] = None
    approach_yaw: Optional[float] = None
    floor: int = 0
    attributes: Dict[str, str] = field(default_factory=dict)

    @property
    def pose(self) -> Pose2D:
        return Pose2D(self.x, self.y, self.yaw)

    @property
    def approach_pose(self) -> Pose2D:
        """Where the robot should stop.

        Defaults to a point `radius_m` back from the landmark, facing it, so a
        landmark defined without an explicit approach still yields a reachable
        goal rather than a goal inside a solid object.
        """
        if self.approach_x is not None and self.approach_y is not None:
            yaw = self.approach_yaw
            if yaw is None:
                yaw = math.atan2(self.y - self.approach_y, self.x - self.approach_x)
            return Pose2D(self.approach_x, self.approach_y, yaw)

        back = self.radius_m + 0.35
        return Pose2D(
            self.x - back * math.cos(self.yaw),
            self.y - back * math.sin(self.yaw),
            self.yaw,
        )

    def all_names(self) -> List[str]:
        return [self.name, *self.aliases]


def _normalise(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"^(the|a|an)\s+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class MatchResult:
    landmark: Optional[Landmark]
    score: float
    matched_on: str = ""
    alternatives: List[Tuple[str, float]] = field(default_factory=list)

    @property
    def confident(self) -> bool:
        return self.landmark is not None and self.score >= 0.75

    @property
    def ambiguous(self) -> bool:
        """Two candidates close enough that picking either is a coin flip.

        This deserves a question to the user, not a guess. A robot that guesses
        between "the office" and "the office kitchen" and drives off is worse
        than one that asks, because by then the person has stopped watching.
        """
        if self.landmark is None or not self.alternatives:
            return False
        return (self.score - self.alternatives[0][1]) < 0.08


class SemanticMap:
    def __init__(self, landmarks: Optional[Sequence[Landmark]] = None):
        self._landmarks: Dict[str, Landmark] = {}
        for landmark in landmarks or []:
            self.add(landmark)

    def add(self, landmark: Landmark) -> None:
        key = _normalise(landmark.name)
        if key in self._landmarks:
            raise ValueError(f"duplicate landmark name: {landmark.name}")
        self._landmarks[key] = landmark

    def __len__(self) -> int:
        return len(self._landmarks)

    def __iter__(self):
        return iter(self._landmarks.values())

    def names(self) -> List[str]:
        return [lm.name for lm in self._landmarks.values()]

    def get(self, name: str) -> Optional[Landmark]:
        return self._landmarks.get(_normalise(name))

    # -- resolution ----------------------------------------------------------

    def resolve(self, phrase: str) -> MatchResult:
        """Map a phrase to a landmark: exact, then alias, then fuzzy.

        Deliberately *not* an embedding model. Fuzzy string matching over a few
        dozen known names is accurate, instant, offline, and — the part that
        matters for a robot — returns a score you can threshold meaningfully.
        An embedding similarity of 0.71 between "the kitchen" and "the bathroom"
        tells you nothing actionable; both are rooms in a house and the model
        knows it.

        Semantics belong in the LLM layer above this, which turns free text into
        a *candidate name*. This layer's job is to be strict about whether that
        name exists.
        """
        query = _normalise(phrase)
        if not query:
            return MatchResult(None, 0.0, "empty query")

        scored: List[Tuple[Landmark, float, str]] = []
        for landmark in self._landmarks.values():
            best = 0.0
            best_on = ""
            for candidate in landmark.all_names():
                normalised = _normalise(candidate)
                if normalised == query:
                    score, on = 1.0, f"exact match on '{candidate}'"
                elif query in normalised:
                    # The candidate contains the query: the user named LESS than
                    # the full landmark. "kitchen" for "office kitchen" is a
                    # person being brief, which is normal and a strong signal.
                    score = 0.80 + 0.15 * (len(query) / len(normalised))
                    on = f"substring match on '{candidate}'"
                elif normalised in query:
                    # The query contains the candidate: the user named MORE than
                    # the landmark. Much weaker, and the two directions were
                    # scored identically until perception started producing
                    # compound names like "chair near the kitchen".
                    #
                    # That query contains "kitchen", so it scored 0.85 and the
                    # robot confidently drove to the kitchen - the wrong place,
                    # silently, which is exactly the failure the allowlist
                    # exists to prevent. The extra words are the specific part
                    # of the request, not noise to be discarded.
                    #
                    # Score by how much of what the user ASKED FOR this
                    # landmark actually explains.
                    score = 0.55 + 0.25 * (len(normalised) / len(query))
                    on = (f"'{candidate}' explains only "
                          f"{len(normalised) / len(query):.0%} of the request")
                else:
                    score = SequenceMatcher(None, query, normalised).ratio()
                    on = f"fuzzy match on '{candidate}'"
                if score > best:
                    best, best_on = score, on
            scored.append((landmark, best, best_on))

        scored.sort(key=lambda item: item[1], reverse=True)
        top, score, matched_on = scored[0]
        alternatives = [(lm.name, s) for lm, s, _ in scored[1:4]]

        if score < 0.55:
            return MatchResult(None, score,
                               f"no landmark resembles '{phrase}'", alternatives)
        return MatchResult(top, score, matched_on, alternatives)

    # -- spatial queries -----------------------------------------------------

    def nearest(self, pose: Pose2D,
                category: Optional[str] = None) -> Optional[Landmark]:
        candidates = [lm for lm in self._landmarks.values()
                      if category is None or lm.category == category]
        if not candidates:
            return None
        return min(candidates, key=lambda lm: pose.distance_to(lm.pose))

    def within(self, pose: Pose2D, radius_m: float) -> List[Landmark]:
        found = [lm for lm in self._landmarks.values()
                 if pose.distance_to(lm.pose) <= radius_m]
        found.sort(key=lambda lm: pose.distance_to(lm.pose))
        return found

    def by_category(self, category: str) -> List[Landmark]:
        return [lm for lm in self._landmarks.values() if lm.category == category]

    def describe(self) -> str:
        """A compact map summary to put in an LLM prompt.

        Deliberately terse. Every token spent describing the map is a token not
        spent on the instruction, and the model only needs to know which names
        exist.
        """
        lines = []
        for category in sorted({lm.category for lm in self._landmarks.values()}):
            names = sorted(lm.name for lm in self.by_category(category))
            lines.append(f"{category}: {', '.join(names)}")
        return "\n".join(lines)

    # -- persistence ---------------------------------------------------------

    @classmethod
    def from_dict(cls, payload: dict) -> "SemanticMap":
        return cls([Landmark(**entry) for entry in payload.get("landmarks", [])])

    def to_dict(self) -> dict:
        out = []
        for lm in self._landmarks.values():
            entry = {
                "name": lm.name, "category": lm.category,
                "x": lm.x, "y": lm.y, "yaw": lm.yaw,
                "radius_m": lm.radius_m, "floor": lm.floor,
            }
            if lm.aliases:
                entry["aliases"] = list(lm.aliases)
            if lm.approach_x is not None:
                entry["approach_x"] = lm.approach_x
                entry["approach_y"] = lm.approach_y
                if lm.approach_yaw is not None:
                    entry["approach_yaw"] = lm.approach_yaw
            if lm.attributes:
                entry["attributes"] = dict(lm.attributes)
            out.append(entry)
        return {"landmarks": out}

    @classmethod
    def load(cls, path) -> "SemanticMap":
        import yaml
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(yaml.safe_load(text) or {})

    def save(self, path) -> None:
        import yaml
        # Explicit LF so a map written on Windows is byte-identical to one
        # written on Linux. Without it Python translates the line endings on
        # write and the file changes for a reason nobody chose.
        Path(path).write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False),
            encoding="utf-8", newline="\n",
        )
