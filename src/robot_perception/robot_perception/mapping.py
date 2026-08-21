"""Confirmed tracks become landmark proposals — under the same allowlist.

**The argument this module is built on.** `semantic_nav` treats the language
model as untrusted input and puts an allowlist between it and the motors. A
detector deserves exactly the same treatment, for exactly the same reasons:

  * It hallucinates. A pattern on a carpet becomes a "dog" at 0.6 confidence.
  * It can be fooled deliberately. Adversarial patches on objects are a
    published attack, and a printed photograph of a chair is a much easier one.
  * It can be fooled accidentally, which is more common: a poster, a reflection
    in a glass partition, a chair on a monitor showing a video call.

So perception gets the same positive-security model:

  1. **A category allowlist.** Only `INDOOR_CLASSES` can become a landmark.
     Whatever else the model emits, there is no code that would map it.
  2. **Propose, never overwrite.** A hand-authored landmark is immutable to
     perception. The robot may add "chair near the kitchen"; it may never move
     "charging dock", because a detector that relocates the dock strands the
     robot with a flat battery.
  3. **A budget.** At most `max_new_landmarks` additions per session. A
     detector stuck in a failure mode emits hundreds of boxes a second, and
     without a cap it fills the map with garbage faster than anyone notices.
  4. **Every proposal carries its evidence.** Sightings, viewpoints, spread,
     confidence. A landmark you cannot audit is a landmark you cannot trust,
     and this is what makes the map reviewable by a person before it is
     committed.

No ROS, no ML. Produces `robot_core.semantic_map.Landmark` objects that the
existing navigation stack already knows how to resolve and drive to.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from robot_core.geometry import Pose2D
from robot_core.semantic_map import Landmark, SemanticMap
from robot_perception.detection import INDOOR_CLASSES, NEVER_MAP
from robot_perception.tracking import ConfirmationPolicy, Track, evaluate

#: How close a proposal must be to an existing landmark of the same label to be
#: treated as the same object rather than a new one. Generous, because the
#: alternative failure — three "chair" landmarks stacked on one chair — makes
#: the map unusable for navigation, while merging two genuinely adjacent chairs
#: merely makes it slightly coarse.
MERGE_RADIUS_M = 1.2

#: Rough physical radius per class, used to size the landmark so the approach
#: pose lands outside the object rather than inside it. `SemanticMap` derives
#: the approach pose from this when none is given explicitly.
CLASS_RADIUS_M: Dict[str, float] = {
    "person": 0.35, "chair": 0.35, "couch": 0.90, "bed": 1.00,
    "dining table": 0.80, "potted plant": 0.30, "refrigerator": 0.40,
    "sink": 0.35, "toilet": 0.35, "tv": 0.45, "laptop": 0.20,
    "keyboard": 0.20, "mouse": 0.10, "book": 0.15, "cup": 0.08,
    "bottle": 0.08, "backpack": 0.25, "suitcase": 0.30, "clock": 0.15,
}

#: The category every perceived landmark is filed under in the semantic map.
#: Distinct from the hand-authored categories on purpose: it makes "which parts
#: of this map did a model write" answerable with a filter rather than by
#: memory.
PERCEIVED_CATEGORY = "perceived"


@dataclass
class LandmarkProposal:
    """A confirmed track, with the evidence that confirmed it."""

    name: str
    label: str
    x: float
    y: float
    radius_m: float
    observations: int
    distinct_viewpoints: int
    viewpoint_arc_rad: float
    spread_m: float
    confidence: float
    near: Optional[str] = None
    evidence: List[str] = field(default_factory=list)

    def to_landmark(self) -> Landmark:
        return Landmark(
            name=self.name,
            category=PERCEIVED_CATEGORY,
            x=self.x,
            y=self.y,
            radius_m=self.radius_m,
            attributes={
                "source": "perception",
                "class": self.label,
                "observations": str(self.observations),
                "viewpoints": str(self.distinct_viewpoints),
                "spread_m": f"{self.spread_m:.3f}",
                "confidence": f"{self.confidence:.3f}",
            },
        )

    def describe(self) -> str:
        return (f"{self.name}  ({self.x:.2f}, {self.y:.2f})  "
                f"{self.observations} sightings / "
                f"{self.distinct_viewpoints} viewpoints / "
                f"{math.degrees(self.viewpoint_arc_rad):.0f} deg arc / "
                f"spread {self.spread_m:.2f} m / conf {self.confidence:.2f}")


@dataclass
class ProposalOutcome:
    """What perception wants to change, and what it was refused."""

    proposals: List[LandmarkProposal] = field(default_factory=list)
    rejected: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.proposals)

    def summary(self) -> str:
        lines = [f"{len(self.proposals)} proposal(s), "
                 f"{len(self.rejected)} rejected"]
        lines += [f"  + {p.describe()}" for p in self.proposals]
        lines += [f"  - {r}" for r in self.rejected]
        return "\n".join(lines)


def slugify(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def name_for(label: str, x: float, y: float, semantic_map: SemanticMap,
             taken: Sequence[str], near_radius_m: float = 4.0) -> tuple:
    """A name a person would use, and the landmark it is near.

    "chair 3" is unusable — nobody can say which chair they mean. "chair near
    the kitchen" is something an operator can act on, and it is resolvable by
    the existing fuzzy matcher in `SemanticMap`. Falling back to a numbered
    name only when there is no anchor nearby keeps the map speakable.

    Anchors are drawn from the HAND-AUTHORED landmarks only. Naming a perceived
    chair after another perceived chair would compound one model's error into
    the name of the next thing, and the names drift further from reality with
    every patrol.
    """
    anchor = None
    best = near_radius_m
    for landmark in semantic_map:
        if landmark.category == PERCEIVED_CATEGORY:
            continue
        distance = math.hypot(landmark.x - x, landmark.y - y)
        if distance <= best:
            anchor, best = landmark, distance

    base = f"{label} near the {anchor.name}" if anchor else label
    existing = {slugify(n) for n in taken} | {slugify(n) for n in semantic_map.names()}

    if slugify(base) not in existing:
        return base, (anchor.name if anchor else None)

    for index in range(2, 100):
        candidate = f"{base} {index}"
        if slugify(candidate) not in existing:
            return candidate, (anchor.name if anchor else None)

    # 99 chairs near one anchor is a detector failure, not a furnished room.
    raise ValueError(f"cannot name another '{base}'; the detector is likely stuck")


def _existing_nearby(semantic_map: SemanticMap, label: str,
                     x: float, y: float) -> Optional[Landmark]:
    """An existing landmark that plausibly IS this object."""
    for landmark in semantic_map:
        if math.hypot(landmark.x - x, landmark.y - y) > MERGE_RADIUS_M:
            continue
        if landmark.attributes.get("class") == label:
            return landmark
        if slugify(landmark.name).startswith(slugify(label)):
            return landmark
    return None


def make_tracker(gate_m: float = 1.0,
                 policy: Optional[ConfirmationPolicy] = None):
    """A Tracker whose association gate knows how big each class of object is.

    Lives here rather than in `tracking` so that module keeps no table of
    real-world object sizes - it is about association, not about furniture.
    """
    from robot_perception.tracking import Tracker
    return Tracker(gate_m=gate_m, policy=policy, class_radius_m=CLASS_RADIUS_M)


def propose(tracks: Sequence[Track], semantic_map: SemanticMap,
            policy: Optional[ConfirmationPolicy] = None,
            max_new_landmarks: int = 12) -> ProposalOutcome:
    """Turn confirmed tracks into landmark proposals, refusing the rest.

    Refusals are returned rather than dropped. "The robot saw four chairs and
    mapped one" is a question an operator will ask, and the answer has to be in
    the output, not in a log line that scrolled past.
    """
    policy = policy or ConfirmationPolicy()
    outcome = ProposalOutcome()

    # Strongest evidence first, so the budget is spent on the best candidates
    # rather than on whatever the tracker happened to create earliest.
    ordered = sorted(
        tracks,
        key=lambda t: (-len(t), -t.confidence) if t.observations else (0, 0),
    )

    for track in ordered:
        if not track.observations:
            continue

        if track.label in NEVER_MAP:
            # Detected, published for the safety layer, never written down. A
            # map that records where somebody stood on Tuesday is both useless
            # and a privacy problem.
            outcome.rejected.append(
                f"{track.label}: excluded from mapping by policy (it moves, "
                f"and recording where a person stood is not navigation data)"
            )
            continue

        if track.label not in INDOOR_CLASSES:
            outcome.rejected.append(
                f"{track.label}: not in the indoor allowlist, so there is no "
                f"code path that would map it"
            )
            continue

        result = evaluate(track, policy, CLASS_RADIUS_M.get(track.label, 0.0))
        if not result.confirmed:
            outcome.rejected.append(f"{track.label}: {'; '.join(result.reasons)}")
            continue

        x, y = track.position
        duplicate = _existing_nearby(semantic_map, track.label, x, y)
        if duplicate is not None:
            outcome.rejected.append(
                f"{track.label} at ({x:.2f}, {y:.2f}): already mapped as "
                f"'{duplicate.name}' {math.hypot(duplicate.x - x, duplicate.y - y):.2f} m away"
            )
            continue

        if len(outcome.proposals) >= max_new_landmarks:
            outcome.rejected.append(
                f"{track.label} at ({x:.2f}, {y:.2f}): budget of "
                f"{max_new_landmarks} new landmarks is spent. A detector stuck "
                f"in a failure mode emits hundreds of boxes a second, and the "
                f"cap is what stops it filling the map faster than anyone "
                f"notices"
            )
            continue

        name, near = name_for(track.label, x, y, semantic_map,
                              [p.name for p in outcome.proposals])
        outcome.proposals.append(LandmarkProposal(
            name=name,
            label=track.label,
            x=x, y=y,
            radius_m=CLASS_RADIUS_M.get(track.label, 0.35),
            observations=len(track),
            distinct_viewpoints=track.distinct_viewpoints(
                policy.viewpoint_separation_m),
            viewpoint_arc_rad=track.viewpoint_arc(),
            spread_m=track.spread_m,
            confidence=track.confidence,
            near=near,
            evidence=list(result.reasons),
        ))

    return outcome


def apply(outcome: ProposalOutcome, semantic_map: SemanticMap) -> List[str]:
    """Add the proposals to the map. Returns the names actually added.

    **Additive only, by construction.** There is no code here that edits or
    removes an existing landmark, which is what makes "perception cannot move
    the charging dock" a property of the module rather than a promise in a
    docstring. Correcting a hand-authored landmark is a human decision, and a
    detector that relocates the dock strands the robot with a flat battery.
    """
    added: List[str] = []
    for proposal in outcome.proposals:
        landmark = proposal.to_landmark()
        try:
            semantic_map.add(landmark)
        except ValueError:
            # Name collision. `name_for` already avoids these, so reaching here
            # means two proposals raced; skipping is correct and the map is
            # unchanged either way.
            continue
        added.append(landmark.name)
    return added


def perceived(semantic_map: SemanticMap) -> List[Landmark]:
    """Only the landmarks a model wrote.

    The filter that makes "which parts of this map did perception author"
    answerable — for review before committing, and for discarding a whole
    patrol's output if the detector turns out to have been misbehaving.
    """
    return [lm for lm in semantic_map if lm.category == PERCEIVED_CATEGORY]


def coverage(semantic_map: SemanticMap, pose: Pose2D,
             radius_m: float = 3.0) -> Dict[str, int]:
    """What the map thinks is around a given pose, by class.

    Used by the inspection report: "the robot stood here and the map says there
    should be two chairs and a plant" is what a missing-object check compares
    against.
    """
    counts: Dict[str, int] = {}
    for landmark in semantic_map:
        if pose.distance_to(landmark.pose) > radius_m:
            continue
        key = landmark.attributes.get("class", landmark.category)
        counts[key] = counts.get(key, 0) + 1
    return counts
