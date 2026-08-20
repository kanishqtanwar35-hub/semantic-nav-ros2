"""The command schema — the contract between the language model and the robot.

Everything an LLM is allowed to ask this robot to do is enumerated here. That
is the single most important security property in the repository, and it is the
same positive-security model used in the text-to-SQL project: **an allowlist of
what is permitted, never a denylist of what is forbidden.**

A denylist for a robot ("don't drive into people, don't leave the building")
fails the first time someone phrases a request the list did not anticipate. An
allowlist fails closed: an instruction that does not parse into one of these
verbs, with a target that exists in the map, does not execute. There is no
third outcome.

Note what is *not* in this schema: coordinates. The model may not say "drive to
(4.2, -1.8)". It may only name a landmark, and the semantic map decides whether
that name exists and where it is. A model that hallucinates a place produces a
resolution failure, not a robot driving into an unmapped void.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Verb(str, Enum):
    """Everything the robot can be asked to do. Adding one is a deliberate act."""

    GO_TO = "go_to"
    PATROL = "patrol"
    RETURN_HOME = "return_home"
    STOP = "stop"
    WAIT = "wait"
    REPORT = "report"


#: Verbs that do not need a target. Everything else must name one.
TARGETLESS = {Verb.RETURN_HOME, Verb.STOP, Verb.WAIT, Verb.REPORT}

#: Hard ceiling on a single instruction. A model that emits "patrol the kitchen,
#: the lobby, the kitchen, the lobby..." 400 times has produced a denial of
#: service against a physical machine, and the fix is a limit rather than trust.
MAX_STEPS = 8

#: Longest a single WAIT may block for.
MAX_WAIT_S = 300.0


@dataclass
class Step:
    verb: Verb
    target: Optional[str] = None       # a landmark NAME, never coordinates
    seconds: Optional[float] = None    # WAIT only


@dataclass
class Command:
    """A parsed, not-yet-validated instruction."""

    steps: List[Step] = field(default_factory=list)
    utterance: str = ""
    source: str = "rules"              # rules | llm | llm-repaired
    confidence: float = 1.0
    notes: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.steps)


class ValidationError(Exception):
    """Raised when a command cannot be made safe. Never repaired silently."""


@dataclass
class ValidatedCommand:
    """A command whose every step resolves to a real, reachable pose.

    Constructing one of these is the *only* way to reach the Nav2 client. That
    is enforced by types rather than by discipline: `Nav2Client.execute` takes a
    `ValidatedCommand`, so there is no code path from raw model output to the
    motors that skips validation.
    """

    steps: List[Step]
    poses: List[tuple]                 # (x, y, yaw) per step, [] for targetless
    utterance: str
    source: str
    resolutions: List[str] = field(default_factory=list)

    def describe(self) -> str:
        lines = []
        for step, resolution in zip(self.steps, self.resolutions):
            lines.append(f"  {step.verb.value:<12} {resolution}")
        return "\n".join(lines)


def validate(command: Command, semantic_map, min_score: float = 0.75
             ) -> ValidatedCommand:
    """Turn a parsed command into an executable one, or refuse.

    Every rejection below is a real attack or a real failure mode:

      * **Unknown landmark** — the model hallucinated a place. Refusing is the
        only safe answer; the nearest-sounding match is a robot confidently
        driving somewhere nobody asked for.
      * **Ambiguous landmark** — two candidates within a hair of each other.
        Ask, do not guess.
      * **Too many steps** — a physical denial of service.
      * **Unbounded wait** — a robot parked in a doorway for eleven hours.
      * **Unknown verb** — cannot happen through `Verb`, but the JSON path can
        produce arbitrary strings, so the enum conversion is where it dies.
    """
    if not command.steps:
        raise ValidationError("no actionable step was found in that instruction")

    if len(command.steps) > MAX_STEPS:
        raise ValidationError(
            f"{len(command.steps)} steps exceeds the {MAX_STEPS}-step limit"
        )

    poses: List[tuple] = []
    resolutions: List[str] = []

    for position, step in enumerate(command.steps, start=1):
        if not isinstance(step.verb, Verb):
            raise ValidationError(f"step {position}: unknown verb {step.verb!r}")

        if step.verb is Verb.WAIT:
            seconds = step.seconds if step.seconds is not None else 5.0
            if seconds <= 0:
                raise ValidationError(f"step {position}: wait must be positive")
            if seconds > MAX_WAIT_S:
                raise ValidationError(
                    f"step {position}: wait of {seconds:g}s exceeds the "
                    f"{MAX_WAIT_S:g}s limit"
                )
            step.seconds = seconds
            poses.append(())
            resolutions.append(f"wait {seconds:g}s")
            continue

        if step.verb in TARGETLESS:
            if step.verb is Verb.RETURN_HOME:
                home = semantic_map.get("charging dock") or semantic_map.get("home")
                if home is None:
                    raise ValidationError(
                        "no 'charging dock' landmark exists in the map, so "
                        "there is nowhere to return to"
                    )
                pose = home.approach_pose
                poses.append((pose.x, pose.y, pose.yaw))
                resolutions.append(f"-> {home.name}")
            else:
                poses.append(())
                resolutions.append(step.verb.value)
            continue

        if not step.target:
            raise ValidationError(
                f"step {position}: '{step.verb.value}' needs somewhere to go"
            )

        match = semantic_map.resolve(step.target)
        if match.landmark is None:
            known = ", ".join(sorted(semantic_map.names()))
            raise ValidationError(
                f"step {position}: I don't know a place called "
                f"'{step.target}'. I know: {known}"
            )
        if match.score < min_score:
            raise ValidationError(
                f"step {position}: '{step.target}' only weakly matches "
                f"'{match.landmark.name}' ({match.score:.2f}); I'd rather ask "
                f"than guess"
            )
        if match.ambiguous:
            options = [match.landmark.name] + [n for n, _ in match.alternatives[:1]]
            raise ValidationError(
                f"step {position}: '{step.target}' could mean "
                f"{' or '.join(options)} — which did you mean?"
            )

        pose = match.landmark.approach_pose
        poses.append((pose.x, pose.y, pose.yaw))
        # Store the RESOLVED name, not what the user said. The log has to show
        # where the robot actually went.
        step.target = match.landmark.name
        resolutions.append(f"-> {match.landmark.name} "
                           f"({pose.x:.2f}, {pose.y:.2f}) [{match.score:.2f}]")

    return ValidatedCommand(
        steps=command.steps,
        poses=poses,
        utterance=command.utterance,
        source=command.source,
        resolutions=resolutions,
    )
