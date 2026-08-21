"""Turning noisy per-frame detections into things you are willing to put on a map.

**The insight this module exists for.** Everybody knows single-frame detections
are noisy, so everybody averages over frames. That removes *noise* and leaves
*bias* exactly where it was — and the dominant error here is bias, not noise.

`camera.ground_error_m` shows why: an object whose boxed point sits 5 cm off the
floor projects 1.5x too far away at this camera height. That error is identical
in every frame taken from the same place. A robot that stops and stares at a
shelf for a hundred frames gets a hundred consistent measurements of the wrong
position, and averaging makes it *more* confident about being wrong.

So confirmation requires **distinct viewpoints**, not merely many observations:

    N observations  AND  seen from >= K positions at least D apart

Two views separated by a metre disagree about a mis-projected object and agree
about a correctly-projected one. That disagreement is the signal — `spread_m`
measures it, and a track whose views disagree is refused rather than mapped,
because a large spread is direct evidence the ground-plane assumption is
failing for that particular object.

No ROS, no ML. Everything here is unit-testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Observation:
    """One detection, grounded into map coordinates."""

    label: str
    confidence: float
    #: Where the object appeared to be, in the map frame.
    x: float
    y: float
    #: Where the robot was standing when it saw it. Kept because the whole
    #: confirmation argument turns on viewpoint diversity, and that cannot be
    #: recovered later from the object position alone.
    from_x: float
    from_y: float
    #: Seconds. Used for ageing, not for ordering — frames can arrive out of
    #: order over a lossy transport.
    stamp: float = 0.0
    range_m: float = 0.0

    def distance_to(self, x: float, y: float) -> float:
        return math.hypot(self.x - x, self.y - y)


@dataclass
class Track:
    """Repeated observations believed to be the same physical object."""

    label: str
    observations: List[Observation] = field(default_factory=list)
    track_id: int = 0

    def add(self, observation: Observation) -> None:
        self.observations.append(observation)

    def __len__(self) -> int:
        return len(self.observations)

    @property
    def last_stamp(self) -> float:
        return max((o.stamp for o in self.observations), default=0.0)

    @property
    def confidence(self) -> float:
        """The best evidence seen, not the average.

        A detector that is 0.9 confident once and 0.4 confident four times has
        seen the object clearly once. Averaging to 0.5 throws away the good look
        in favour of the bad ones, which is backwards: the poor frames are
        usually poor because of distance or motion blur, not because the object
        is not there.
        """
        return max((o.confidence for o in self.observations), default=0.0)

    @property
    def position(self) -> Tuple[float, float]:
        """Component-wise median of the observed positions.

        Median rather than mean, and it matters more than it looks: one badly
        mis-projected frame — a box truncated at the image edge, a reflection —
        moves a mean by metres and a median not at all. The component-wise
        median is not the geometric median, but with a handful of points the
        difference is far below the measurement error and it costs nothing.
        """
        if not self.observations:
            raise ValueError("an empty track has no position")
        return (_median([o.x for o in self.observations]),
                _median([o.y for o in self.observations]))

    @property
    def spread_m(self) -> float:
        """How far apart the individual observations are.

        The honesty metric. Small spread means the views agree and the ground
        plane assumption is holding for this object. Large spread means it is
        not — the object is probably not on the floor — and the right response
        is to refuse it rather than to average the disagreement away.
        """
        if len(self.observations) < 2:
            return 0.0
        x, y = self.position
        return max(o.distance_to(x, y) for o in self.observations)

    @property
    def viewpoints(self) -> List[Tuple[float, float]]:
        return [(o.from_x, o.from_y) for o in self.observations]

    def distinct_viewpoints(self, min_separation_m: float) -> int:
        """How many meaningfully different places this was seen from.

        Greedy: walk the viewpoints and count one whenever it is at least
        `min_separation_m` from every viewpoint already counted. Greedy
        under-counts a little compared with the true maximum-separation subset,
        which is the safe direction — it can only make confirmation stricter.
        """
        chosen: List[Tuple[float, float]] = []
        for point in self.viewpoints:
            if all(math.hypot(point[0] - c[0], point[1] - c[1]) >= min_separation_m
                   for c in chosen):
                chosen.append(point)
        return len(chosen)

    def viewpoint_arc(self) -> float:
        """Angular spread of the viewpoints as seen FROM the object, in radians.

        The geometrically meaningful quantity. Two viewpoints ten metres apart
        but both directly in front of the object give almost no new information
        about its depth; two a metre apart at right angles to each other give a
        great deal. This is what actually breaks the ground-plane bias.
        """
        if len(self.observations) < 2:
            return 0.0
        x, y = self.position
        angles = sorted(math.atan2(o.from_y - y, o.from_x - x)
                        for o in self.observations)
        if len(angles) < 2:
            return 0.0

        # Largest gap on the circle; the arc actually covered is its complement.
        gaps = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]
        gaps.append(angles[0] + 2 * math.pi - angles[-1])
        return 2 * math.pi - max(gaps)


@dataclass(frozen=True)
class ConfirmationPolicy:
    """When a track is trustworthy enough to become a landmark.

    Every number here buys something and costs something, so each says which.
    """

    #: Total sightings. Cheap noise rejection; on its own it is the weak test,
    #: because a stationary robot accumulates it without learning anything.
    min_observations: int = 4

    #: Sightings from places at least `viewpoint_separation_m` apart. The strong
    #: test: this is what removes projection bias rather than just noise.
    min_distinct_viewpoints: int = 2
    viewpoint_separation_m: float = 0.6

    #: Refuse tracks whose own observations disagree by more than this. Direct
    #: evidence the ground-plane assumption is failing for that object.
    max_spread_m: float = 0.75

    #: The detector must have been sure at least once.
    min_confidence: float = 0.5

    #: Seconds without a sighting before a track is dropped. Long enough to
    #: survive an occlusion while the robot drives past a pillar; short enough
    #: that a moved object does not linger.
    stale_after_s: float = 30.0


@dataclass
class ConfirmationResult:
    confirmed: bool
    reasons: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.confirmed


def evaluate(track: Track, policy: ConfirmationPolicy,
             class_radius_m: float = 0.0) -> ConfirmationResult:
    """Decide whether a track earns a place on the map, and say why not.

    Returns the reasons either way. An operator asking "why is the chair not on
    the map" deserves "seen 6 times but only from one position" rather than
    silence, and that message is also the fastest way to debug a patrol route
    that never triangulates anything.

    `class_radius_m` widens the spread limit by the object's own size, and it
    is not optional bookkeeping - leaving it out produced a bug the demo patrol
    caught twice over.

    First a fixed 1 m association gate SPLIT the 3.2 m meeting table into two
    tracks. Widening the gate by the class radius fixed that and immediately
    broke the reception desk instead: with more observations now correctly
    merged into one track, the observations of a wide object spanned 1.5 m and
    tripped the 0.75 m spread limit, so a correctly-tracked table was rejected
    as "the views do not triangulate".

    Both limits are asking the same question - *could these observations be one
    object?* - and the honest answer depends on how big the object is. A cup
    and a dining table do not deserve the same radius in either test, and
    scaling one without the other just moves the failure.
    """
    reasons: List[str] = []

    if len(track) < policy.min_observations:
        reasons.append(
            f"seen {len(track)} times, needs {policy.min_observations}"
        )

    distinct = track.distinct_viewpoints(policy.viewpoint_separation_m)
    if distinct < policy.min_distinct_viewpoints:
        reasons.append(
            f"seen from {distinct} distinct viewpoint(s) at least "
            f"{policy.viewpoint_separation_m} m apart, needs "
            f"{policy.min_distinct_viewpoints} - more sightings from the same "
            f"spot would not help, the robot has to move"
        )

    if track.confidence < policy.min_confidence:
        reasons.append(
            f"best confidence {track.confidence:.2f} < {policy.min_confidence}"
        )

    spread = track.spread_m
    spread_limit = policy.max_spread_m + class_radius_m
    if spread > spread_limit:
        reasons.append(
            f"observations disagree by {spread:.2f} m (limit "
            f"{spread_limit:.2f} m = {policy.max_spread_m} m of tolerance plus "
            f"{class_radius_m:.2f} m of object); the views do not triangulate, "
            f"which usually means the object is not standing on the floor"
        )

    if reasons:
        return ConfirmationResult(False, reasons)
    return ConfirmationResult(True, [
        f"{len(track)} sightings from {distinct} viewpoints, "
        f"spread {spread:.2f} m, confidence {track.confidence:.2f}"
    ])


class Tracker:
    """Associates observations into tracks by label and proximity.

    Nearest-neighbour association with a gate. Not a Kalman filter and
    deliberately not: the objects being mapped are furniture, which does not
    move while being observed, so a motion model would add state and tuning for
    no gain. Moving objects are excluded from mapping entirely (`NEVER_MAP`),
    which is what makes the simple thing correct here rather than merely easy.
    """

    def __init__(self, gate_m: float = 1.0,
                 policy: Optional[ConfirmationPolicy] = None,
                 class_radius_m: Optional[Dict[str, float]] = None):
        #: Base association radius, sized to the projection error at typical
        #: detection range. Too large merges two chairs into one; too small
        #: splits one object into two tracks that each fail confirmation.
        self.gate_m = gate_m

        #: Per-class physical radius, ADDED to the base gate.
        #:
        #: A fixed gate was the first version and it was wrong in a way the
        #: demo patrol exposed: the 3.2 m meeting table split into two tracks
        #: 0.48 m apart, because a wide object seen from two sides gives ground
        #: points on two different parts of its edge, further apart than the
        #: 1 m gate. The gate is asking "could these be the same object", and
        #: the honest answer depends on how big the object is. A cup and a
        #: dining table do not deserve the same radius.
        self.class_radius_m = dict(class_radius_m or {})

        self.policy = policy or ConfirmationPolicy()
        self.tracks: List[Track] = []
        self._next_id = 1

    def gate_for(self, label: str) -> float:
        return self.gate_m + self.class_radius_m.get(label, 0.0)

    def update(self, observations: Sequence[Observation]) -> List[Track]:
        """Fold one frame's observations into the tracks. Returns those touched.

        Two rules, and the first was a bug found by a test:

        **A track accepts at most one observation per frame.** One physical
        object cannot produce two boxes in a single image once duplicates are
        suppressed, so two simultaneous detections are two different objects by
        definition. Without this, two chairs 2 m apart both fall inside a 3 m
        gate, merge into one track, and that track's median position sits in the
        empty floor between them — a confident landmark where nothing is.

        **Assignment is globally greedy, not first-come-first-served.** All
        candidate pairs within the gate are sorted by distance and taken best
        first. Matching in arrival order makes the result depend on whatever
        order the detector happened to emit its boxes in, which is a real source
        of run-to-run flapping.
        """
        # Snapshot positions once. Updating a track mid-frame would move the
        # target its own siblings are being matched against.
        candidates: List[Tuple[float, int, int]] = []
        positions = [t.position for t in self.tracks]

        for oi, observation in enumerate(observations):
            for ti, track in enumerate(self.tracks):
                # Label must match. Associating a detected chair with a tracked
                # table because they are close produces a track whose label is
                # whichever the detector said first.
                if track.label != observation.label:
                    continue
                x, y = positions[ti]
                distance = observation.distance_to(x, y)
                if distance <= self.gate_for(observation.label):
                    candidates.append((distance, oi, ti))

        # Tie-break on indices so equal distances resolve deterministically.
        candidates.sort()

        assigned_obs: Dict[int, Track] = {}
        taken_tracks: set = set()
        for _, oi, ti in candidates:
            if oi in assigned_obs or ti in taken_tracks:
                continue
            assigned_obs[oi] = self.tracks[ti]
            taken_tracks.add(ti)

        touched: List[Track] = []
        for oi, observation in enumerate(observations):
            track = assigned_obs.get(oi)
            if track is None:
                track = Track(label=observation.label, track_id=self._next_id)
                self._next_id += 1
                self.tracks.append(track)
            track.add(observation)
            touched.append(track)
        return touched

    def prune(self, now: float) -> List[Track]:
        """Drop tracks not seen recently. Returns the dropped ones."""
        alive, dropped = [], []
        for track in self.tracks:
            if now - track.last_stamp > self.policy.stale_after_s:
                dropped.append(track)
            else:
                alive.append(track)
        self.tracks = alive
        return dropped

    def _radius(self, label: str) -> float:
        return self.class_radius_m.get(label, 0.0)

    def confirmed(self) -> List[Track]:
        return [t for t in self.tracks
                if evaluate(t, self.policy, self._radius(t.label)).confirmed]

    def pending(self) -> List[Tuple[Track, ConfirmationResult]]:
        """Tracks that have not qualified yet, with the reason.

        Exposed because "what has the robot nearly decided" is the single most
        useful thing to look at when a patrol maps nothing.
        """
        out = []
        for track in self.tracks:
            result = evaluate(track, self.policy, self._radius(track.label))
            if not result.confirmed:
                out.append((track, result))
        return out

    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for track in self.confirmed():
            counts[track.label] = counts.get(track.label, 0) + 1
        return counts


def _median(values: List[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    middle = n // 2
    if n % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0
