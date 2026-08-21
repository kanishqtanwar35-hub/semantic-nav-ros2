"""Detections, and the backend interface that keeps the model swappable.

**Why there is an interface here at all.** A perception package that imports
`ultralytics` at module scope has three problems, and only the first is obvious:

1. CI has to download hundreds of megabytes of weights to run any test.
2. Every test of the *logic* — association, confirmation, grounding — is now a
   test of the *model* too, so a failure could be either, and the suite becomes
   slow enough that people stop running it.
3. The package can never be run on hardware whose detector is different. On a
   real robot the detector is frequently a vendor SDK, an NPU runtime, or a
   ROS node someone else owns.

So the detector is a protocol with three implementations: a scripted stub that
CI uses, a deterministic synthetic detector for the simulator, and a real
model that is imported lazily and only when actually asked for. The same
argument as the lazy Nav2 import in `semantic_nav/nav_client.py`, applied to ML.

**The ground-contact point.** A `Detection` reports `ground_pixel` as the
BOTTOM-CENTRE of its box, not the centre. The bottom edge is where the object
meets the floor; the centre is halfway up it and projects systematically too
far away on every single frame. A consistent bias is worse than noise, because
averaging over frames removes noise and leaves bias exactly where it was.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

#: The subset of COCO classes that mean something to an indoor facility robot.
#: Everything else a general detector emits — aeroplane, giraffe, surfboard —
#: is noise in this context, and filtering at the source keeps it out of the
#: map rather than relying on a downstream check that someone will forget.
#:
#: The value is the height above the floor of the point the detector typically
#: boxes, used by `camera.ground_error_m` to bound the projection error. These
#: are rough by nature and stated as estimates rather than dressed up as
#: measurements.
INDOOR_CLASSES: Dict[str, float] = {
    "person": 0.0,
    "chair": 0.0,
    "couch": 0.0,
    "bed": 0.0,
    "dining table": 0.0,
    "potted plant": 0.0,
    "refrigerator": 0.0,
    "sink": 0.0,
    "toilet": 0.0,
    "tv": 0.60,
    "laptop": 0.72,
    "keyboard": 0.72,
    "mouse": 0.72,
    "book": 0.72,
    "cup": 0.72,
    "bottle": 0.72,
    "backpack": 0.0,
    "suitcase": 0.0,
    "clock": 1.80,
}

#: Classes that are never mapped as landmarks, whatever the detector says.
#: A person is the most important thing a robot can detect and the least
#: appropriate thing to write into a persistent map — they move, and a map
#: that records where somebody stood on Tuesday is both useless and a privacy
#: problem. They are still published for the safety layer to react to.
NEVER_MAP = frozenset({"person"})


@dataclass(frozen=True)
class Detection:
    """One box from one frame."""

    label: str
    confidence: float
    #: Pixel box, (x1, y1, x2, y2), with y increasing downward.
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(
                f"box corners are inverted: ({self.x1}, {self.y1}) to "
                f"({self.x2}, {self.y2}); backends differ on corner order and "
                f"silently accepting this mirrors every detection"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence {self.confidence} is not a probability")

    @property
    def ground_pixel(self) -> Tuple[float, float]:
        """Bottom-centre: where the object meets the floor."""
        return ((self.x1 + self.x2) / 2.0, self.y2)

    @property
    def centre(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def mappable(self) -> bool:
        return self.label in INDOOR_CLASSES and self.label not in NEVER_MAP

    @property
    def assumed_height_m(self) -> float:
        return INDOOR_CLASSES.get(self.label, 0.0)

    def iou(self, other: "Detection") -> float:
        """Intersection over union. Used to suppress duplicate boxes.

        Returns 0.0 for non-overlapping boxes rather than raising, because a
        detector legitimately emits boxes that do not overlap and the caller
        should not have to special-case it.
        """
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        intersection = (ix2 - ix1) * (iy2 - iy1)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0

    def touches_edge(self, width: int, height: int, margin: float = 2.0) -> bool:
        """Whether the box runs off the side of the frame.

        A truncated box has a truncated bottom edge, so its ground-contact point
        is wrong — the object continues below what the camera saw. These get
        detected and reported but never grounded, which is a distinction most
        pipelines skip and then wonder why objects at the frame edge land in the
        wrong place.
        """
        return (self.x1 <= margin or self.y1 <= margin
                or self.x2 >= width - margin or self.y2 >= height - margin)


class DetectorBackend(Protocol):
    """What any detector must provide.

    Deliberately minimal. Anything richer — segmentation masks, keypoints,
    tracking ids — belongs in a subclass, because requiring it here would rule
    out the simplest backends and those are the ones CI depends on.
    """

    @property
    def name(self) -> str: ...

    def detect(self, image) -> List[Detection]: ...


@dataclass
class ScriptedDetector:
    """Replays a fixed list of detections per frame. This is what CI runs.

    Not a mock in the testing sense — it is a real backend that happens to be
    driven by a script, so the whole pipeline downstream of it is exercised
    exactly as it would be with a real model. The one thing it cannot test is
    the model, which is the point.
    """

    frames: List[List[Detection]] = field(default_factory=list)
    calls: int = 0

    @property
    def name(self) -> str:
        return "scripted"

    def detect(self, image=None) -> List[Detection]:
        if self.calls >= len(self.frames):
            # Running past the end returns nothing rather than raising. A
            # perception loop that crashes when the recording ends is useless
            # for replay, which is the main reason this backend exists.
            self.calls += 1
            return []
        result = self.frames[self.calls]
        self.calls += 1
        return list(result)


def non_max_suppression(detections: Sequence[Detection],
                        iou_threshold: float = 0.5) -> List[Detection]:
    """Drop lower-confidence boxes that overlap a kept one.

    Most detectors do this internally, and this exists for the ones that do not
    and for merging the output of two backends. Suppression is per-label: a
    chair box overlapping a person box is two real objects, and suppressing
    across labels would delete one of them.
    """
    kept: List[Detection] = []
    for candidate in sorted(detections, key=lambda d: -d.confidence):
        if all(candidate.iou(k) < iou_threshold
               for k in kept if k.label == candidate.label):
            kept.append(candidate)
    return kept


def filter_detections(detections: Sequence[Detection],
                      min_confidence: float,
                      image_width: Optional[int] = None,
                      image_height: Optional[int] = None,
                      min_area_px: float = 400.0) -> List[Detection]:
    """Apply the cheap rejections before anything expensive happens.

    `min_area_px` at 400 is a 20x20 box. Below that the bottom edge is only
    known to within a few pixels, and at the bottom of a 480-row frame a few
    pixels of vertical error is tens of centimetres on the floor — the
    projection error grows as roughly 1/tan, so small distant boxes are exactly
    where the geometry is least trustworthy.
    """
    out = []
    for detection in detections:
        if detection.confidence < min_confidence:
            continue
        if detection.area < min_area_px:
            continue
        if (image_width is not None and image_height is not None
                and detection.touches_edge(image_width, image_height)):
            continue
        out.append(detection)
    return out


def load_backend(name: str = "auto", min_confidence: float = 0.35,
                 weights: Optional[str] = None) -> DetectorBackend:
    """Resolve a backend by name, importing heavy dependencies lazily.

    `auto` prefers a real model and falls back to the synthetic one, saying so.
    Silent fallback would be worse: an operator would see a robot mapping
    nothing and have no idea the detector never loaded.
    """
    if name in {"scripted", "none"}:
        return ScriptedDetector()

    if name in {"auto", "yolo", "ultralytics"}:
        try:
            from robot_perception.backends.yolo import YoloDetector
            return YoloDetector(min_confidence=min_confidence, weights=weights)
        except Exception as error:                        # noqa: BLE001
            if name != "auto":
                raise
            import logging
            logging.getLogger(__name__).warning(
                "no real detector available (%s: %s); falling back to the "
                "synthetic backend. Detections will be geometric, not learned.",
                type(error).__name__, error,
            )

    from robot_perception.backends.synthetic import SyntheticDetector
    return SyntheticDetector(min_confidence=min_confidence)


def summarise(detections: Sequence[Detection]) -> str:
    """One line for a log. Counts by label, highest confidence first."""
    if not detections:
        return "no detections"
    counts: Dict[str, int] = {}
    best: Dict[str, float] = {}
    for d in detections:
        counts[d.label] = counts.get(d.label, 0) + 1
        best[d.label] = max(best.get(d.label, 0.0), d.confidence)
    ordered = sorted(counts, key=lambda label: -best[label])
    return ", ".join(
        f"{counts[label]}x {label} ({best[label]:.2f})" for label in ordered
    )


def bearing_of(detection: Detection, image_width: int,
               horizontal_fov: float) -> float:
    """Approximate bearing to a detection, in radians, positive to the left.

    Cheap enough to run on every frame, and it is what the safety layer wants:
    "is there a person in the forward arc" needs an angle, not a floor
    coordinate, and unlike the floor coordinate it does not depend on the
    ground-plane assumption holding.
    """
    u, _ = detection.centre
    return -math.atan((u - image_width / 2.0)
                      / ((image_width / 2.0) / math.tan(horizontal_fov / 2.0)))
