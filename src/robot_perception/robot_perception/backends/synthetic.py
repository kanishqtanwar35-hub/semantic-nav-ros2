"""A detector with no model in it.

Given a scene of known boxes and the robot's pose, this projects each object
back through the camera model and emits the pixel box a real detector *would*
have produced. It is the forward geometry run backwards.

**Why that is worth building rather than mocking.** Three things fall out of it
that a mock cannot give you:

1. **The whole pipeline runs with no weights, no GPU and no network.** CI
   exercises projection, association, confirmation, naming and mapping end to
   end in milliseconds.
2. **It closes a loop.** The scene positions are ground truth, so the error at
   the far end of the pipeline is measurable rather than eyeballed. A change
   that quietly degrades the geometry shows up as a number.
3. **It can be made to lie in specific ways.** Real detectors miss frames, wobble
   their boxes, and occasionally emit something that is not there. Each of those
   is a parameter here, so the confirmation policy is tested against the failure
   modes it exists for instead of against perfect input.

The noise is deterministic — derived from a hash of the object and the frame
index, not from a global RNG — so a failing test reproduces exactly. A
randomised detector with no seed produces a suite that fails once a fortnight
for no reason, and the usual response is to delete the test.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from robot_perception.camera import (
    CameraExtrinsics,
    CameraIntrinsics,
    map_to_base,
    world_to_pixel,
)
from robot_perception.detection import Detection


@dataclass(frozen=True)
class SceneObject:
    """A real thing at a known place. Ground truth for the simulator."""

    label: str
    x: float
    y: float
    #: Footprint half-extent in metres. Objects are treated as squares in plan;
    #: the detector only ever sees a 2D box anyway, so modelling orientation
    #: would add parameters without changing what comes out.
    radius_m: float = 0.3
    height_m: float = 0.9
    #: Probability the detector reports it when it is plainly in view. Below 1.0
    #: on purpose: a policy that only works with a perfect detector is not a
    #: policy.
    detectability: float = 0.9


#: The office from `semantic_nav.demo_world`, plus the furniture a patrol is
#: meant to discover. The three tables match the rectangles in `build_grid`
#: exactly, so a detection of the reception desk lands where the occupancy grid
#: says the reception desk is - which is what makes the round-trip error
#: meaningful rather than self-confirming.
DEMO_SCENE: List[SceneObject] = [
    SceneObject("dining table", 4.5, 0.95, radius_m=1.1, height_m=0.75),
    SceneObject("dining table", 3.0, 7.0, radius_m=1.6, height_m=0.75),
    SceneObject("tv", 9.8, 2.7, radius_m=0.4, height_m=0.90),
    SceneObject("chair", 9.0, 6.2, radius_m=0.3, height_m=0.85),
    SceneObject("chair", 2.2, 5.9, radius_m=0.3, height_m=0.85),
    SceneObject("potted plant", 10.8, 7.6, radius_m=0.3, height_m=0.80),
    SceneObject("potted plant", 1.0, 3.4, radius_m=0.3, height_m=0.80),
    SceneObject("refrigerator", 11.2, 8.2, radius_m=0.35, height_m=1.60),
]


def _jitter(seed_parts: Tuple, spread: float) -> float:
    """Deterministic pseudo-noise in [-spread, +spread].

    A hash of the inputs rather than an RNG, so each frame is independently
    reproducible without threading a generator through the call chain.

    **blake2b, not the builtin `hash()`.** The first version used `hash()` and
    was not reproducible at all: Python randomises string hashing per process
    unless PYTHONHASHSEED is set, so every run produced different noise and the
    accuracy assertions drifted across runs.

    What makes that worth writing down is that the determinism test *passed*.
    It compared two detectors inside one process, where the hash seed is shared.
    Cross-process variation is invisible to a within-process check, and the
    symptom was a threshold that failed roughly one run in three - exactly the
    flaky test the RRT planner in `robot_core` already has a seed to prevent.
    The same mistake, one package over.
    """
    digest = hashlib.blake2b(repr(seed_parts).encode("utf-8"), digest_size=8)
    value = int.from_bytes(digest.digest(), "big") / float(1 << 64)
    return (value * 2.0 - 1.0) * spread


@dataclass
class SyntheticDetector:
    """Projects a known scene into plausible detections."""

    scene: List[SceneObject] = field(default_factory=lambda: list(DEMO_SCENE))
    intrinsics: CameraIntrinsics = field(
        default_factory=lambda: CameraIntrinsics.from_fov(640, 480, 1.089))
    extrinsics: CameraExtrinsics = field(default_factory=CameraExtrinsics)
    min_confidence: float = 0.35

    #: Pixels of box jitter. A real detector's box breathes by a few pixels
    #: frame to frame, and at the bottom of the image a few pixels is tens of
    #: centimetres on the floor.
    box_noise_px: float = 3.0
    #: Chance per frame of inventing an object that is not there. Low, but not
    #: zero, because the confirmation policy's job is to survive exactly this.
    false_positive_rate: float = 0.0
    #: How far the detector can see. Beyond this, boxes are too small for the
    #: bottom edge to be located usefully.
    max_range_m: float = 6.0

    frame: int = 0
    #: Set by `look_from`. Kept as state so `detect()` matches the
    #: DetectorBackend protocol, which takes only an image.
    pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def name(self) -> str:
        return "synthetic"

    def look_from(self, x: float, y: float, yaw: float) -> None:
        self.pose = (x, y, yaw)

    def detect(self, image=None) -> List[Detection]:
        """Emit what a detector would report from the current pose."""
        detections: List[Detection] = []
        robot_x, robot_y, robot_yaw = self.pose

        for index, obj in enumerate(self.scene):
            detection = self._project(obj, index, robot_x, robot_y, robot_yaw)
            if detection is not None:
                detections.append(detection)

        if self.false_positive_rate > 0:
            ghost = self._ghost(robot_x, robot_y, robot_yaw)
            if ghost is not None:
                detections.append(ghost)

        self.frame += 1
        return detections

    # -- internals ----------------------------------------------------------

    def _project(self, obj: SceneObject, index: int,
                 robot_x: float, robot_y: float, robot_yaw: float
                 ) -> Optional[Detection]:
        base = map_to_base((obj.x, obj.y), robot_x, robot_y, robot_yaw)
        range_m = math.hypot(base[0], base[1])
        if range_m > self.max_range_m or base[0] <= 0.0:
            return None

        # Bottom-centre is the ground contact point; top-centre gives the box
        # its height. Both must be on the sensor or the box is truncated, and a
        # truncated bottom edge is a wrong ground point.
        bottom = world_to_pixel(self.intrinsics, self.extrinsics, base, 0.0)
        top = world_to_pixel(self.intrinsics, self.extrinsics, base, obj.height_m)
        if bottom is None or top is None:
            return None

        # Half-width in pixels from the object's physical radius at this range.
        half_width_px = (self.intrinsics.fx * obj.radius_m) / max(range_m, 1e-6)

        noise = self.box_noise_px
        x1 = bottom[0] - half_width_px + _jitter((index, self.frame, "x1"), noise)
        x2 = bottom[0] + half_width_px + _jitter((index, self.frame, "x2"), noise)
        y1 = top[1] + _jitter((index, self.frame, "y1"), noise)
        y2 = bottom[1] + _jitter((index, self.frame, "y2"), noise)
        if x2 <= x1 or y2 <= y1:
            return None

        # Missed frames. Deterministic per (object, frame) so a replay is exact.
        if _jitter((index, self.frame, "seen"), 0.5) + 0.5 > obj.detectability:
            return None

        # Confidence falls with range and with being off-centre, which is what
        # real detectors do and what makes "best look" a meaningful statistic.
        centrality = 1.0 - abs(bottom[0] - self.intrinsics.cx) / self.intrinsics.cx
        confidence = 0.95 * (1.0 - 0.55 * range_m / self.max_range_m) \
            * (0.75 + 0.25 * max(0.0, centrality))
        confidence = min(0.99, max(0.0, confidence
                                   + _jitter((index, self.frame, "c"), 0.04)))
        if confidence < self.min_confidence:
            return None

        return Detection(obj.label, confidence, x1, y1, x2, y2)

    def _ghost(self, robot_x: float, robot_y: float,
               robot_yaw: float) -> Optional[Detection]:
        """A detection of something that is not there.

        Reflections in glass partitions, a chair on a monitor in a video call, a
        pattern in the carpet. The point of generating them is that the
        confirmation policy should reject them: a ghost appears from one
        viewpoint and vanishes from the next, so it never accumulates the
        distinct viewpoints confirmation requires.
        """
        if _jitter((self.frame, "ghost"), 0.5) + 0.5 > self.false_positive_rate:
            return None

        u = self.intrinsics.cx + _jitter((self.frame, "gu"), 200.0)
        v = self.intrinsics.cy + 120 + _jitter((self.frame, "gv"), 80.0)
        if not self.intrinsics.contains(u, v):
            return None
        size = 60.0
        return Detection("chair", 0.55, u - size / 2, v - size, u + size / 2, v)


def observations_from(detector: SyntheticDetector,
                      detections: Sequence[Detection],
                      stamp: float = 0.0):
    """Ground a frame's detections into map-frame observations.

    Convenience for tests and the CLI. The ROS node does the same thing but
    reads the pose from TF instead of from the detector.
    """
    from robot_perception.camera import base_to_map, detection_to_ground
    from robot_perception.tracking import Observation

    robot_x, robot_y, robot_yaw = detector.pose
    out = []
    for detection in detections:
        if not detection.mappable:
            continue
        u, v = detection.ground_pixel
        hit = detection_to_ground(detector.intrinsics, detector.extrinsics, u, v)
        if hit is None:
            continue
        x, y = base_to_map(hit.as_tuple(), robot_x, robot_y, robot_yaw)
        out.append(Observation(
            label=detection.label, confidence=detection.confidence,
            x=x, y=y, from_x=robot_x, from_y=robot_y,
            stamp=stamp, range_m=hit.range_m,
        ))
    return out
