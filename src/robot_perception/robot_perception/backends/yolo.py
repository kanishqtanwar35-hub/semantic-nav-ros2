"""The real detector. Optional, lazily imported, and never touched by CI.

Nothing else in `robot_perception` imports this module. `detection.load_backend`
reaches for it and falls back to the synthetic backend if it is not installed,
which is what keeps the package installable, testable and runnable on a machine
with no ML stack at all.

**Why YOLOv8n specifically**, out of everything that could go here:

  * It runs on **CPU**. The target is a small indoor robot, and putting a
    discrete GPU on one to find chairs is not a trade anyone makes. Nothing in
    this repository requires a GPU at any point.
  * The nano weights are about **6 MB**. That is small enough to cache in CI
    without a cache being a load-bearing part of the build.
  * COCO's class list already contains most indoor furniture, so no
    fine-tuning is needed to demonstrate the pipeline. Fine-tuning on
    site-specific classes is the obvious next step and is deliberately not
    pretended to have been done.

**What this file is not.** It is not where the interesting engineering is. The
detector is the commodity part — swap it for a vendor SDK, an NPU runtime, or a
different model and everything downstream is unchanged, which is the entire
point of the backend protocol. The judgement lives in the projection, the
confirmation policy and the allowlist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from robot_perception.detection import INDOOR_CLASSES, Detection

LOGGER = logging.getLogger(__name__)

#: Inference resolution. 640 is the size the weights were trained at and gives
#: the published accuracy; 416 is roughly 2.4x faster on CPU and loses small and
#: distant objects first. Distant objects are exactly the ones whose ground
#: projection is least trustworthy anyway (`camera.ground_error_m` grows with
#: range), so dropping them early costs less than it appears to.
DEFAULT_IMAGE_SIZE = 640


@dataclass
class YoloDetector:
    """Ultralytics YOLO behind the `DetectorBackend` protocol."""

    min_confidence: float = 0.35
    weights: Optional[str] = None
    image_size: int = DEFAULT_IMAGE_SIZE
    #: Restrict to the indoor allowlist at the source. Filtering here rather
    #: than downstream means an aeroplane detection never enters the pipeline
    #: at all, instead of being carried around and discarded later by a check
    #: someone might remove.
    restrict_to_indoor: bool = True
    _model: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # Import inside the constructor, not at module scope. Constructing this
        # class is the point at which the caller has actually asked for a real
        # model, so it is the right place for the dependency to be required.
        from ultralytics import YOLO

        # 'yolov8n.pt' downloads on first use and is cached by ultralytics.
        # After that it runs fully offline, which matters for a robot on a site
        # network that does not reach the internet.
        self._model = YOLO(self.weights or "yolov8n.pt")
        LOGGER.info("loaded %s at imgsz=%d on CPU",
                    self.weights or "yolov8n.pt", self.image_size)

    @property
    def name(self) -> str:
        return f"yolo({self.weights or 'yolov8n'})"

    def detect(self, image) -> List[Detection]:
        """Run inference on one frame.

        `device="cpu"` is explicit rather than left to autodetect. On a machine
        that happens to have CUDA, autodetect silently produces latency numbers
        that the target hardware cannot reproduce — and a perception rate tuned
        against those numbers is a robot that falls behind its own camera in the
        field.
        """
        results = self._model.predict(
            image,
            conf=self.min_confidence,
            imgsz=self.image_size,
            device="cpu",
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        names = getattr(result, "names", {}) or {}
        detections: List[Detection] = []

        for box in boxes:
            label = names.get(int(box.cls[0]), str(int(box.cls[0])))
            if self.restrict_to_indoor and label not in INDOOR_CLASSES:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            confidence = float(box.conf[0])
            try:
                detections.append(Detection(label, confidence, x1, y1, x2, y2))
            except ValueError as error:
                # A malformed box is dropped rather than allowed to abort the
                # frame. One bad row must not blind the robot for a cycle.
                LOGGER.warning("discarding a malformed box: %s", error)

        return detections


def benchmark(detector: "YoloDetector", frames: int = 20,
              width: int = 640, height: int = 480) -> dict:
    """Measure inference latency on this machine, on synthetic frames.

    Reported rather than assumed. The perception rate has to be chosen from the
    hardware's real throughput: if the detector cannot keep up with the camera,
    frames queue, the pose attached to each frame goes stale, and detections get
    grounded against where the robot *was* — which shows up as a smear of
    landmarks along the robot's path rather than as a latency problem.
    """
    import time

    import numpy as np

    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)

    detector.detect(image)      # warm-up; the first call builds the graph

    times = []
    for _ in range(frames):
        start = time.perf_counter()
        detector.detect(image)
        times.append((time.perf_counter() - start) * 1000.0)

    times.sort()
    return {
        "frames": frames,
        "mean_ms": sum(times) / len(times),
        "p50_ms": times[len(times) // 2],
        "p95_ms": times[int(len(times) * 0.95) - 1],
        "max_fps": 1000.0 / (sum(times) / len(times)),
        "image_size": detector.image_size,
    }
