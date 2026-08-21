"""Run a patrol and map what it finds, without ROS.

    python -m robot_perception.cli patrol         # the full pipeline
    python -m robot_perception.cli patrol --strict-off   # what happens without the policy
    python -m robot_perception.cli accuracy       # error against ground truth
    python -m robot_perception.cli bench          # detector latency on this machine

The default backend is synthetic, so this runs with no model, no GPU and no
network. `--backend yolo` swaps in a real detector if one is installed.
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import List

from robot_perception.backends.synthetic import (
    DEMO_SCENE,
    SyntheticDetector,
    observations_from,
)
from robot_perception.mapping import apply, make_tracker, propose
from robot_perception.patrol import PatrolSettings, demo_route, describe, sampled
from robot_perception.tracking import ConfirmationPolicy
from semantic_nav.demo_world import build_semantic_map


def _stdout_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _run_patrol(args, policy=None):
    settings = PatrolSettings()
    frames = sampled(demo_route(settings), settings.detect_every)

    detector = SyntheticDetector(false_positive_rate=args.false_positives)
    tracker = make_tracker(policy=policy)

    raw = 0
    for index, (x, y, yaw) in enumerate(frames):
        detector.look_from(x, y, yaw)
        detections = detector.detect()
        raw += len(detections)
        tracker.update(observations_from(detector, detections, stamp=float(index)))

    return settings, frames, raw, tracker


def _accuracy(proposals) -> List[float]:
    errors = []
    for proposal in proposals:
        nearest = min(DEMO_SCENE,
                      key=lambda o: math.hypot(o.x - proposal.x, o.y - proposal.y))
        errors.append(math.hypot(nearest.x - proposal.x, nearest.y - proposal.y))
    return errors


def cmd_patrol(args) -> int:
    policy = ConfirmationPolicy(
        min_observations=1, min_distinct_viewpoints=1,
        max_spread_m=99.0, min_confidence=0.0,
    ) if args.strict_off else None

    settings, frames, raw, tracker = _run_patrol(args, policy)
    semantic_map = build_semantic_map()
    before = len(semantic_map)

    print(f"patrol : {describe(frames if False else demo_route(settings), settings)}")
    print(f"frames : {len(frames)} detector frames, {raw} raw detections")
    print(f"tracks : {len(tracker.tracks)}")
    if args.strict_off:
        print("policy : DISABLED (--strict-off): anything seen once is mapped")
    print()

    outcome = propose(tracker.tracks, semantic_map, policy=policy)
    print(outcome.summary())

    added = apply(outcome, semantic_map)
    print(f"\nmap    : {before} hand-authored -> {len(semantic_map)} "
          f"({len(added)} perceived)")

    errors = _accuracy(outcome.proposals)
    if errors:
        print(f"error  : mean {sum(errors) / len(errors):.3f} m, "
              f"max {max(errors):.3f} m against ground truth")

    print("\nthe point of all this:")
    grounder_demo(semantic_map)
    return 0


def grounder_demo(semantic_map) -> None:
    """Show that a perceived landmark is now something you can talk to."""
    from semantic_nav.commands import ValidationError, validate
    from semantic_nav.grounding import Grounder

    grounder = Grounder(semantic_map, llm=None)
    for utterance in ["go to the chair near the kitchen",
                      "go to the potted plant near the kitchen"]:
        try:
            command = validate(grounder.ground(utterance), semantic_map)
            print(f"  '{utterance}'")
            print(f"{command.describe()}")
        except ValidationError as error:
            print(f"  '{utterance}' -> refused: {error}")


def cmd_accuracy(args) -> int:
    """Positional error against ground truth, and what the policy costs.

    Both numbers matter. Turning the confirmation policy off maps more objects
    and maps them worse, and the whole argument of this package is that the
    second effect outweighs the first.
    """
    rows = []
    for label, policy in [
        ("policy on", None),
        ("policy off", ConfirmationPolicy(min_observations=1,
                                          min_distinct_viewpoints=1,
                                          max_spread_m=99.0, min_confidence=0.0)),
    ]:
        _, _, raw, tracker = _run_patrol(args, policy)
        outcome = propose(tracker.tracks, build_semantic_map(), policy=policy,
                          max_new_landmarks=99)
        errors = _accuracy(outcome.proposals)
        ghosts = sum(1 for p in outcome.proposals
                     if min(math.hypot(o.x - p.x, o.y - p.y) for o in DEMO_SCENE) > 1.0)
        rows.append((label, len(outcome.proposals), ghosts,
                     sum(errors) / len(errors) if errors else float("nan"),
                     max(errors) if errors else float("nan")))

    print(f"{'':12} {'mapped':>7} {'ghosts':>7} {'mean err':>9} {'max err':>9}")
    print("-" * 48)
    for label, mapped, ghosts, mean, worst in rows:
        print(f"{label:12} {mapped:>7} {ghosts:>7} {mean:>8.3f} m {worst:>8.3f} m")
    print()
    print("A ghost is a mapped landmark more than 1 m from any real object.")
    print("Turning the policy off maps more and maps it worse. That trade is")
    print("the entire argument of this package.")
    return 0


def cmd_bench(args) -> int:
    from robot_perception.detection import load_backend

    backend = load_backend(args.backend)
    print(f"backend: {backend.name}")

    if args.backend in {"yolo", "ultralytics"} or backend.name.startswith("yolo"):
        from robot_perception.backends.yolo import benchmark
        for size in (640, 416, 320):
            backend.image_size = size
            result = benchmark(backend, frames=args.frames)
            print(f"  imgsz={size:4}  mean {result['mean_ms']:7.1f} ms  "
                  f"p95 {result['p95_ms']:7.1f} ms  {result['max_fps']:5.2f} fps")
        print()
        print("Compare against the camera: 15 Hz is 67 ms per frame. If the")
        print("detector is slower than that - and on a CPU it is - frames must")
        print("be SKIPPED rather than queued. A queued frame is grounded against")
        print("a stale pose, which smears landmarks along the robot's path and")
        print("presents as a mapping bug rather than a latency one.")
    else:
        import time
        detector = SyntheticDetector()
        detector.look_from(3.0, 3.0, 0.0)
        start = time.perf_counter()
        for _ in range(args.frames):
            detector.detect()
        elapsed = (time.perf_counter() - start) / args.frames * 1000
        print(f"  {elapsed:.3f} ms/frame (no model - this is pure geometry)")
    return 0


def main(argv=None) -> int:
    _stdout_utf8()
    parser = argparse.ArgumentParser(
        prog="robot_perception",
        description="build a semantic map from perception, without ROS")
    sub = parser.add_subparsers(dest="command", required=True)

    patrol = sub.add_parser("patrol", help="run a patrol and map what it finds")
    patrol.add_argument("--false-positives", type=float, default=0.05,
                        help="per-frame chance of a ghost detection")
    patrol.add_argument("--strict-off", action="store_true",
                        help="disable the confirmation policy, to show the cost")
    patrol.set_defaults(func=cmd_patrol)

    accuracy = sub.add_parser("accuracy", help="error against ground truth")
    accuracy.add_argument("--false-positives", type=float, default=0.05)
    accuracy.set_defaults(func=cmd_accuracy)

    bench = sub.add_parser("bench", help="detector latency on this machine")
    bench.add_argument("--backend", default="synthetic")
    bench.add_argument("--frames", type=int, default=10)
    bench.set_defaults(func=cmd_bench)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
