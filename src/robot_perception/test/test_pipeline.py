"""End to end: a patrol, through the policy, into the map.

These are the tests that pin the numbers the README quotes. If the confirmation
policy stops paying for itself, or the positional error degrades, they fail here
rather than in a demo.
"""

import math

import pytest

from robot_perception.backends.synthetic import (
    DEMO_SCENE,
    SceneObject,
    SyntheticDetector,
    observations_from,
)
from robot_perception.detection import (
    INDOOR_CLASSES,
    NEVER_MAP,
    Detection,
    ScriptedDetector,
    bearing_of,
    filter_detections,
    non_max_suppression,
    summarise,
)
from robot_perception.mapping import (
    PERCEIVED_CATEGORY,
    apply,
    make_tracker,
    name_for,
    perceived,
    propose,
)
from robot_perception.patrol import (
    PatrolSettings,
    demo_route,
    interpolate,
    path_length,
    sampled,
    scan_in_place,
)
from robot_perception.tracking import ConfirmationPolicy
from semantic_nav.demo_world import build_semantic_map


def box(label="chair", conf=0.8, x1=100, y1=200, x2=160, y2=300):
    return Detection(label, conf, x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def test_the_ground_pixel_is_the_bottom_edge_not_the_centre():
    """The centre is halfway up the object and projects systematically too far
    away on EVERY frame. A consistent bias survives averaging; noise does not."""
    d = box(x1=100, y1=200, x2=160, y2=300)
    assert d.ground_pixel == (130.0, 300.0)
    assert d.centre == (130.0, 250.0)


def test_inverted_corners_are_rejected():
    """Backends differ on corner order, and silently accepting an inverted box
    mirrors every detection."""
    with pytest.raises(ValueError, match="inverted"):
        Detection("chair", 0.8, 200, 300, 100, 200)


def test_a_confidence_outside_zero_to_one_is_rejected():
    with pytest.raises(ValueError):
        Detection("chair", 1.4, 0, 0, 10, 10)


def test_people_are_detected_but_never_mapped():
    """The most important thing a robot can detect and the least appropriate
    thing to write into a persistent map."""
    assert "person" in INDOOR_CLASSES
    assert "person" in NEVER_MAP
    assert not box(label="person").mappable
    assert box(label="chair").mappable


def test_classes_outside_the_allowlist_are_not_mappable():
    assert not box(label="giraffe").mappable


def test_iou_of_a_box_with_itself_is_one():
    assert box().iou(box()) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    assert box(x1=0, y1=0, x2=10, y2=10).iou(box(x1=50, y1=50, x2=60, y2=60)) == 0.0


def test_suppression_is_per_label():
    """A chair box overlapping a person box is two real objects. Suppressing
    across labels would delete one of them."""
    chair = Detection("chair", 0.9, 0, 0, 100, 100)
    person = Detection("person", 0.8, 5, 5, 105, 105)
    assert len(non_max_suppression([chair, person])) == 2


def test_suppression_keeps_the_most_confident_duplicate():
    strong = Detection("chair", 0.9, 0, 0, 100, 100)
    weak = Detection("chair", 0.4, 5, 5, 105, 105)
    kept = non_max_suppression([strong, weak])
    assert kept == [strong]


def test_edge_boxes_are_filtered():
    """A truncated box has a truncated bottom edge, so its ground contact point
    is wrong - the object continues below what the camera saw."""
    edge = Detection("chair", 0.9, 0, 200, 60, 479)
    assert edge.touches_edge(640, 480)
    assert filter_detections([edge], 0.3, 640, 480) == []


def test_tiny_boxes_are_filtered():
    """Below ~20x20 the bottom edge is known to a few pixels, and a few pixels
    at the bottom of the frame is tens of centimetres on the floor."""
    tiny = Detection("chair", 0.9, 300, 300, 310, 310)
    assert filter_detections([tiny], 0.3) == []


def test_low_confidence_is_filtered():
    assert filter_detections([box(conf=0.1)], 0.35) == []


def test_summarise_orders_by_confidence():
    text = summarise([box("chair", 0.5), box("tv", 0.9), box("chair", 0.4)])
    assert text.startswith("1x tv")
    assert "2x chair" in text


def test_summarise_of_nothing():
    assert summarise([]) == "no detections"


def test_bearing_is_positive_to_the_left():
    left = Detection("chair", 0.8, 50, 200, 110, 300)
    right = Detection("chair", 0.8, 530, 200, 590, 300)
    assert bearing_of(left, 640, 1.089) > 0
    assert bearing_of(right, 640, 1.089) < 0


def test_the_scripted_backend_replays_and_then_goes_quiet():
    """Running past the end returns nothing rather than raising. A perception
    loop that crashes when a recording ends is useless for replay."""
    detector = ScriptedDetector(frames=[[box()], []])
    assert len(detector.detect()) == 1
    assert detector.detect() == []
    assert detector.detect() == []


# ---------------------------------------------------------------------------
# Patrol
# ---------------------------------------------------------------------------

def test_interpolation_faces_the_direction_of_travel():
    """A differential-drive robot points where it is going. Interpolating yaw
    independently produces a crab-walk no real base performs, and the camera
    views places the robot never looked."""
    poses = list(interpolate((0, 0, 0.0), (0, 5, 3.0), 1.0))
    assert all(p[2] == pytest.approx(math.pi / 2) for p in poses)


def test_interpolation_reaches_the_endpoint():
    poses = list(interpolate((0, 0, 0), (3, 4, 0), 0.5))
    assert poses[-1][0] == pytest.approx(3.0)
    assert poses[-1][1] == pytest.approx(4.0)


def test_a_scan_does_not_translate():
    """Rotation changes what is in view; it does not triangulate anything,
    because the camera has not moved."""
    poses = list(scan_in_place((2.0, 3.0, 0.0), math.pi, math.radians(20)))
    assert all((p[0], p[1]) == (2.0, 3.0) for p in poses)
    assert len({round(p[2], 3) for p in poses}) > 5


def test_the_step_follows_from_speed_and_frame_rate():
    settings = PatrolSettings(speed_mps=0.45, camera_hz=15.0)
    assert settings.step_m == pytest.approx(0.03)


def test_the_demo_route_covers_both_halves_of_the_building():
    poses = demo_route()
    assert any(y > 5.0 for _, y, _ in poses)      # north of the divider
    assert any(y < 4.0 for _, y, _ in poses)      # south of it
    assert path_length(poses) > 20.0


def test_the_demo_route_stays_inside_the_building():
    for x, y, _ in demo_route():
        assert 0.2 <= x <= 11.8
        assert 0.2 <= y <= 8.8


def test_sampling_thins_the_route():
    poses = demo_route()
    assert len(sampled(poses, 8)) == pytest.approx(len(poses) / 8, rel=0.05)


# ---------------------------------------------------------------------------
# The whole pipeline
# ---------------------------------------------------------------------------

def run_patrol(false_positives=0.05, policy=None):
    settings = PatrolSettings()
    frames = sampled(demo_route(settings), settings.detect_every)
    detector = SyntheticDetector(false_positive_rate=false_positives)
    tracker = make_tracker(policy=policy)
    raw = 0
    for index, (x, y, yaw) in enumerate(frames):
        detector.look_from(x, y, yaw)
        detections = detector.detect()
        raw += len(detections)
        tracker.update(observations_from(detector, detections, stamp=float(index)))
    return tracker, raw, len(frames)


def errors_against_truth(proposals):
    return [min(math.hypot(o.x - p.x, o.y - p.y) for o in DEMO_SCENE)
            for p in proposals]


@pytest.fixture(scope="module")
def patrol_result():
    tracker, raw, frames = run_patrol()
    outcome = propose(tracker.tracks, build_semantic_map())
    return tracker, outcome, raw, frames


def test_the_patrol_actually_detects_things(patrol_result):
    _, _, raw, frames = patrol_result
    assert frames > 100
    assert raw > 50


def test_the_patrol_maps_several_objects(patrol_result):
    _, outcome, _, _ = patrol_result
    assert len(outcome.proposals) >= 4


def test_mapped_positions_are_accurate(patrol_result):
    """The number the README quotes. Ground truth is known by construction,
    so this is measured rather than eyeballed."""
    _, outcome, _, _ = patrol_result
    errors = errors_against_truth(outcome.proposals)
    # Measured: mean 0.067 m, max 0.161 m, and identical on every run now that
    # the detector's noise no longer depends on PYTHONHASHSEED. The bounds
    # carry headroom so a small policy change does not fail the build, but they
    # are tight enough that a broken projection would.
    assert max(errors) < 0.40
    assert sum(errors) / len(errors) < 0.15


def test_no_ghost_survives_the_policy(patrol_result):
    """A ghost is a mapped landmark more than a metre from any real object.
    Ghosts appear from one viewpoint and vanish from the next, so they never
    accumulate the distinct viewpoints confirmation requires."""
    _, outcome, _, _ = patrol_result
    assert all(e < 1.0 for e in errors_against_truth(outcome.proposals))


def test_every_refusal_says_why(patrol_result):
    _, outcome, _, _ = patrol_result
    assert outcome.rejected
    assert all(":" in r for r in outcome.rejected)


def test_the_policy_pays_for_itself():
    """The headline comparison. Disabling confirmation maps roughly twice as
    many objects, admits ghosts, and makes the mean error several times worse.
    If that trade ever inverts, this package has no argument left."""
    off = ConfirmationPolicy(min_observations=1, min_distinct_viewpoints=1,
                             max_spread_m=99.0, min_confidence=0.0)

    strict_tracker, _, _ = run_patrol()
    loose_tracker, _, _ = run_patrol(policy=off)

    strict = propose(strict_tracker.tracks, build_semantic_map(),
                     max_new_landmarks=99)
    loose = propose(loose_tracker.tracks, build_semantic_map(), policy=off,
                    max_new_landmarks=99)

    strict_errors = errors_against_truth(strict.proposals)
    loose_errors = errors_against_truth(loose.proposals)

    assert len(loose.proposals) > len(strict.proposals)
    assert sum(loose_errors) / len(loose_errors) > \
        2 * sum(strict_errors) / len(strict_errors)
    assert max(loose_errors) > 1.0      # at least one ghost gets through
    assert max(strict_errors) < 1.0     # none do with the policy on


# ---------------------------------------------------------------------------
# Mapping and the allowlist
# ---------------------------------------------------------------------------

def test_perceived_landmarks_are_tagged_as_such(patrol_result):
    """"Which parts of this map did a model write" has to be answerable with a
    filter, not from memory."""
    _, outcome, _, _ = patrol_result
    semantic_map = build_semantic_map()
    apply(outcome, semantic_map)
    assert len(perceived(semantic_map)) == len(outcome.proposals)
    assert all(lm.category == PERCEIVED_CATEGORY for lm in perceived(semantic_map))


def test_perception_never_moves_a_hand_authored_landmark(patrol_result):
    """A detector that relocates the charging dock strands the robot with a flat
    battery. `apply` is additive by construction - there is no code in it that
    edits an existing landmark."""
    _, outcome, _, _ = patrol_result
    semantic_map = build_semantic_map()
    before = {lm.name: (lm.x, lm.y) for lm in semantic_map}

    apply(outcome, semantic_map)

    for name, position in before.items():
        landmark = semantic_map.get(name)
        assert (landmark.x, landmark.y) == position


def test_a_person_is_never_proposed():
    tracker = make_tracker()
    from robot_perception.tracking import Observation
    for i in range(10):
        tracker.update([Observation("person", 0.95, 5.0, 3.0, float(i), 0.0,
                                    stamp=i)])
    outcome = propose(tracker.tracks, build_semantic_map())
    assert outcome.proposals == []
    assert any("excluded from mapping by policy" in r for r in outcome.rejected)


def test_a_class_outside_the_allowlist_is_never_proposed():
    from robot_perception.tracking import Observation
    tracker = make_tracker()
    for i in range(10):
        tracker.update([Observation("giraffe", 0.99, 5.0, 3.0, float(i), 0.0,
                                    stamp=i)])
    outcome = propose(tracker.tracks, build_semantic_map())
    assert outcome.proposals == []
    assert any("allowlist" in r for r in outcome.rejected)


def test_the_landmark_budget_is_enforced():
    """A detector stuck in a failure mode emits hundreds of boxes a second."""
    from robot_perception.tracking import Observation
    tracker = make_tracker()
    for cluster in range(30):
        for i in range(6):
            tracker.update([Observation("chair", 0.9, cluster * 3.0, 1.0,
                                        float(i) * 0.8, 0.0, stamp=i)])
    outcome = propose(tracker.tracks, build_semantic_map(), max_new_landmarks=5)
    assert len(outcome.proposals) == 5
    assert any("budget" in r for r in outcome.rejected)


def test_names_are_anchored_to_hand_authored_landmarks():
    """"chair 3" is unusable - nobody can say which chair they mean."""
    semantic_map = build_semantic_map()
    name, near = name_for("chair", 9.0, 6.5, semantic_map, [])
    assert near == "kitchen"
    assert name == "chair near the kitchen"


def test_names_do_not_collide():
    semantic_map = build_semantic_map()
    first, _ = name_for("chair", 9.0, 6.5, semantic_map, [])
    second, _ = name_for("chair", 9.2, 6.6, semantic_map, [first])
    assert first != second


def test_a_perceived_landmark_never_anchors_another_name():
    """Naming a perceived chair after another perceived chair compounds one
    model's error into the name of the next thing."""
    semantic_map = build_semantic_map()
    from robot_core.semantic_map import Landmark
    semantic_map.add(Landmark("chair near the kitchen", PERCEIVED_CATEGORY,
                              9.0, 6.5))
    name, near = name_for("potted plant", 9.1, 6.6, semantic_map, [])
    assert near == "kitchen"
    assert "chair" not in name


def test_a_duplicate_of_an_existing_landmark_is_refused():
    from robot_core.semantic_map import Landmark
    from robot_perception.tracking import Observation

    semantic_map = build_semantic_map()
    semantic_map.add(Landmark("chair near the kitchen", PERCEIVED_CATEGORY,
                              9.0, 6.2, attributes={"class": "chair"}))

    tracker = make_tracker()
    for i in range(6):
        tracker.update([Observation("chair", 0.9, 9.05, 6.25, float(i) * 0.8,
                                    5.0, stamp=i)])
    outcome = propose(tracker.tracks, semantic_map)
    assert outcome.proposals == []
    assert any("already mapped" in r for r in outcome.rejected)


# ---------------------------------------------------------------------------
# The synthetic backend
# ---------------------------------------------------------------------------

def test_the_synthetic_detector_is_deterministic():
    """Noise comes from a hash of (object, frame), not a global RNG, so a
    failing test reproduces exactly."""
    a, b = SyntheticDetector(), SyntheticDetector()
    a.look_from(3.0, 3.0, 0.0)
    b.look_from(3.0, 3.0, 0.0)
    assert [(d.label, round(d.x1, 6)) for d in a.detect()] == \
           [(d.label, round(d.x1, 6)) for d in b.detect()]


def test_objects_behind_the_robot_are_not_detected():
    detector = SyntheticDetector(scene=[SceneObject("chair", 0.0, 3.0)])
    detector.look_from(3.0, 3.0, 0.0)      # facing +x, object is at -x
    assert detector.detect() == []


def test_objects_beyond_the_range_limit_are_not_detected():
    detector = SyntheticDetector(scene=[SceneObject("chair", 30.0, 3.0)],
                                 max_range_m=6.0)
    detector.look_from(3.0, 3.0, 0.0)
    assert detector.detect() == []


def test_confidence_falls_with_range():
    near = SyntheticDetector(scene=[SceneObject("chair", 5.0, 3.0)],
                             box_noise_px=0.0)
    far = SyntheticDetector(scene=[SceneObject("chair", 8.0, 3.0)],
                            box_noise_px=0.0)
    near.look_from(3.0, 3.0, 0.0)
    far.look_from(3.0, 3.0, 0.0)
    n, f = near.detect(), far.detect()
    if n and f:
        assert n[0].confidence > f[0].confidence


def test_grounding_a_frame_produces_map_coordinates():
    detector = SyntheticDetector(scene=[SceneObject("chair", 5.0, 3.0)],
                                 box_noise_px=0.0)
    detector.look_from(3.0, 3.0, 0.0)
    observations = observations_from(detector, detector.detect())
    assert observations
    assert observations[0].x == pytest.approx(5.0, abs=0.6)
    assert observations[0].y == pytest.approx(3.0, abs=0.6)


def test_the_detector_is_deterministic_across_PROCESSES():
    """The test the within-process version could not have caught.

    The first implementation used the builtin `hash()`. Python randomises
    string hashing per process unless PYTHONHASHSEED is set, so every RUN
    produced different noise while two detectors inside ONE process still
    agreed perfectly. The accuracy assertion failed roughly one run in three
    and passed on a rerun - the classic flaky test, and the same mistake the
    RRT planner in `robot_core` already carries a seed to prevent.

    This spawns two interpreters with deliberately different hash seeds and
    requires byte-identical output.
    """
    import os
    import subprocess
    import sys

    script = (
        "from robot_perception.backends.synthetic import SyntheticDetector;"
        "d=SyntheticDetector();d.look_from(3.0,3.0,0.0);"
        "print([(x.label, round(x.x1,9)) for x in d.detect()])"
    )
    outputs = []
    for seed in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        result = subprocess.run([sys.executable, "-c", script],
                                capture_output=True, text=True, env=env)
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout.strip())

    assert outputs[0] and outputs[0] == outputs[1], (
        "the synthetic detector's noise depends on PYTHONHASHSEED; use a "
        "stable hash (hashlib) rather than the builtin hash()"
    )
