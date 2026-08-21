import math

import pytest

from robot_perception.tracking import (
    ConfirmationPolicy,
    Observation,
    Track,
    Tracker,
    evaluate,
)


def observe(x, y, from_x, from_y, label="chair", confidence=0.8, stamp=0.0):
    return Observation(label=label, confidence=confidence, x=x, y=y,
                       from_x=from_x, from_y=from_y, stamp=stamp)


@pytest.fixture
def policy():
    return ConfirmationPolicy()


# -- position estimation ----------------------------------------------------

def test_a_single_observation_is_its_own_position():
    track = Track("chair", [observe(2.0, 3.0, 0, 0)])
    assert track.position == (2.0, 3.0)


def test_an_empty_track_has_no_position():
    with pytest.raises(ValueError):
        Track("chair").position


def test_the_position_is_a_median_not_a_mean():
    """One badly mis-projected frame — a truncated box, a reflection — moves a
    mean by metres and a median not at all."""
    track = Track("chair", [
        observe(2.0, 3.0, 0, 0),
        observe(2.1, 3.1, 1, 0),
        observe(1.9, 2.9, 0, 1),
        observe(40.0, 90.0, 1, 1),      # one wild projection
    ])
    x, y = track.position
    assert x == pytest.approx(2.05, abs=0.01)
    assert y == pytest.approx(3.05, abs=0.01)


def test_confidence_is_the_best_look_not_the_average():
    """A detector 0.9 sure once and 0.4 sure four times has seen the object
    clearly once. The poor frames are poor because of distance or blur, not
    because the object is absent."""
    track = Track("chair", [
        observe(1, 1, 0, 0, confidence=0.9),
        *[observe(1, 1, 0, 0, confidence=0.4) for _ in range(4)],
    ])
    assert track.confidence == 0.9


def test_spread_is_zero_for_one_observation():
    assert Track("chair", [observe(1, 1, 0, 0)]).spread_m == 0.0


def test_spread_measures_disagreement():
    tight = Track("chair", [observe(2.0, 3.0, 0, 0), observe(2.05, 3.0, 1, 0)])
    loose = Track("chair", [observe(2.0, 3.0, 0, 0), observe(5.0, 3.0, 1, 0)])
    assert tight.spread_m < 0.1
    assert loose.spread_m > 1.0


# -- THE headline property --------------------------------------------------

def test_many_sightings_from_one_spot_do_not_confirm(policy):
    """The whole reason this module exists.

    A robot that stops and stares gets a hundred consistent measurements of the
    *same wrong position*, because the ground-plane projection error is identical
    from an identical viewpoint. Averaging makes it more confident about being
    wrong. Confirmation must require the robot to have MOVED.
    """
    track = Track("chair", [observe(2.0, 3.0, 0.0, 0.0, stamp=i)
                            for i in range(50)])

    result = evaluate(track, policy)
    assert not result.confirmed
    assert any("distinct viewpoint" in r for r in result.reasons)
    assert any("the robot has to move" in r for r in result.reasons)


def test_the_same_object_from_two_viewpoints_does_confirm(policy):
    track = Track("chair", [
        observe(2.00, 3.00, 0.0, 0.0, stamp=1),
        observe(2.05, 3.02, 0.1, 0.0, stamp=2),
        observe(1.98, 2.97, 1.4, 0.3, stamp=3),
        observe(2.02, 3.01, 1.5, 0.4, stamp=4),
    ])
    result = evaluate(track, policy)
    assert result.confirmed, result.reasons


def test_viewpoints_closer_than_the_separation_do_not_count(policy):
    """Drifting 5 cm is not a new viewpoint. Counting it would let a
    stationary-but-jittery robot confirm anything."""
    track = Track("chair", [observe(2.0, 3.0, 0.01 * i, 0.0, stamp=i)
                            for i in range(20)])
    assert track.distinct_viewpoints(policy.viewpoint_separation_m) == 1
    assert not evaluate(track, policy).confirmed


def test_disagreeing_views_are_refused_not_averaged(policy):
    """A large spread is direct evidence the ground-plane assumption is failing
    for this object — it is probably not standing on the floor. Averaging the
    disagreement away would put a confident landmark in the wrong place."""
    track = Track("tv", [
        observe(2.0, 3.0, 0.0, 0.0, label="tv", stamp=1),
        observe(2.1, 3.0, 0.2, 0.0, label="tv", stamp=2),
        observe(6.0, 3.0, 2.0, 0.0, label="tv", stamp=3),
        observe(6.2, 3.1, 2.2, 0.0, label="tv", stamp=4),
    ])
    result = evaluate(track, policy)
    assert not result.confirmed
    assert any("do not triangulate" in r for r in result.reasons)


def test_low_confidence_is_refused(policy):
    track = Track("chair", [
        observe(2.0, 3.0, 0.0, 0.0, confidence=0.3, stamp=1),
        observe(2.0, 3.0, 1.0, 0.0, confidence=0.3, stamp=2),
        observe(2.0, 3.0, 2.0, 0.0, confidence=0.3, stamp=3),
        observe(2.0, 3.0, 3.0, 0.0, confidence=0.3, stamp=4),
    ])
    assert not evaluate(track, policy).confirmed


def test_a_refusal_always_explains_itself(policy):
    """"Why is the chair not on the map" deserves an answer, and the answer is
    also the fastest way to debug a patrol route that never triangulates."""
    result = evaluate(Track("chair", [observe(1, 1, 0, 0)]), policy)
    assert not result.confirmed
    assert result.reasons
    assert all(isinstance(r, str) and r for r in result.reasons)


def test_a_confirmation_also_explains_itself(policy):
    track = Track("chair", [
        observe(2.0, 3.0, 0.0, 0.0, stamp=1),
        observe(2.0, 3.0, 1.0, 0.0, stamp=2),
        observe(2.0, 3.0, 2.0, 0.0, stamp=3),
        observe(2.0, 3.0, 3.0, 0.0, stamp=4),
    ])
    result = evaluate(track, policy)
    assert result.confirmed
    assert "sightings from" in result.reasons[0]


# -- viewpoint geometry -----------------------------------------------------

def test_viewpoints_in_a_line_cover_a_small_arc():
    """Two viewpoints far apart but both in front of the object give almost no
    new depth information. Distance alone is the wrong measure."""
    track = Track("chair", [
        observe(0.0, 0.0, 5.0, 0.0),
        observe(0.0, 0.0, 9.0, 0.0),
    ])
    assert track.viewpoint_arc() == pytest.approx(0.0, abs=1e-9)


def test_viewpoints_at_right_angles_cover_a_quarter_turn():
    track = Track("chair", [
        observe(0.0, 0.0, 3.0, 0.0),
        observe(0.0, 0.0, 0.0, 3.0),
    ])
    assert track.viewpoint_arc() == pytest.approx(math.pi / 2, abs=1e-9)


def test_a_single_viewpoint_covers_no_arc():
    assert Track("chair", [observe(0, 0, 1, 1)]).viewpoint_arc() == 0.0


# -- association ------------------------------------------------------------

def test_nearby_observations_join_one_track():
    tracker = Tracker(gate_m=1.0)
    tracker.update([observe(2.0, 3.0, 0, 0)])
    tracker.update([observe(2.2, 3.1, 1, 0)])
    assert len(tracker.tracks) == 1
    assert len(tracker.tracks[0]) == 2


def test_distant_observations_start_a_new_track():
    tracker = Tracker(gate_m=1.0)
    tracker.update([observe(2.0, 3.0, 0, 0)])
    tracker.update([observe(8.0, 9.0, 1, 0)])
    assert len(tracker.tracks) == 2


def test_different_labels_never_merge():
    """Associating a detected chair with a tracked table because they are close
    produces a track whose label is whichever the detector said first."""
    tracker = Tracker(gate_m=5.0)
    tracker.update([observe(2.0, 3.0, 0, 0, label="chair")])
    tracker.update([observe(2.1, 3.0, 1, 0, label="dining table")])
    assert len(tracker.tracks) == 2
    assert {t.label for t in tracker.tracks} == {"chair", "dining table"}


def test_an_observation_joins_the_nearest_candidate():
    tracker = Tracker(gate_m=3.0)
    tracker.update([observe(0.0, 0.0, 0, 0), observe(2.5, 0.0, 0, 0)])
    tracker.update([observe(2.4, 0.0, 1, 0)])

    near = [t for t in tracker.tracks if abs(t.position[0] - 2.45) < 0.2]
    assert len(near) == 1
    assert len(near[0]) == 2


def test_track_ids_are_unique_and_stable():
    tracker = Tracker()
    tracker.update([observe(0, 0, 0, 0), observe(9, 9, 0, 0)])
    ids = [t.track_id for t in tracker.tracks]
    assert len(set(ids)) == 2
    tracker.update([observe(0.1, 0.1, 1, 0)])
    assert [t.track_id for t in tracker.tracks] == ids


# -- ageing -----------------------------------------------------------------

def test_stale_tracks_are_pruned():
    tracker = Tracker(policy=ConfirmationPolicy(stale_after_s=10.0))
    tracker.update([observe(0, 0, 0, 0, stamp=0.0)])
    dropped = tracker.prune(now=100.0)
    assert len(dropped) == 1
    assert tracker.tracks == []


def test_recent_tracks_survive_pruning():
    """Long enough to survive an occlusion while the robot drives past a
    pillar."""
    tracker = Tracker(policy=ConfirmationPolicy(stale_after_s=30.0))
    tracker.update([observe(0, 0, 0, 0, stamp=100.0)])
    assert tracker.prune(now=110.0) == []
    assert len(tracker.tracks) == 1


# -- the tracker's verdicts -------------------------------------------------

def test_confirmed_and_pending_partition_the_tracks():
    tracker = Tracker()
    for i in range(4):
        tracker.update([observe(2.0, 3.0, float(i), 0.0, stamp=i)])
    tracker.update([observe(9.0, 9.0, 0.0, 0.0, stamp=5)])

    assert len(tracker.confirmed()) == 1
    assert len(tracker.pending()) == 1
    assert len(tracker.confirmed()) + len(tracker.pending()) == len(tracker.tracks)


def test_pending_carries_the_reason():
    tracker = Tracker()
    tracker.update([observe(9.0, 9.0, 0.0, 0.0)])
    _, result = tracker.pending()[0]
    assert result.reasons


def test_summary_counts_confirmed_by_label():
    tracker = Tracker()
    for i in range(4):
        tracker.update([
            observe(2.0, 3.0, float(i), 0.0, stamp=i),
            observe(7.0, 1.0, float(i), 0.0, stamp=i, label="potted plant"),
        ])
    assert tracker.summary() == {"chair": 1, "potted plant": 1}


def test_two_detections_in_one_frame_never_join_one_track():
    """Found by a test, and it is the kind of bug that produces a confident
    landmark in empty floor.

    Two chairs 2 m apart both fall inside a 3 m gate. Matching them one at a
    time merged them into a single track whose median position sat in the gap
    between them — a chair on the map where there is carpet.
    """
    tracker = Tracker(gate_m=3.0)
    tracker.update([observe(0.0, 0.0, 0, 0), observe(2.5, 0.0, 0, 0)])
    assert len(tracker.tracks) == 2


def test_assignment_does_not_depend_on_detector_output_order():
    """Greedy global assignment, not arrival order. Matching in the order the
    detector happened to emit boxes makes the result flap between runs."""
    forward = Tracker(gate_m=3.0)
    forward.update([observe(0.0, 0.0, 0, 0), observe(2.5, 0.0, 0, 0)])
    forward.update([observe(0.1, 0.0, 1, 0), observe(2.4, 0.0, 1, 0)])

    reverse = Tracker(gate_m=3.0)
    reverse.update([observe(2.5, 0.0, 0, 0), observe(0.0, 0.0, 0, 0)])
    reverse.update([observe(2.4, 0.0, 1, 0), observe(0.1, 0.0, 1, 0)])

    assert sorted(round(t.position[0], 3) for t in forward.tracks) == \
           sorted(round(t.position[0], 3) for t in reverse.tracks)
    assert [len(t) for t in forward.tracks] == [2, 2]


def test_a_track_updated_mid_frame_does_not_move_its_own_targets():
    """Positions are snapshotted before assignment. Updating a track while its
    siblings are still being matched moves the target they are measured
    against, and the effect depends on iteration order."""
    tracker = Tracker(gate_m=1.5)
    tracker.update([observe(0.0, 0.0, 0, 0)])
    tracker.update([observe(0.2, 0.0, 1, 0), observe(1.4, 0.0, 1, 0)])
    assert len(tracker.tracks) == 2
