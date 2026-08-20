import math

import pytest

from robot_core.safety import STOP, SafetyGovernor, SafetyLimits, Twist


@pytest.fixture
def governor():
    return SafetyGovernor(SafetyLimits(
        max_linear=0.5, max_angular=1.0,
        max_linear_accel=100.0, max_angular_accel=100.0,   # effectively off
        stop_distance=0.4, slow_distance=1.0,
        scan_timeout_s=0.5,
    ))


def clear(governor, desired, now=1.0, distance=5.0):
    """Helper: a fresh, close-enough scan with nothing nearby."""
    return governor.limit(desired, now, nearest_obstacle_m=distance, scan_age_s=0.05)


# -- fail closed ------------------------------------------------------------

def test_stops_when_no_scan_has_arrived(governor):
    decision = governor.limit(Twist(0.4, 0.0), now=1.0, scan_age_s=None)
    assert decision.command == STOP
    assert "no scan" in decision.reasons[0]


def test_stops_on_a_stale_scan(governor):
    """A scan older than the timeout is not evidence about the present. This is
    the check that catches a crashed or wedged lidar driver — the robot keeps
    receiving velocity commands and would otherwise keep driving blind."""
    decision = governor.limit(Twist(0.4, 0.0), now=1.0,
                              nearest_obstacle_m=5.0, scan_age_s=2.0)
    assert decision.command == STOP
    assert "old" in decision.reasons[0]


def test_a_fresh_scan_is_allowed_through(governor):
    assert clear(governor, Twist(0.3, 0.0)).command.linear_x > 0


# -- e-stop -----------------------------------------------------------------

def test_estop_overrides_everything(governor):
    governor.engage_estop()
    decision = governor.limit(Twist(0.5, 0.5), now=1.0,
                              nearest_obstacle_m=99.0, scan_age_s=0.0)
    assert decision.command == STOP
    assert decision.reasons == ["e-stop engaged"]


def test_estop_does_not_clear_itself(governor):
    governor.engage_estop()
    for t in range(1, 20):
        assert governor.limit(Twist(0.5, 0.0), now=float(t),
                              nearest_obstacle_m=99.0, scan_age_s=0.0).stopped
    assert governor.estopped


def test_estop_requires_an_explicit_release(governor):
    governor.engage_estop()
    governor.release_estop()
    assert not governor.estopped
    assert clear(governor, Twist(0.3, 0.0)).command.linear_x > 0


# -- obstacle response ------------------------------------------------------

def test_full_speed_with_nothing_in_range(governor):
    decision = clear(governor, Twist(0.5, 0.0), distance=8.0)
    assert math.isclose(decision.command.linear_x, 0.5)
    assert decision.scale == 1.0


def test_hard_stop_inside_the_stop_distance(governor):
    decision = clear(governor, Twist(0.5, 0.0), distance=0.3)
    assert decision.command.linear_x == 0.0
    assert decision.scale == 0.0


def test_stop_distance_boundary_is_inclusive(governor):
    assert clear(governor, Twist(0.5, 0.0), distance=0.4).command.linear_x == 0.0


def test_scales_linearly_across_the_slow_zone(governor):
    midpoint = clear(governor, Twist(0.5, 0.0), distance=0.7)
    assert math.isclose(midpoint.scale, 0.5, abs_tol=1e-9)
    assert math.isclose(midpoint.command.linear_x, 0.25, abs_tol=1e-9)


def test_speed_is_monotonic_in_distance(governor):
    previous = -1.0
    for distance in [0.4, 0.5, 0.6, 0.8, 1.0, 2.0]:
        speed = SafetyGovernor(governor.limits).limit(
            Twist(0.5, 0.0), 1.0, nearest_obstacle_m=distance, scan_age_s=0.0
        ).command.linear_x
        assert speed >= previous
        previous = speed


def test_reversing_away_from_an_obstacle_is_not_blocked(governor):
    """Backing off is the correct recovery when something is in front, so the
    obstacle must not veto the manoeuvre that escapes it."""
    decision = clear(governor, Twist(-0.3, 0.0), distance=0.1)
    assert decision.command.linear_x < 0


def test_turning_in_place_is_allowed_near_an_obstacle(governor):
    decision = clear(governor, Twist(0.0, 0.8), distance=0.2)
    assert math.isclose(decision.command.angular_z, 0.8)


def test_missing_obstacle_distance_does_not_scale(governor):
    # Distinct from a missing *scan*: a fresh scan with no return inside the
    # forward arc means "clear", not "unknown".
    decision = governor.limit(Twist(0.4, 0.0), now=1.0,
                              nearest_obstacle_m=None, scan_age_s=0.05)
    assert math.isclose(decision.command.linear_x, 0.4)


# -- absolute limits --------------------------------------------------------

def test_clamps_excessive_linear_velocity(governor):
    decision = clear(governor, Twist(9.0, 0.0))
    assert decision.command.linear_x == 0.5
    assert decision.limited


def test_clamps_excessive_angular_velocity(governor):
    decision = clear(governor, Twist(0.0, -9.0))
    assert decision.command.angular_z == -1.0


def test_a_command_within_limits_is_untouched(governor):
    decision = clear(governor, Twist(0.2, 0.3))
    assert decision.command == Twist(0.2, 0.3)
    assert not decision.limited


# -- acceleration -----------------------------------------------------------

def test_acceleration_is_ramped_not_stepped():
    """A step change slips the wheels, which corrupts odometry, which corrupts
    localisation — a control problem that presents as a mapping bug."""
    governor = SafetyGovernor(SafetyLimits(max_linear_accel=0.5))
    governor.limit(Twist(0.0, 0.0), now=0.0, nearest_obstacle_m=9.0, scan_age_s=0.0)
    decision = governor.limit(Twist(0.45, 0.0), now=0.1,
                              nearest_obstacle_m=9.0, scan_age_s=0.0)
    assert math.isclose(decision.command.linear_x, 0.05, abs_tol=1e-9)
    assert "acceleration limited" in " ".join(decision.reasons)


def test_ramps_up_over_several_cycles():
    governor = SafetyGovernor(SafetyLimits(max_linear=0.4, max_linear_accel=0.5))
    speeds = []
    for i in range(1, 15):
        speeds.append(governor.limit(
            Twist(0.4, 0.0), now=i * 0.1, nearest_obstacle_m=9.0, scan_age_s=0.0
        ).command.linear_x)
    assert speeds == sorted(speeds)
    assert math.isclose(speeds[-1], 0.4, abs_tol=1e-9)


def test_deceleration_is_also_ramped():
    governor = SafetyGovernor(SafetyLimits(max_linear=0.4, max_linear_accel=0.5))
    for i in range(1, 20):
        governor.limit(Twist(0.4, 0.0), now=i * 0.1,
                       nearest_obstacle_m=9.0, scan_age_s=0.0)
    decision = governor.limit(Twist(0.0, 0.0), now=2.0,
                              nearest_obstacle_m=9.0, scan_age_s=0.0)
    assert decision.command.linear_x > 0.0


def test_estop_bypasses_the_acceleration_ramp():
    """The ramp is a comfort and traction feature. An emergency stop is not
    comfortable and must not be smoothed."""
    governor = SafetyGovernor(SafetyLimits(max_linear=0.4, max_linear_accel=0.1))
    for i in range(1, 20):
        governor.limit(Twist(0.4, 0.0), now=i * 0.1,
                       nearest_obstacle_m=9.0, scan_age_s=0.0)
    governor.engage_estop()
    assert governor.limit(Twist(0.4, 0.0), now=3.0,
                          nearest_obstacle_m=9.0, scan_age_s=0.0).command == STOP


# -- scan parsing -----------------------------------------------------------

def test_nearest_in_arc_ignores_returns_behind_the_robot(governor):
    """8 beams over a full circle starting at -pi, so index 0 points backwards
    and index 4 points straight ahead. A wall 20 cm behind a robot driving
    forwards is not a reason to stop."""
    ranges = [0.2, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    nearest = governor.nearest_in_arc(ranges, angle_min=-math.pi,
                                      angle_increment=2 * math.pi / 8)
    assert nearest == 5.0


def test_nearest_in_arc_does_not_ignore_a_return_straight_ahead(governor):
    ranges = [5.0, 5.0, 5.0, 5.0, 0.2, 5.0, 5.0, 5.0]
    nearest = governor.nearest_in_arc(ranges, angle_min=-math.pi,
                                      angle_increment=2 * math.pi / 8)
    assert nearest == 0.2


def test_nearest_in_arc_finds_the_closest_forward_return(governor):
    ranges = [9.0] * 9
    ranges[4] = 0.6                      # straight ahead
    nearest = governor.nearest_in_arc(ranges, angle_min=-math.pi / 2,
                                      angle_increment=math.pi / 8)
    assert nearest == 0.6


def test_nan_and_inf_returns_are_dropped(governor):
    """A lidar reports inf for no-return, which often means glass or a matte
    black surface absorbed the beam — the two obstacles most likely to be hit.
    Treating inf as 'far' is exactly backwards."""
    ranges = [float("inf"), float("nan"), -1.0, 0.0, 2.5]
    nearest = governor.nearest_in_arc(ranges, angle_min=0.0, angle_increment=0.01)
    assert nearest == 2.5


def test_an_all_invalid_scan_reports_nothing_rather_than_a_false_clear(governor):
    ranges = [float("inf")] * 20
    assert governor.nearest_in_arc(ranges, 0.0, 0.01) is None


# -- ordering ---------------------------------------------------------------

def test_clamping_happens_before_obstacle_scaling(governor):
    """Order matters, and getting it backwards defeats the scaling entirely.

    Scale-then-clamp: 50 m/s scaled by 0.5 is 25 m/s, which still clamps to the
    0.5 m/s maximum — the robot drives at full speed at a wall half a metre
    away and the safety scaling had literally no effect.

    Clamp-then-scale: 50 clamps to 0.5, then scales to 0.25. Correct.
    """
    decision = clear(governor, Twist(50.0, 0.0), distance=0.7)
    assert math.isclose(decision.command.linear_x, 0.25, abs_tol=1e-9)
