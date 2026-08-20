import math

import pytest

from robot_core.geometry import (
    Pose2D,
    angle_difference,
    normalise_angle,
    quaternion_to_yaw,
    unicycle_control,
    yaw_to_quaternion,
)


# -- angle wrapping ---------------------------------------------------------

@pytest.mark.parametrize("angle,expected", [
    (0.0, 0.0),
    (math.pi / 2, math.pi / 2),
    (3 * math.pi, math.pi),
    (-3 * math.pi, math.pi),
    (2 * math.pi, 0.0),
    (7 * math.pi / 4, -math.pi / 4),
])
def test_normalise_angle(angle, expected):
    assert math.isclose(normalise_angle(angle), expected, abs_tol=1e-9)


def test_normalise_angle_always_lands_in_the_interval():
    for k in range(-20, 21):
        wrapped = normalise_angle(k * 1.1)
        assert -math.pi < wrapped <= math.pi + 1e-12


def test_the_wrap_around_bug():
    """179 degrees to -179 degrees is a 2 degree turn, not 358.

    Without wrapping the controller sees the large number and spins the long
    way round — the single most common bug in this file's subject area.
    """
    difference = angle_difference(math.radians(-179), math.radians(179))
    assert math.isclose(abs(difference), math.radians(2), abs_tol=1e-9)


def test_angle_difference_is_signed():
    assert angle_difference(math.radians(10), math.radians(0)) > 0
    assert angle_difference(math.radians(-10), math.radians(0)) < 0


# -- quaternions ------------------------------------------------------------

@pytest.mark.parametrize("yaw", [0.0, 0.5, -1.2, math.pi / 2, 3.0, -3.0])
def test_quaternion_round_trip(yaw):
    x, y, z, w = yaw_to_quaternion(yaw)
    assert math.isclose(quaternion_to_yaw(x, y, z, w), yaw, abs_tol=1e-9)


def test_quaternion_is_unit_length():
    x, y, z, w = yaw_to_quaternion(1.234)
    assert math.isclose(math.sqrt(x * x + y * y + z * z + w * w), 1.0)


def test_yaw_only_rotation_has_no_roll_or_pitch_components():
    x, y, _, _ = yaw_to_quaternion(0.9)
    assert x == 0.0 and y == 0.0


def test_negated_quaternion_is_the_same_rotation():
    """q and -q represent the identical rotation. Comparing quaternions
    componentwise reports a difference that does not physically exist."""
    x, y, z, w = yaw_to_quaternion(1.1)
    assert math.isclose(quaternion_to_yaw(-x, -y, -z, -w),
                        quaternion_to_yaw(x, y, z, w), abs_tol=1e-9)


def test_pose_from_ros_needs_no_geometry_msgs_import():
    class _Position:
        x, y, z = 1.5, -2.5, 0.0

    class _Orientation:
        x, y, z, w = yaw_to_quaternion(0.75)

    class _Pose:
        position = _Position()
        orientation = _Orientation()

    pose = Pose2D.from_ros(_Pose())
    assert (pose.x, pose.y) == (1.5, -2.5)
    assert math.isclose(pose.yaw, 0.75, abs_tol=1e-9)


# -- pose maths -------------------------------------------------------------

def test_distance():
    assert math.isclose(Pose2D(0, 0).distance_to(Pose2D(3, 4)), 5.0)


def test_heading_uses_atan2_so_the_quadrant_survives():
    """atan would fold the third quadrant onto the first — a robot facing
    backwards reported as facing forwards."""
    assert math.isclose(Pose2D(0, 0).heading_to(Pose2D(-1, -1)),
                        math.radians(-135), abs_tol=1e-9)
    assert math.isclose(Pose2D(0, 0).heading_to(Pose2D(1, 1)),
                        math.radians(45), abs_tol=1e-9)


def test_bearing_is_relative_to_current_heading():
    facing_north = Pose2D(0, 0, math.pi / 2)
    assert math.isclose(facing_north.bearing_to(Pose2D(0, 5)), 0.0, abs_tol=1e-9)
    assert math.isclose(facing_north.bearing_to(Pose2D(5, 0)),
                        -math.pi / 2, abs_tol=1e-9)


# -- controller -------------------------------------------------------------

def test_drives_forward_when_already_aligned():
    linear, angular = unicycle_control(Pose2D(0, 0, 0), Pose2D(2, 0, 0))
    assert linear > 0
    assert math.isclose(angular, 0.0, abs_tol=1e-9)


def test_turns_in_place_when_badly_misaligned():
    """Driving forward while 90 degrees off traces a long arc away from the
    goal; in a corridor that arc is a wall."""
    linear, angular = unicycle_control(Pose2D(0, 0, 0), Pose2D(0, 2, 0))
    assert linear == 0.0
    assert angular > 0


def test_turns_the_short_way_round():
    behind = unicycle_control(Pose2D(0, 0, math.radians(170)),
                              Pose2D(-1, -0.05, 0))
    assert abs(behind[1]) < 1.5 * math.pi


def test_stops_translating_at_the_goal_and_settles_heading():
    linear, angular = unicycle_control(Pose2D(0, 0, 0), Pose2D(0.05, 0, 1.0))
    assert linear == 0.0
    assert angular > 0


def test_slows_down_as_it_approaches():
    far = unicycle_control(Pose2D(0, 0, 0), Pose2D(5, 0, 0))[0]
    near = unicycle_control(Pose2D(0, 0, 0), Pose2D(1, 0, 0))[0]
    assert far > near > 0
