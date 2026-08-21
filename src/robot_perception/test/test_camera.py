import math

import pytest

from robot_perception.camera import (
    MIN_DOWNWARD,
    CameraExtrinsics,
    CameraIntrinsics,
    base_to_map,
    detection_to_ground,
    ground_error_m,
    horizon_row,
    optical_to_body,
    project_to_ground,
)

#: Matches robot.gazebo.xacro. test_bringup.py asserts the URDF still says this.
WIDTH, HEIGHT, HFOV = 640, 480, 1.089


@pytest.fixture
def intrinsics():
    return CameraIntrinsics.from_fov(WIDTH, HEIGHT, HFOV)


@pytest.fixture
def extrinsics():
    return CameraExtrinsics()


# -- intrinsics -------------------------------------------------------------

def test_focal_length_matches_the_field_of_view(intrinsics):
    assert intrinsics.fx == pytest.approx((WIDTH / 2) / math.tan(HFOV / 2))
    assert intrinsics.fx == pytest.approx(529.3, abs=1.0)


def test_square_pixels(intrinsics):
    """fy == fx keeps the vertical FOV consistent with the aspect ratio instead
    of letting the two drift apart."""
    assert intrinsics.fy == intrinsics.fx


def test_the_principal_point_is_the_image_centre(intrinsics):
    assert (intrinsics.cx, intrinsics.cy) == (320.0, 240.0)


def test_horizontal_fov_round_trips(intrinsics):
    assert intrinsics.horizontal_fov == pytest.approx(HFOV)


def test_vertical_fov_follows_the_aspect_ratio(intrinsics):
    assert intrinsics.vertical_fov < intrinsics.horizontal_fov
    assert intrinsics.vertical_fov == pytest.approx(
        2 * math.atan((HEIGHT / 2) / intrinsics.fx)
    )


@pytest.mark.parametrize("width,height,fov", [
    (0, 480, 1.0), (640, 0, 1.0), (640, 480, 0.0), (640, 480, math.pi),
])
def test_impossible_cameras_are_rejected(width, height, fov):
    with pytest.raises(ValueError):
        CameraIntrinsics.from_fov(width, height, fov)


def test_a_fisheye_is_rejected_rather_than_silently_wrong():
    """Beyond 180 degrees a pinhole model is not merely inaccurate, it is
    undefined — tan(fov/2) changes sign."""
    with pytest.raises(ValueError, match="fisheye"):
        CameraIntrinsics.from_fov(640, 480, math.pi + 0.1)


def test_the_centre_pixel_looks_straight_down_the_optical_axis(intrinsics):
    ray = intrinsics.ray_in_optical(intrinsics.cx, intrinsics.cy)
    assert ray == pytest.approx((0.0, 0.0, 1.0))


def test_rays_are_unit_length(intrinsics):
    """An unnormalised ray still hits the ground plane in the right place, but
    every distance derived from it is scaled by a position-dependent factor —
    which looks like a distance-dependent calibration error."""
    for u, v in [(0, 0), (639, 479), (320, 240), (100, 400)]:
        x, y, z = intrinsics.ray_in_optical(u, v)
        assert math.sqrt(x * x + y * y + z * z) == pytest.approx(1.0)


def test_a_pixel_right_of_centre_gives_a_ray_to_the_right(intrinsics):
    assert intrinsics.ray_in_optical(intrinsics.cx + 100, intrinsics.cy)[0] > 0


def test_a_pixel_below_centre_gives_a_ray_pointing_down(intrinsics):
    # +y is DOWN in the optical convention.
    assert intrinsics.ray_in_optical(intrinsics.cx, intrinsics.cy + 100)[1] > 0


def test_contains(intrinsics):
    assert intrinsics.contains(0, 0)
    assert intrinsics.contains(639, 479)
    assert not intrinsics.contains(640, 240)
    assert not intrinsics.contains(-1, 240)


# -- REP-103 axis swap ------------------------------------------------------

def test_optical_forward_becomes_body_forward():
    """The swap the URDF encodes as rpy="-pi/2 0 -pi/2". Getting it wrong
    rotates every detection 90 degrees on the floor — plausible enough to be
    chased as a localisation bug."""
    assert optical_to_body((0, 0, 1)) == pytest.approx((1, 0, 0))


def test_optical_right_becomes_body_right():
    # Body +y is LEFT, so right is -y.
    assert optical_to_body((1, 0, 0)) == pytest.approx((0, -1, 0))


def test_optical_down_becomes_body_down():
    assert optical_to_body((0, 1, 0)) == pytest.approx((0, 0, -1))


def test_the_axis_swap_preserves_length():
    ray = (0.3, -0.5, 0.81)
    before = math.sqrt(sum(c * c for c in ray))
    after = math.sqrt(sum(c * c for c in optical_to_body(ray)))
    assert after == pytest.approx(before)


# -- ground projection ------------------------------------------------------

def test_a_ray_at_45_degrees_hits_at_the_camera_height():
    hit = project_to_ground((1.0, 0.0, -1.0), height_m=0.145)
    assert hit is not None
    assert hit.x == pytest.approx(0.145)
    assert hit.y == pytest.approx(0.0)


def test_a_shallower_ray_hits_further_away():
    near = project_to_ground((1.0, 0.0, -1.0), 0.145)
    far = project_to_ground((1.0, 0.0, -0.2), 0.145)
    assert far.x > near.x


def test_a_ray_at_the_horizon_has_no_intersection():
    """The case that must not return a number. Naive arithmetic doesn't fail
    here — it silently returns an enormous coordinate, and the landmark lands
    hundreds of metres away."""
    assert project_to_ground((1.0, 0.0, 0.0), 0.145) is None


def test_a_ray_above_the_horizon_has_no_intersection():
    assert project_to_ground((1.0, 0.0, 0.5), 0.145) is None


def test_a_ray_marginally_below_the_horizon_is_refused():
    assert project_to_ground((1.0, 0.0, -MIN_DOWNWARD / 2), 0.145) is None


def test_a_ray_pointing_backwards_is_refused():
    assert project_to_ground((-1.0, 0.0, -1.0), 0.145) is None


def test_range_is_the_planar_distance():
    hit = project_to_ground((1.0, 1.0, -1.0), 0.145)
    assert hit.range_m == pytest.approx(math.hypot(hit.x, hit.y))


# -- the ground-plane assumption's cost -------------------------------------

def test_an_object_on_the_floor_has_no_error():
    assert ground_error_m(range_m=2.0, height_m=0.145, object_height_m=0.0) == 0.0


def test_a_marking_5cm_up_projects_half_a_metre_too_far_at_one_metre():
    """The number that justifies multi-frame confirmation. At this camera height
    a point only 5 cm off the floor lands 1.5x too far away — a consistent bias,
    which is worse than noise because averaging does not remove it."""
    error = ground_error_m(range_m=1.0, height_m=0.145, object_height_m=0.05)
    assert error == pytest.approx(0.526, abs=0.01)


def test_the_error_grows_with_range():
    near = ground_error_m(1.0, 0.145, 0.05)
    far = ground_error_m(4.0, 0.145, 0.05)
    assert far == pytest.approx(4 * near)


def test_an_object_at_camera_height_never_reaches_the_floor():
    assert math.isinf(ground_error_m(2.0, 0.145, 0.145))
    assert math.isinf(ground_error_m(2.0, 0.145, 1.5))


# -- pixel to base frame ----------------------------------------------------

def test_a_pixel_below_centre_lands_in_front_of_the_robot(intrinsics, extrinsics):
    hit = detection_to_ground(intrinsics, extrinsics, 320, 400)
    assert hit is not None
    # 0.4796 m from the lens, plus the 0.18 m camera offset along the base x axis.
    assert hit.x == pytest.approx(0.6596, abs=0.005)
    assert hit.y == pytest.approx(0.0, abs=1e-9)


def test_the_geometry_agrees_with_trigonometry(intrinsics, extrinsics):
    """Independent derivation: the depression angle straight from the pixel
    offset, then distance = height / tan(angle)."""
    v = 400
    depression = math.atan((v - intrinsics.cy) / intrinsics.fy)
    expected = extrinsics.z / math.tan(depression) + extrinsics.x

    hit = detection_to_ground(intrinsics, extrinsics, 320, v)
    assert hit.x == pytest.approx(expected, abs=1e-6)


def test_a_pixel_left_of_centre_lands_to_the_left(intrinsics, extrinsics):
    hit = detection_to_ground(intrinsics, extrinsics, 200, 400)
    assert hit.y > 0        # +y is left


def test_a_pixel_at_or_above_the_horizon_returns_nothing(intrinsics, extrinsics):
    assert detection_to_ground(intrinsics, extrinsics, 320, 240) is None
    assert detection_to_ground(intrinsics, extrinsics, 320, 100) is None


def test_a_pixel_outside_the_image_returns_nothing(intrinsics, extrinsics):
    assert detection_to_ground(intrinsics, extrinsics, 700, 400) is None
    assert detection_to_ground(intrinsics, extrinsics, -5, 400) is None


def test_lower_pixels_are_always_nearer(intrinsics, extrinsics):
    """Monotonicity. If this ever inverts, a sign is wrong somewhere in the
    chain and everything downstream is quietly mirrored."""
    ranges = [detection_to_ground(intrinsics, extrinsics, 320, v).range_m
              for v in range(260, 480, 20)]
    assert ranges == sorted(ranges, reverse=True)


# -- mounting angles --------------------------------------------------------

def test_pitching_the_camera_down_brings_the_horizon_into_view(intrinsics):
    level = CameraExtrinsics()
    # POSITIVE pitch is down in REP-103: a positive rotation about the
    # left-pointing y axis takes the forward vector downward. This is the
    # opposite of the aerospace convention, and it cost two failing tests.
    tilted = CameraExtrinsics(pitch=0.3)

    # With a level camera the image centre is exactly the horizon.
    assert detection_to_ground(intrinsics, level, 320, 240) is None
    # Tilted down, the same pixel now looks at the floor.
    assert detection_to_ground(intrinsics, tilted, 320, 240) is not None


def test_pitching_down_moves_the_horizon_up_the_image(intrinsics):
    level = horizon_row(intrinsics, CameraExtrinsics())
    tilted = horizon_row(intrinsics, CameraExtrinsics(pitch=0.3))
    assert tilted < level


def test_a_level_cameras_horizon_is_the_principal_row(intrinsics):
    assert horizon_row(intrinsics, CameraExtrinsics()) == intrinsics.cy


def test_pitching_up_pushes_the_horizon_below_the_image(intrinsics):
    """A camera tilted up far enough sees no floor at all, and every detection
    in the frame must be refused rather than projected to a huge coordinate."""
    up = CameraExtrinsics(pitch=-0.5)
    assert horizon_row(intrinsics, up) > intrinsics.height
    assert detection_to_ground(intrinsics, up, 320, 479) is None


def test_yaw_rotates_the_hit_about_the_robot(intrinsics):
    straight = detection_to_ground(intrinsics, CameraExtrinsics(), 320, 400)
    yawed = detection_to_ground(intrinsics, CameraExtrinsics(yaw=0.4), 320, 400)
    assert yawed.y > straight.y
    assert yawed.range_m == pytest.approx(straight.range_m, rel=0.05)


# -- base frame to map frame ------------------------------------------------

def test_no_rotation_is_a_translation():
    assert base_to_map((1.0, 2.0), 10.0, 20.0, 0.0) == pytest.approx((11.0, 22.0))


def test_a_quarter_turn_maps_forward_to_left():
    x, y = base_to_map((1.0, 0.0), 0.0, 0.0, math.pi / 2)
    assert (x, y) == pytest.approx((0.0, 1.0), abs=1e-9)


def test_rotation_preserves_distance_from_the_robot():
    for yaw in [0.0, 0.7, math.pi, -2.1]:
        x, y = base_to_map((1.5, -0.5), 3.0, 4.0, yaw)
        assert math.hypot(x - 3.0, y - 4.0) == pytest.approx(math.hypot(1.5, -0.5))


def test_the_full_chain_places_a_detection_ahead_of_a_rotated_robot(intrinsics,
                                                                    extrinsics):
    """End to end: a pixel low in the frame, on a robot facing north at (3, 3),
    must land north of the robot."""
    hit = detection_to_ground(intrinsics, extrinsics, 320, 420)
    x, y = base_to_map(hit.as_tuple(), 3.0, 3.0, math.pi / 2)
    assert y > 3.0
    assert x == pytest.approx(3.0, abs=1e-9)


# -- the inverse, and the round trip ----------------------------------------

def test_body_to_optical_inverts_optical_to_body():
    from robot_perception.camera import body_to_optical
    for ray in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0.3, -0.5, 0.81)]:
        assert body_to_optical(optical_to_body(ray)) == pytest.approx(ray)


def test_map_to_base_inverts_base_to_map():
    from robot_perception.camera import map_to_base
    for yaw in [0.0, 0.9, -2.2, math.pi]:
        point = (1.7, -0.4)
        there = base_to_map(point, 3.0, 4.0, yaw)
        back = map_to_base(there, 3.0, 4.0, yaw)
        assert back == pytest.approx(point)


def test_pixel_to_ground_and_back_round_trips(intrinsics, extrinsics):
    """The strongest single check on the whole geometry chain. If any sign,
    axis or order is wrong in either direction, this fails."""
    from robot_perception.camera import world_to_pixel
    for u, v in [(320, 400), (100, 350), (540, 470), (200, 300)]:
        hit = detection_to_ground(intrinsics, extrinsics, u, v)
        assert hit is not None
        back = world_to_pixel(intrinsics, extrinsics, hit.as_tuple(), 0.0)
        assert back == pytest.approx((u, v), abs=1e-6)


def test_the_round_trip_survives_a_mounting_angle(intrinsics):
    from robot_perception.camera import world_to_pixel
    tilted = CameraExtrinsics(pitch=0.25, yaw=0.15)
    for u, v in [(320, 300), (150, 420), (500, 260)]:
        hit = detection_to_ground(intrinsics, tilted, u, v)
        if hit is None:
            continue
        assert world_to_pixel(intrinsics, tilted, hit.as_tuple(), 0.0) == \
            pytest.approx((u, v), abs=1e-6)


def test_a_point_behind_the_camera_has_no_pixel(intrinsics, extrinsics):
    from robot_perception.camera import world_to_pixel
    assert world_to_pixel(intrinsics, extrinsics, (-2.0, 0.0), 0.0) is None


def test_a_point_outside_the_field_of_view_has_no_pixel(intrinsics, extrinsics):
    from robot_perception.camera import world_to_pixel
    assert world_to_pixel(intrinsics, extrinsics, (1.0, 9.0), 0.0) is None


def test_a_taller_point_projects_higher_in_the_image(intrinsics, extrinsics):
    from robot_perception.camera import world_to_pixel
    floor = world_to_pixel(intrinsics, extrinsics, (1.5, 0.0), 0.0)
    up = world_to_pixel(intrinsics, extrinsics, (1.5, 0.0), 0.10)
    assert up[1] < floor[1]      # v increases downward
