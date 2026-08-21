"""Pinhole camera model and ground-plane projection.

**The real problem this package solves.** A semantic map that is typed by hand
goes stale the moment somebody moves a desk. In a warehouse or an office that is
weekly. So the robot has to build the map from what it sees, and keep it
current — which means turning a bounding box in an image into a coordinate on
the floor, reliably enough to act on.

That conversion is this file. It is pure geometry, no ROS and no ML, so every
step of it is unit-testable.

**How a pixel becomes a floor coordinate**

    (u, v) in the image
        │  intrinsics: fx, fy, cx, cy      ─ derived from the camera's FOV
        ▼
    a ray in the OPTICAL frame (x right, y down, z forward)
        │  R_camera_from_optical           ─ the REP-103 axis swap
        ▼
    a ray in the camera BODY frame (x forward, y left, z up)
        │  camera extrinsics on the robot
        ▼
    a ray in the robot BASE frame
        │  the robot's pose in the map
        ▼
    a ray in the MAP frame
        │  intersect with the ground plane z = 0
        ▼
    (x, y) on the floor

**The assumption that does the work, and its cost.** Intersecting with a ground
plane is what lets a *single* camera produce a position at all — one image
cannot recover depth on its own. It is exact for anything standing on the floor
and wrong for anything that is not, and the error grows with height: a sign on a
wall at 1.5 m projects to a floor point well beyond the wall. `ground_error_m`
quantifies that rather than hiding it, and `detection_to_ground` refuses rays
that point at or above the horizon, where the intersection is at infinity and
the arithmetic silently produces an enormous number instead of an error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

#: Rotation from the optical frame to the camera body frame.
#:
#: REP-103: ROS body frames are x-forward, y-left, z-up. Vision libraries use
#: x-right, y-down, z-forward. The URDF encodes this as
#: ``rpy="-pi/2 0 -pi/2"`` on ``camera_optical_joint``, and this matrix is that
#: rotation written out, so the two can be checked against each other.
#:
#:     optical +z (forward) -> body +x (forward)
#:     optical +x (right)   -> body -y (right, since +y is left)
#:     optical +y (down)    -> body -z (down)
#:
#: Getting this wrong produces detections that are rotated 90 degrees on the
#: floor — plausible enough to be chased as a localisation bug for a long time.
R_BODY_FROM_OPTICAL: Tuple[Tuple[float, float, float], ...] = (
    (0.0, 0.0, 1.0),
    (-1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
)


@dataclass(frozen=True)
class CameraIntrinsics:
    """The pinhole model, derived from what a simulator or datasheet gives you.

    Cameras are specified by field of view; projection needs focal lengths in
    pixels. `from_fov` converts, so the numbers in the URDF stay the source of
    truth and nothing is duplicated by hand.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_fov(cls, width: int, height: int, horizontal_fov_rad: float
                 ) -> "CameraIntrinsics":
        """Intrinsics from a horizontal field of view.

        Square pixels are assumed, so ``fy == fx``. That is true of essentially
        every modern sensor and of Gazebo's camera, and assuming it keeps the
        vertical FOV consistent with the aspect ratio instead of letting the two
        drift apart.
        """
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        if not 0.0 < horizontal_fov_rad < math.pi:
            raise ValueError(
                f"horizontal FOV {horizontal_fov_rad} is not in (0, pi); a "
                f"fisheye needs a different model than a pinhole"
            )

        fx = (width / 2.0) / math.tan(horizontal_fov_rad / 2.0)
        return cls(width=width, height=height, fx=fx, fy=fx,
                   cx=width / 2.0, cy=height / 2.0)

    @property
    def horizontal_fov(self) -> float:
        return 2.0 * math.atan((self.width / 2.0) / self.fx)

    @property
    def vertical_fov(self) -> float:
        return 2.0 * math.atan((self.height / 2.0) / self.fy)

    def contains(self, u: float, v: float) -> bool:
        return 0 <= u < self.width and 0 <= v < self.height

    def ray_in_optical(self, u: float, v: float) -> Tuple[float, float, float]:
        """Unit ray through pixel (u, v), in the optical frame.

        Normalised on purpose. An unnormalised ray still intersects the ground
        plane in the right place, but every distance computed from it is scaled
        by an arbitrary factor — and that factor changes with pixel position, so
        the bug looks like a distance-dependent calibration error.
        """
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        z = 1.0
        norm = math.sqrt(x * x + y * y + z * z)
        return (x / norm, y / norm, z / norm)


@dataclass(frozen=True)
class CameraExtrinsics:
    """Where the camera sits on the robot, in the base frame.

    Defaults match ``robot.urdf.xacro``: ``camera_joint`` puts camera_link at
    (chassis_length/2, 0, chassis_height*0.75) = (0.18, 0, 0.09) relative to
    base_link, and ``base_joint`` lifts base_link by one wheel radius (0.055)
    above base_footprint. So the lens is 0.145 m above the floor.

    They are defaults rather than hard-coded constants because a real deployment
    reads them from TF. `test_bringup.py` asserts these match the URDF, so the
    two cannot drift apart silently.
    """

    x: float = 0.18
    y: float = 0.0
    #: Height above the GROUND, not above base_link. Ground-plane intersection
    #: needs the distance to the floor, and using the base_link-relative 0.09
    #: here would put every detection 60% too close.
    z: float = 0.145
    #: Radians. **Positive pitch tilts the camera DOWN.** Worth stating,
    #: because it is the opposite of the aerospace convention people carry in
    #: from elsewhere: aerospace uses x-forward/y-right/z-down and positive is
    #: nose-up, while ROS (REP-103) uses x-forward/y-left/z-up, and a positive
    #: rotation about the left-pointing y axis takes the forward vector
    #: downward. Ry(t) * (1,0,0) = (cos t, 0, -sin t).
    pitch: float = 0.0
    #: Radians, positive rotates left (toward +y).
    yaw: float = 0.0


def optical_to_body(ray: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Apply the REP-103 axis swap."""
    r = R_BODY_FROM_OPTICAL
    return (
        r[0][0] * ray[0] + r[0][1] * ray[1] + r[0][2] * ray[2],
        r[1][0] * ray[0] + r[1][1] * ray[1] + r[1][2] * ray[2],
        r[2][0] * ray[0] + r[2][1] * ray[1] + r[2][2] * ray[2],
    )


def _rotate_pitch_yaw(ray: Tuple[float, float, float], pitch: float, yaw: float
                      ) -> Tuple[float, float, float]:
    """Rotate a body-frame ray by the camera's mounting pitch and yaw."""
    x, y, z = ray
    if pitch:
        cp, sp = math.cos(pitch), math.sin(pitch)
        x, z = x * cp + z * sp, -x * sp + z * cp
    if yaw:
        cy_, sy = math.cos(yaw), math.sin(yaw)
        x, y = x * cy_ - y * sy, x * sy + y * cy_
    return (x, y, z)


#: A ray whose downward component is smaller than this is treated as parallel to
#: the floor. At 0.145 m camera height this caps the reported range at about
#: 14.5 m, which is already beyond anything a 640x480 detector resolves usefully.
MIN_DOWNWARD = 0.01


@dataclass(frozen=True)
class GroundHit:
    """Where a ray meets the floor, in the frame the ray was expressed in."""

    x: float
    y: float
    range_m: float

    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


def project_to_ground(ray_body: Tuple[float, float, float],
                      height_m: float) -> Optional[GroundHit]:
    """Intersect a body-frame ray with the plane z = -height_m.

    Returns None when the ray points at or above the horizon. That case has no
    intersection in front of the camera, and the naive arithmetic does not fail
    — it silently returns an enormous or negative coordinate. Callers that skip
    this check end up placing landmarks hundreds of metres away and then
    debugging the planner.
    """
    x, y, z = ray_body
    if z >= -MIN_DOWNWARD:
        return None

    scale = height_m / -z
    hit_x, hit_y = x * scale, y * scale
    if hit_x <= 0.0:
        # Behind the camera. Reachable through an extreme mounting pitch, and a
        # detection behind the robot is never something the camera saw.
        return None

    return GroundHit(hit_x, hit_y, math.hypot(hit_x, hit_y))


def ground_error_m(range_m: float, height_m: float, object_height_m: float) -> float:
    """How far the ground-plane assumption misplaces an object off the floor.

    By similar triangles, an object whose observed point sits ``h`` above the
    floor projects further away by ``range * h / (camera_height - h)``.

    Worth computing rather than assuming: at 0.145 m camera height, a marking
    only 0.05 m up projects **1.5x too far**. That is why the pipeline confirms
    landmarks across frames from different viewpoints rather than trusting one
    projection, and why a detection's *lowest* edge is used as its ground
    contact point.
    """
    if object_height_m <= 0:
        return 0.0
    if object_height_m >= height_m:
        return float("inf")     # the ray never reaches the floor at all
    return range_m * object_height_m / (height_m - object_height_m)


def detection_to_ground(intrinsics: CameraIntrinsics,
                        extrinsics: CameraExtrinsics,
                        u: float, v: float) -> Optional[GroundHit]:
    """Pixel to a floor coordinate in the robot's BASE frame.

    Pass the bounding box's **bottom** edge as `v`. That is where the object
    meets the floor; the centre of the box is halfway up the object and projects
    systematically too far away — by `ground_error_m` of half its height, on
    every single detection. A consistent bias is worse than noise, because
    averaging across frames does not remove it.
    """
    if not intrinsics.contains(u, v):
        return None

    ray = optical_to_body(intrinsics.ray_in_optical(u, v))
    ray = _rotate_pitch_yaw(ray, extrinsics.pitch, extrinsics.yaw)

    hit = project_to_ground(ray, extrinsics.z)
    if hit is None:
        return None
    return GroundHit(hit.x + extrinsics.x, hit.y + extrinsics.y, hit.range_m)


def base_to_map(point: Tuple[float, float],
                robot_x: float, robot_y: float, robot_yaw: float
                ) -> Tuple[float, float]:
    """Rotate and translate a base-frame point into the map frame."""
    c, s = math.cos(robot_yaw), math.sin(robot_yaw)
    x, y = point
    return (robot_x + x * c - y * s, robot_y + x * s + y * c)


def body_to_optical(ray: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Inverse of `optical_to_body`.

    R_BODY_FROM_OPTICAL is a rotation, so its inverse is its transpose. Writing
    the transpose out rather than inverting numerically keeps it exact.
    """
    r = R_BODY_FROM_OPTICAL
    return (
        r[0][0] * ray[0] + r[1][0] * ray[1] + r[2][0] * ray[2],
        r[0][1] * ray[0] + r[1][1] * ray[1] + r[2][1] * ray[2],
        r[0][2] * ray[0] + r[1][2] * ray[1] + r[2][2] * ray[2],
    )


def map_to_base(point: Tuple[float, float],
                robot_x: float, robot_y: float, robot_yaw: float
                ) -> Tuple[float, float]:
    """Inverse of `base_to_map`."""
    c, s = math.cos(robot_yaw), math.sin(robot_yaw)
    dx, dy = point[0] - robot_x, point[1] - robot_y
    return (dx * c + dy * s, -dx * s + dy * c)


def world_to_pixel(intrinsics: CameraIntrinsics, extrinsics: CameraExtrinsics,
                   point_base: Tuple[float, float],
                   height_m: float = 0.0) -> Optional[Tuple[float, float]]:
    """Project a point in the robot BASE frame back to a pixel.

    The inverse of `detection_to_ground`, and it exists for two reasons beyond
    rendering: it makes the forward geometry testable by round trip
    (world -> pixel -> world must return the original point), and it is what
    lets a synthetic detector produce boxes from a known map without any model.

    `height_m` is the point's height above the floor, so the top and bottom of
    an object can both be projected and a real box drawn.

    Returns None when the point is behind the camera or off the sensor.
    """
    x = point_base[0] - extrinsics.x
    y = point_base[1] - extrinsics.y
    z = height_m - extrinsics.z

    # Undo the mounting rotation: yaw first, then pitch, each negated, which is
    # the reverse order of _rotate_pitch_yaw.
    if extrinsics.yaw:
        c, s = math.cos(-extrinsics.yaw), math.sin(-extrinsics.yaw)
        x, y = x * c - y * s, x * s + y * c
    if extrinsics.pitch:
        c, s = math.cos(-extrinsics.pitch), math.sin(-extrinsics.pitch)
        x, z = x * c + z * s, -x * s + z * c

    ox, oy, oz = body_to_optical((x, y, z))
    if oz <= 1e-9:
        return None      # at or behind the image plane

    u = intrinsics.fx * ox / oz + intrinsics.cx
    v = intrinsics.fy * oy / oz + intrinsics.cy
    if not intrinsics.contains(u, v):
        return None
    return (u, v)


def horizon_row(intrinsics: CameraIntrinsics, extrinsics: CameraExtrinsics
                ) -> float:
    """The image row where the floor recedes to infinity.

    Everything above it is wall, ceiling or sky and cannot be projected onto the
    floor at all. Useful as a sanity check on a whole frame, and as a cheap
    filter: a detection whose bottom edge sits above this row is not standing on
    the ground the robot is driving on.
    """
    if extrinsics.pitch == 0.0:
        return intrinsics.cy
    return intrinsics.cy - intrinsics.fy * math.tan(extrinsics.pitch)
