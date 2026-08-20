"""2D pose maths — quaternions, yaw, angle wrapping, differential-drive control.

Small file, disproportionate number of production robotics bugs. Three of them
are pinned by tests in this package:

  * **Angle wrapping.** The difference between 179° and −179° is 2°, not 358°.
    An unwrapped controller sees the large number, spins the long way round,
    and looks broken in a way that is hard to attribute to one line of maths.
  * **Quaternion sign.** q and −q are the same rotation. Comparing quaternions
    componentwise reports a difference that does not exist.
  * **atan2 vs atan.** atan loses the quadrant, so a robot facing backwards is
    reported as facing forwards.

No ROS imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


def normalise_angle(angle: float) -> float:
    """Wrap to (-pi, pi].

    The half-open interval needs the epsilon. atan2 returns -pi at the seam,
    and for inputs like -3*pi the round-trip through sin/cos lands a few ulp
    *inside* it (-3.1415926535897927), which an exact `== -math.pi` comparison
    misses. The result is a value that is neither in the interval this function
    promises nor equal to the boundary it checks for.
    """
    wrapped = math.atan2(math.sin(angle), math.cos(angle))
    return math.pi if wrapped <= -math.pi + 1e-12 else wrapped


def angle_difference(target: float, current: float) -> float:
    """Shortest signed rotation from `current` to `target`."""
    return normalise_angle(target - current)


def yaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
    """Yaw about z as (x, y, z, w) — the order ROS geometry_msgs uses.

    scipy and several other libraries use (w, x, y, z). Getting the order wrong
    produces a rotation that is wrong but *plausible*, which is worse than one
    that is obviously wrong.
    """
    half = yaw / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Extract yaw from a quaternion, assuming roll and pitch are small."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass(frozen=True)
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    def distance_to(self, other: "Pose2D") -> float:
        return math.hypot(other.x - self.x, other.y - self.y)

    def heading_to(self, other: "Pose2D") -> float:
        return math.atan2(other.y - self.y, other.x - self.x)

    def bearing_to(self, other: "Pose2D") -> float:
        """Heading to `other` relative to where this pose is already facing."""
        return angle_difference(self.heading_to(other), self.yaw)

    def to_quaternion(self) -> Tuple[float, float, float, float]:
        return yaw_to_quaternion(self.yaw)

    @classmethod
    def from_ros(cls, msg) -> "Pose2D":
        """Build from geometry_msgs/Pose without importing it."""
        q = msg.orientation
        return cls(msg.position.x, msg.position.y,
                   quaternion_to_yaw(q.x, q.y, q.z, q.w))


def unicycle_control(current: Pose2D, target: Pose2D,
                     linear_gain: float = 0.6, angular_gain: float = 1.5,
                     heading_tolerance: float = math.radians(35),
                     position_tolerance: float = 0.12) -> Tuple[float, float]:
    """Proportional controller for a differential-drive base.

    Returns (linear_x, angular_z) *before* the safety governor sees it — this
    function has no authority to move anything on its own, which is the point of
    keeping the two separate.

    Turn-then-drive rather than turn-while-driving: if the heading error is
    larger than `heading_tolerance` the linear term is suppressed. A robot that
    drives forward while badly misaligned traces a long arc away from the goal
    before curving back, and on a narrow corridor that arc is a wall.
    """
    distance = current.distance_to(target)
    if distance <= position_tolerance:
        # Close enough in position — settle the final heading only.
        return 0.0, angular_gain * angle_difference(target.yaw, current.yaw)

    heading_error = current.bearing_to(target)
    angular = angular_gain * heading_error
    linear = 0.0 if abs(heading_error) > heading_tolerance else linear_gain * distance
    return linear, angular
