"""Velocity safety governor — the last thing between a plan and a motor.

This is the most important module in the repository and the one most portfolio
robotics projects do not have at all.

The premise: **everything upstream can be wrong.** The LLM can misunderstand
the command, the planner can produce a path through a wall, the localiser can
drift, a node can crash mid-motion. A safety layer that sits at the very bottom
of the stack and clamps the commanded velocity does not care *why* the command
is bad — it only cares that the robot must not drive into something or keep
moving when nobody is talking to it.

Three properties make it a safety layer rather than a suggestion:

  1. **It is the last writer.** Nothing downstream can override it.
  2. **It fails closed.** Missing data, stale data and errors all produce
     *stop*, never *proceed*.
  3. **It is independent of the AI stack.** It uses raw range data, not
     anything a model produced.

No ROS imports, so every rule below is unit-tested directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class Twist:
    """Linear and angular velocity — mirrors geometry_msgs/Twist (2D subset)."""

    linear_x: float = 0.0
    angular_z: float = 0.0

    def is_stopped(self) -> bool:
        return abs(self.linear_x) < 1e-9 and abs(self.angular_z) < 1e-9


STOP = Twist(0.0, 0.0)


@dataclass(frozen=True)
class SafetyLimits:
    max_linear: float = 0.45          # m/s
    max_angular: float = 1.2          # rad/s
    max_linear_accel: float = 0.6     # m/s^2
    max_angular_accel: float = 2.0    # rad/s^2

    # Distances measured from the robot's footprint edge.
    stop_distance: float = 0.35       # hard stop inside this
    slow_distance: float = 1.00       # begin scaling here

    # A scan older than this is not evidence about the present.
    scan_timeout_s: float = 0.5

    # Forward-facing arc considered for obstacles, in radians either side of
    # straight ahead. A differential-drive robot driving forward does not care
    # about something directly beside it.
    front_arc_rad: float = math.radians(60)


@dataclass
class SafetyDecision:
    command: Twist
    limited: bool = False
    reasons: List[str] = field(default_factory=list)
    nearest_obstacle_m: Optional[float] = None
    scale: float = 1.0

    @property
    def stopped(self) -> bool:
        return self.command.is_stopped()


class SafetyGovernor:
    """Clamps a desired velocity to something safe to execute right now."""

    def __init__(self, limits: Optional[SafetyLimits] = None):
        self.limits = limits or SafetyLimits()
        self._last = STOP
        self._last_time: Optional[float] = None
        self._estopped = False

    # -- emergency stop ------------------------------------------------------

    def engage_estop(self) -> None:
        self._estopped = True

    def release_estop(self) -> None:
        """Deliberately explicit. An e-stop that clears itself is not an e-stop
        — a human decides when the situation is safe again."""
        self._estopped = False
        self._last = STOP

    @property
    def estopped(self) -> bool:
        return self._estopped

    # -- obstacle distance ---------------------------------------------------

    def nearest_in_arc(self, ranges: Sequence[float], angle_min: float,
                       angle_increment: float) -> Optional[float]:
        """Closest valid return within the forward arc.

        NaN and inf are dropped rather than treated as "far". A lidar reports
        inf for no-return, which frequently means a glass wall or a black
        surface absorbed the beam — the two obstacles most likely to be hit.
        Treating them as free space is precisely backwards, so they are ignored
        and the caller's stale/empty handling takes over.
        """
        nearest: Optional[float] = None
        for i, value in enumerate(ranges):
            if value is None or math.isnan(value) or math.isinf(value):
                continue
            if value <= 0.0:
                continue
            angle = angle_min + i * angle_increment
            if abs(angle) > self.limits.front_arc_rad:
                continue
            if nearest is None or value < nearest:
                nearest = value
        return nearest

    # -- the governor --------------------------------------------------------

    def limit(self, desired: Twist, now: float,
              nearest_obstacle_m: Optional[float] = None,
              scan_age_s: Optional[float] = None) -> SafetyDecision:
        reasons: List[str] = []

        # 1. E-stop beats everything.
        if self._estopped:
            self._last = STOP
            self._last_time = now
            return SafetyDecision(STOP, limited=True, reasons=["e-stop engaged"])

        # 2. Fail closed on missing or stale perception. "We have no idea what
        #    is in front of us" must mean stop, not proceed at full speed.
        if scan_age_s is None:
            self._last = STOP
            self._last_time = now
            return SafetyDecision(
                STOP, limited=True, reasons=["no scan received yet"],
            )
        if scan_age_s > self.limits.scan_timeout_s:
            self._last = STOP
            self._last_time = now
            return SafetyDecision(
                STOP, limited=True,
                reasons=[f"scan is {scan_age_s:.2f}s old "
                         f"(limit {self.limits.scan_timeout_s}s)"],
            )

        linear = desired.linear_x
        angular = desired.angular_z
        scale = 1.0

        # 3. Absolute limits FIRST, obstacle scaling second. The order is not
        #    cosmetic. Scaling before clamping means a command of 50 m/s in
        #    front of a wall scales to 25 m/s, still clamps to the maximum, and
        #    the robot drives at full speed into the obstacle — the scaling had
        #    no effect at all. Clamping first makes the scale factor apply to
        #    the speed actually achievable.
        clamped_linear = max(-self.limits.max_linear,
                             min(self.limits.max_linear, linear))
        clamped_angular = max(-self.limits.max_angular,
                              min(self.limits.max_angular, angular))
        if clamped_linear != linear:
            reasons.append(f"linear clamped to {self.limits.max_linear} m/s")
        if clamped_angular != angular:
            reasons.append(f"angular clamped to {self.limits.max_angular} rad/s")
        linear, angular = clamped_linear, clamped_angular

        # 4. Obstacle scaling — only for forward motion. Reversing away from
        #    something in front is the correct recovery, so it must not be
        #    blocked by the thing being escaped.
        if nearest_obstacle_m is not None and linear > 0:
            if nearest_obstacle_m <= self.limits.stop_distance:
                scale = 0.0
                reasons.append(
                    f"obstacle at {nearest_obstacle_m:.2f}m is inside the "
                    f"{self.limits.stop_distance}m stop distance"
                )
            elif nearest_obstacle_m < self.limits.slow_distance:
                span = self.limits.slow_distance - self.limits.stop_distance
                scale = (nearest_obstacle_m - self.limits.stop_distance) / span
                reasons.append(
                    f"scaled to {scale:.0%} for an obstacle at "
                    f"{nearest_obstacle_m:.2f}m"
                )
            linear *= scale

        # 5. Acceleration limits. A step change in commanded velocity makes the
        #    wheels slip, which corrupts the odometry, which corrupts
        #    localisation — a control problem that presents as a mapping bug.
        if self._last_time is not None:
            dt = max(1e-3, now - self._last_time)
            max_dv = self.limits.max_linear_accel * dt
            max_dw = self.limits.max_angular_accel * dt

            if abs(linear - self._last.linear_x) > max_dv:
                linear = self._last.linear_x + math.copysign(
                    max_dv, linear - self._last.linear_x
                )
                reasons.append("linear acceleration limited")
            if abs(angular - self._last.angular_z) > max_dw:
                angular = self._last.angular_z + math.copysign(
                    max_dw, angular - self._last.angular_z
                )
                reasons.append("angular acceleration limited")

        command = Twist(linear, angular)
        self._last = command
        self._last_time = now

        return SafetyDecision(
            command=command,
            limited=bool(reasons),
            reasons=reasons,
            nearest_obstacle_m=nearest_obstacle_m,
            scale=scale,
        )
