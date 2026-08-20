"""Navigation backends and the mission runner.

Two implementations behind one interface:

  `SimulatedNavigator` — pure Python. Plans with A* on the inflated grid and
  walks the path. No ROS, no Nav2, no Gazebo. This is what CI runs, and it is
  what lets the mission logic below be tested at all.

  `Nav2Navigator` — sends `NavigateToPose` goals to the real stack. Every ROS
  import inside it is **lazy**, so importing this module on a machine with no
  ROS installed works fine and the failure only happens if you actually try to
  drive a real robot without a robot.

The lazy import is not a trick to make tests pass. It is what keeps the mission
runner honest: because the runner cannot see any ROS type, it cannot come to
depend on one, and the same code path is exercised in CI and on hardware.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from robot_core.geometry import Pose2D
from robot_core.occupancy import OccupancyGrid
from robot_core.planners import astar, simplify
from semantic_nav.commands import Step, ValidatedCommand, Verb


@dataclass
class StepOutcome:
    step: Step
    ok: bool
    detail: str = ""
    distance_m: float = 0.0
    duration_s: float = 0.0


@dataclass
class MissionOutcome:
    ok: bool
    outcomes: List[StepOutcome] = field(default_factory=list)
    aborted_at: Optional[int] = None
    reason: str = ""

    @property
    def distance_m(self) -> float:
        return sum(o.distance_m for o in self.outcomes)

    def summary(self) -> str:
        lines = [f"{'OK ' if self.ok else 'FAIL'}  "
                 f"{len(self.outcomes)} step(s), {self.distance_m:.2f} m"]
        for index, outcome in enumerate(self.outcomes, start=1):
            mark = "ok" if outcome.ok else "!!"
            lines.append(f"  {mark} {index}. {outcome.step.verb.value:<12} "
                         f"{outcome.detail}")
        if self.reason:
            lines.append(f"  reason: {self.reason}")
        return "\n".join(lines)


class NavigationFailed(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Simulated
# ---------------------------------------------------------------------------

class SimulatedNavigator:
    """Plans and 'drives' in memory. Deterministic, instant, no ROS."""

    def __init__(self, grid: OccupancyGrid, start: Pose2D = Pose2D(),
                 robot_radius_m: float = 0.22, speed_mps: float = 0.4):
        self.grid = grid
        # Inflate once, not per goal. Inflation is O(cells x kernel) and the map
        # does not change between goals in this simulation.
        self.planning_grid = grid.inflate(robot_radius_m)
        self.pose = start
        self.speed_mps = speed_mps
        self.trail: List[Pose2D] = [start]
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def go_to(self, x: float, y: float, yaw: float = 0.0) -> StepOutcome:
        start_cell = self.planning_grid.world_to_cell(self.pose.x, self.pose.y)
        goal_cell = self.planning_grid.world_to_cell(x, y)

        result = astar(self.planning_grid, start_cell, goal_cell)
        if not result.success:
            # Failing with the planner's own reason matters. "Navigation
            # failed" tells an operator nothing; "goal is in an obstacle" tells
            # them the map is wrong, and "no path exists" tells them a door is
            # shut.
            return StepOutcome(Step(Verb.GO_TO), ok=False, detail=result.reason)

        path = simplify(self.planning_grid, result.path)
        distance = 0.0
        previous = self.pose
        for cell in path:
            cx, cy = self.planning_grid.cell_to_world(cell)
            waypoint = Pose2D(cx, cy, previous.heading_to(Pose2D(cx, cy)))
            distance += previous.distance_to(waypoint)
            self.trail.append(waypoint)
            previous = waypoint

        self.pose = Pose2D(x, y, yaw)
        self.trail.append(self.pose)
        return StepOutcome(
            Step(Verb.GO_TO), ok=True,
            detail=f"arrived at ({x:.2f}, {y:.2f}) via {len(path)} waypoints",
            distance_m=distance,
            duration_s=distance / self.speed_mps if self.speed_mps else 0.0,
        )


# ---------------------------------------------------------------------------
# Nav2
# ---------------------------------------------------------------------------

class Nav2Navigator:
    """Sends NavigateToPose goals to a running Nav2 stack.

    Every ROS import happens inside a method. Importing this file on a laptop
    with no ROS works; calling `go_to` without a stack raises, which is the
    correct place for that failure to appear.
    """

    def __init__(self, node=None, timeout_s: float = 120.0):
        self._node = node
        self._client = None
        self.timeout_s = timeout_s

    def _ensure_client(self):
        if self._client is not None:
            return self._client

        try:
            import rclpy
            from rclpy.action import ActionClient
            from nav2_msgs.action import NavigateToPose
        except ImportError as error:
            raise NavigationFailed(
                "Nav2 is not available in this environment; use "
                "SimulatedNavigator or source a ROS 2 workspace"
            ) from error

        if self._node is None:
            if not rclpy.ok():
                rclpy.init()
            self._node = rclpy.create_node("semantic_nav_client")

        self._client = ActionClient(self._node, NavigateToPose, "navigate_to_pose")
        if not self._client.wait_for_server(timeout_sec=10.0):
            raise NavigationFailed("no navigate_to_pose action server after 10s")
        return self._client

    def _pose_stamped(self, x: float, y: float, yaw: float):
        from geometry_msgs.msg import PoseStamped
        from robot_core.geometry import yaw_to_quaternion

        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = self._node.get_clock().now().to_msg()
        goal.pose.position.x = float(x)
        goal.pose.position.y = float(y)
        qx, qy, qz, qw = yaw_to_quaternion(float(yaw))
        goal.pose.orientation.x = qx
        goal.pose.orientation.y = qy
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw
        return goal

    def cancel(self) -> None:
        if self._client is not None:
            self._client.destroy()
            self._client = None

    def go_to(self, x: float, y: float, yaw: float = 0.0) -> StepOutcome:
        import rclpy
        from nav2_msgs.action import NavigateToPose

        client = self._ensure_client()
        goal = NavigateToPose.Goal()
        goal.pose = self._pose_stamped(x, y, yaw)

        send = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self._node, send, timeout_sec=self.timeout_s)
        handle = send.result()
        if handle is None or not handle.accepted:
            return StepOutcome(Step(Verb.GO_TO), ok=False,
                               detail="Nav2 rejected the goal")

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self._node, result_future,
                                         timeout_sec=self.timeout_s)
        result = result_future.result()
        if result is None:
            return StepOutcome(Step(Verb.GO_TO), ok=False,
                               detail=f"Nav2 did not finish within {self.timeout_s}s")

        # status 4 is STATUS_SUCCEEDED in action_msgs/GoalStatus.
        ok = getattr(result, "status", 4) == 4
        return StepOutcome(
            Step(Verb.GO_TO), ok=ok,
            detail=f"Nav2 {'succeeded' if ok else 'failed'} at ({x:.2f}, {y:.2f})",
        )


# ---------------------------------------------------------------------------
# Mission runner
# ---------------------------------------------------------------------------

class MissionRunner:
    """Executes a ValidatedCommand step by step.

    Note the signature: it takes a `ValidatedCommand`, not a `Command` and not a
    string. There is no code path from raw model output to a motion goal that
    skips validation, and that is enforced by the type rather than by a comment
    asking people to be careful.
    """

    def __init__(self, navigator, sleep: Callable[[float], None] = time.sleep,
                 on_report: Optional[Callable[[str], None]] = None):
        self.navigator = navigator
        self._sleep = sleep          # injected so tests do not actually wait
        self._on_report = on_report
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True
        cancel = getattr(self.navigator, "cancel", None)
        if callable(cancel):
            cancel()

    def run(self, command: ValidatedCommand) -> MissionOutcome:
        self._stopped = False
        outcomes: List[StepOutcome] = []

        for index, (step, pose) in enumerate(zip(command.steps, command.poses)):
            if self._stopped:
                return MissionOutcome(False, outcomes, aborted_at=index,
                                      reason="stopped by operator")

            if step.verb is Verb.STOP:
                self.stop()
                outcomes.append(StepOutcome(step, ok=True, detail="stopped"))
                return MissionOutcome(True, outcomes, reason="stop requested")

            if step.verb is Verb.WAIT:
                self._sleep(step.seconds or 0.0)
                outcomes.append(StepOutcome(step, ok=True,
                                            detail=f"waited {step.seconds:g}s",
                                            duration_s=step.seconds or 0.0))
                continue

            if step.verb is Verb.REPORT:
                pose_now = getattr(self.navigator, "pose", None)
                text = (f"at ({pose_now.x:.2f}, {pose_now.y:.2f})"
                        if pose_now is not None else "position unknown")
                if self._on_report:
                    self._on_report(text)
                outcomes.append(StepOutcome(step, ok=True, detail=text))
                continue

            if not pose:
                outcomes.append(StepOutcome(step, ok=False,
                                            detail="no pose for this step"))
                return MissionOutcome(False, outcomes, aborted_at=index,
                                      reason="internal: step had no pose")

            outcome = self.navigator.go_to(*pose)
            outcome.step = step
            outcomes.append(outcome)

            if not outcome.ok:
                # Abort rather than continue. Later steps were planned assuming
                # the robot got where it was going; running them from the wrong
                # place is how a failed navigation becomes a robot somewhere
                # nobody expects.
                return MissionOutcome(False, outcomes, aborted_at=index,
                                      reason=outcome.detail)

        return MissionOutcome(True, outcomes)
