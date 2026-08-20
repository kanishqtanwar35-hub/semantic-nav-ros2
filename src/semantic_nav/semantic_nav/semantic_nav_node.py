"""ROS node: natural language in, Nav2 goals out.

Deliberately thin. Everything this file does is: receive a string, hand it to
`Grounder`, hand the result to `validate`, hand that to `MissionRunner`, and
publish what happened. There is no algorithm here, because anything with an
algorithm in it belongs in `robot_core` or in the plain-Python modules of this
package where a test can reach it without starting a ROS graph.

If you find yourself adding an `if` to this file, that `if` probably belongs
somewhere testable.
"""

import threading

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from robot_core.geometry import Pose2D
from robot_core.occupancy import OccupancyGrid
from robot_core.semantic_map import SemanticMap
from semantic_nav.commands import ValidationError, Verb, validate
from semantic_nav.grounding import Grounder
from semantic_nav.llm import default_backend
from semantic_nav.nav_client import MissionRunner, Nav2Navigator, SimulatedNavigator


class SemanticNavNode(Node):

    def __init__(self):
        super().__init__("semantic_nav")

        self.declare_parameter("semantic_map", "")
        self.declare_parameter("use_llm", True)
        self.declare_parameter("use_nav2", True)
        self.declare_parameter("robot_radius", 0.22)

        map_path = self.get_parameter("semantic_map").value
        self.semantic_map = (SemanticMap.load(map_path) if map_path
                             else SemanticMap())
        self.get_logger().info(
            f"semantic map: {len(self.semantic_map)} landmark(s)"
        )

        backend = default_backend() if self.get_parameter("use_llm").value else None
        if backend is None:
            # Said out loud on purpose. "The robot understands fewer phrasings
            # today" is something an operator should learn from a startup log,
            # not from a command that mysteriously stops working.
            self.get_logger().info(
                "no GEMINI_API_KEY — rule parser only; core commands unaffected"
            )
        self.grounder = Grounder(self.semantic_map, llm=backend,
                                 use_llm=backend is not None)

        if self.get_parameter("use_nav2").value:
            self.navigator = Nav2Navigator(node=self)
        else:
            self.navigator = SimulatedNavigator(
                OccupancyGrid(200, 200, 0.05, origin_x=-5.0, origin_y=-5.0),
                start=Pose2D(),
                robot_radius_m=float(self.get_parameter("robot_radius").value),
            )

        self.runner = MissionRunner(self.navigator,
                                    on_report=lambda text: self._say(text))

        group = ReentrantCallbackGroup()
        self._feedback = self.create_publisher(String, "nav_feedback", 10)
        self.create_subscription(String, "nl_command", self._on_command, 10,
                                 callback_group=group)

        # A mission runs on its own thread so the subscription stays live. If
        # it did not, "stop" would sit in the queue behind the mission it is
        # trying to interrupt — the failure mode that makes an emergency stop
        # useless exactly when it is needed.
        self._mission_thread = None
        self._lock = threading.Lock()

        self.get_logger().info("listening on /nl_command")

    def _say(self, text: str) -> None:
        self.get_logger().info(text)
        self._feedback.publish(String(data=text))

    def _on_command(self, message: String) -> None:
        utterance = (message.data or "").strip()
        if not utterance:
            return

        command = self.grounder.ground(utterance)

        # STOP is handled before anything else and without touching the mission
        # thread's lock, because acquiring a lock held by the thing you are
        # trying to interrupt is a deadlock with wheels on it.
        if any(step.verb is Verb.STOP for step in command.steps):
            self.runner.stop()
            self._say("stopping")
            return

        try:
            validated = validate(command, self.semantic_map)
        except ValidationError as error:
            self._say(f"cannot do that: {error}")
            return

        with self._lock:
            if self._mission_thread and self._mission_thread.is_alive():
                self._say("busy — say 'stop' first")
                return
            self._say(f"[{command.source}] plan:\n{validated.describe()}")
            self._mission_thread = threading.Thread(
                target=self._run, args=(validated,), daemon=True
            )
            self._mission_thread.start()

    def _run(self, validated) -> None:
        try:
            outcome = self.runner.run(validated)
        except Exception as error:                        # noqa: BLE001
            # A crashed mission thread must not take the node with it. The node
            # is what still receives "stop".
            self.get_logger().error(f"mission crashed: {error!r}")
            self._say(f"mission failed: {type(error).__name__}")
            return
        self._say(outcome.summary())


def main(args=None):
    rclpy.init(args=args)
    node = SemanticNavNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
