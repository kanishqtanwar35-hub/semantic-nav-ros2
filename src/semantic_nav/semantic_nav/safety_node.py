"""ROS node: the velocity governor, wired in as the last writer.

Topology matters more than code here:

    Nav2 / teleop / LLM  ──►  /cmd_vel_raw  ──►  [this node]  ──►  /cmd_vel  ──►  base

Nothing publishes to `/cmd_vel` except this node. That single remapping is what
turns `robot_core.safety` from a library someone might call into a guarantee:
there is no way for any other node to reach the wheels without passing through
here, including nodes written later by someone who never read this file.

The node itself is trivial, which is the point — the rules live in
`robot_core/safety.py` where 30-odd tests exercise them without ROS.
"""

import math

import rclpy
from geometry_msgs.msg import Twist as TwistMsg
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String

from robot_core.safety import SafetyGovernor, SafetyLimits, Twist


class SafetyNode(Node):

    def __init__(self):
        super().__init__("safety_governor")

        self.declare_parameter("max_linear", 0.45)
        self.declare_parameter("max_angular", 1.2)
        self.declare_parameter("max_linear_accel", 0.6)
        self.declare_parameter("max_angular_accel", 2.0)
        self.declare_parameter("stop_distance", 0.35)
        self.declare_parameter("slow_distance", 1.0)
        self.declare_parameter("scan_timeout", 0.5)
        self.declare_parameter("front_arc_deg", 60.0)
        self.declare_parameter("rate_hz", 20.0)

        def p(name):
            return self.get_parameter(name).value

        self.governor = SafetyGovernor(SafetyLimits(
            max_linear=float(p("max_linear")),
            max_angular=float(p("max_angular")),
            max_linear_accel=float(p("max_linear_accel")),
            max_angular_accel=float(p("max_angular_accel")),
            stop_distance=float(p("stop_distance")),
            slow_distance=float(p("slow_distance")),
            scan_timeout_s=float(p("scan_timeout")),
            front_arc_rad=math.radians(float(p("front_arc_deg"))),
        ))

        self._desired = Twist()
        self._nearest = None
        self._scan_time = None
        self._command_time = None

        # Sensor QoS: best-effort, keep-last-1. A lidar publishes faster than
        # anything can consume it, and a reliable queue would deliver a backlog
        # of stale scans — old range data presented as current, which is the
        # one thing this node must never act on.
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        self.create_subscription(LaserScan, "scan", self._on_scan, sensor_qos)
        self.create_subscription(TwistMsg, "cmd_vel_raw", self._on_command, 10)
        self.create_subscription(Bool, "estop", self._on_estop, 10)

        self._cmd_pub = self.create_publisher(TwistMsg, "cmd_vel", 10)
        self._status_pub = self.create_publisher(String, "safety_status", 10)

        # A fixed-rate timer, not a passthrough on the command callback. If the
        # publisher of cmd_vel_raw dies mid-motion its last command would
        # otherwise be the robot's standing order forever; the timer keeps
        # evaluating, the scan goes stale, and the robot stops on its own.
        self._period = 1.0 / float(p("rate_hz"))
        self.create_timer(self._period, self._tick)

        self._last_reason = ""
        self.get_logger().info(
            f"governor active: cmd_vel_raw -> cmd_vel at {p('rate_hz')} Hz"
        )

    # -- callbacks ----------------------------------------------------------

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_scan(self, message: LaserScan) -> None:
        self._nearest = self.governor.nearest_in_arc(
            message.ranges, message.angle_min, message.angle_increment
        )
        self._scan_time = self._now()

    def _on_command(self, message: TwistMsg) -> None:
        self._desired = Twist(message.linear.x, message.angular.z)
        self._command_time = self._now()

    def _on_estop(self, message: Bool) -> None:
        if message.data:
            self.governor.engage_estop()
            self.get_logger().warning("E-STOP ENGAGED")
        else:
            self.governor.release_estop()
            self.get_logger().warning("e-stop released")

    # -- control loop -------------------------------------------------------

    def _tick(self) -> None:
        now = self._now()

        # An old velocity command is not a standing order. Two consecutive
        # missed cycles and the robot coasts to a stop by itself.
        desired = self._desired
        if self._command_time is None or (now - self._command_time) > 3 * self._period:
            desired = Twist()

        scan_age = None if self._scan_time is None else now - self._scan_time
        decision = self.governor.limit(desired, now,
                                       nearest_obstacle_m=self._nearest,
                                       scan_age_s=scan_age)

        message = TwistMsg()
        message.linear.x = decision.command.linear_x
        message.angular.z = decision.command.angular_z
        self._cmd_pub.publish(message)

        reason = "; ".join(decision.reasons)
        if reason and reason != self._last_reason:
            # Log on change, not every cycle. At 20 Hz an unconditional log
            # buries every other message on the system inside a minute.
            self.get_logger().info(f"limiting: {reason}")
            self._status_pub.publish(String(data=reason))
        self._last_reason = reason


def main(args=None):
    rclpy.init(args=args)
    node = SafetyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
