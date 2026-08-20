"""ROS node: publish the semantic map as RViz markers.

Small, and worth having. The gap between "the robot went to the wrong place"
and "the landmark is defined in the wrong place" is invisible from a log and
obvious from a picture, and most of the semantic-navigation bugs that survive
unit testing are map-data bugs rather than code bugs.

Publishes two markers per landmark: a sphere at the landmark itself and an
arrow at its approach pose, so you can see the difference between *where the
desk is* and *where the robot will stand* — the distinction that stops goals
being placed inside furniture.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from visualization_msgs.msg import Marker, MarkerArray

from robot_core.semantic_map import SemanticMap

CATEGORY_COLOURS = {
    "room":      (0.20, 0.60, 1.00),
    "furniture": (1.00, 0.65, 0.10),
    "object":    (0.60, 0.30, 0.90),
    "waypoint":  (0.20, 0.85, 0.35),
}
DEFAULT_COLOUR = (0.70, 0.70, 0.70)


class LandmarkMarkersNode(Node):

    def __init__(self):
        super().__init__("landmark_markers")
        self.declare_parameter("semantic_map", "")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("period", 2.0)

        path = self.get_parameter("semantic_map").value
        self.semantic_map = SemanticMap.load(path) if path else SemanticMap()
        self.frame_id = self.get_parameter("frame_id").value

        # Transient-local so RViz started *after* this node still receives the
        # markers. With volatile durability the map appears only if you happen
        # to open RViz first, which reads as "the node is broken".
        qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self._pub = self.create_publisher(MarkerArray, "landmarks", qos)

        self.create_timer(float(self.get_parameter("period").value), self._publish)
        self._publish()
        self.get_logger().info(f"publishing {len(self.semantic_map)} landmark(s)")

    def _publish(self) -> None:
        array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        for index, landmark in enumerate(self.semantic_map):
            r, g, b = CATEGORY_COLOURS.get(landmark.category, DEFAULT_COLOUR)

            body = Marker()
            body.header.frame_id = self.frame_id
            body.header.stamp = stamp
            body.ns = "landmark"
            body.id = index
            body.type = Marker.SPHERE
            body.action = Marker.ADD
            body.pose.position.x = float(landmark.x)
            body.pose.position.y = float(landmark.y)
            body.pose.position.z = 0.15
            body.pose.orientation.w = 1.0
            diameter = max(0.2, float(landmark.radius_m) * 2.0)
            body.scale.x = body.scale.y = body.scale.z = diameter
            body.color.r, body.color.g, body.color.b = r, g, b
            body.color.a = 0.45
            array.markers.append(body)

            label = Marker()
            label.header.frame_id = self.frame_id
            label.header.stamp = stamp
            label.ns = "landmark_label"
            label.id = index
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = float(landmark.x)
            label.pose.position.y = float(landmark.y)
            label.pose.position.z = 0.15 + diameter / 2 + 0.2
            label.pose.orientation.w = 1.0
            label.scale.z = 0.28
            label.color.r = label.color.g = label.color.b = 1.0
            label.color.a = 0.95
            label.text = landmark.name
            array.markers.append(label)

            approach = landmark.approach_pose
            arrow = Marker()
            arrow.header.frame_id = self.frame_id
            arrow.header.stamp = stamp
            arrow.ns = "approach"
            arrow.id = index
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.pose.position.x = float(approach.x)
            arrow.pose.position.y = float(approach.y)
            arrow.pose.position.z = 0.05
            qx, qy, qz, qw = approach.to_quaternion()
            arrow.pose.orientation.x = qx
            arrow.pose.orientation.y = qy
            arrow.pose.orientation.z = qz
            arrow.pose.orientation.w = qw
            arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.45, 0.07, 0.07
            arrow.color.r, arrow.color.g, arrow.color.b = r, g, b
            arrow.color.a = 0.95
            array.markers.append(arrow)

        self._pub.publish(array)


def main(args=None):
    rclpy.init(args=args)
    node = LandmarkMarkersNode()
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
