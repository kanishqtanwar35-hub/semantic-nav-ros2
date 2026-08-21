"""ROS node: camera frames in, landmark proposals out.

Thin, like the other nodes in this workspace. It reads an image, asks TF where
the robot was **when that image was taken**, calls four functions from the
library half of this package, and publishes. There is no algorithm in this file,
because anything with an algorithm in it belongs where a test can reach it
without starting a ROS graph.

Three things here are ROS-specific and worth reading:

**The pose comes from the image's own timestamp**, not from "now". A detector
that takes 300 ms to run is 300 ms behind the camera, and grounding its output
against the robot's current pose places every landmark wherever the robot has
driven to since. That failure looks like a smear of landmarks along the patrol
route and reads as a mapping bug rather than a latency one.

**Frames are dropped, not queued.** The detector is slower than the camera on a
CPU, so a queue can only grow. Dropping keeps the newest frame and bounds the
staleness; queueing unbounds it.

**Nothing is written to the map automatically.** The node publishes proposals
and only commits them when asked, because a model writing directly into the
map the navigation stack trusts is the thing this whole package is arguing
against.
"""

import threading

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from robot_core.geometry import quaternion_to_yaw
from robot_core.semantic_map import SemanticMap
from robot_perception.camera import (
    CameraExtrinsics,
    CameraIntrinsics,
    base_to_map,
    detection_to_ground,
)
from robot_perception.detection import filter_detections, load_backend, summarise
from robot_perception.mapping import apply, make_tracker, propose
from robot_perception.tracking import Observation


def image_to_array(message: Image):
    """Decode a sensor_msgs/Image without cv_bridge.

    cv_bridge is a common install headache and pulls OpenCV into the node's
    process for what is, for the encodings a camera driver actually publishes,
    a reshape of a buffer. Avoiding the dependency is worth five lines.
    """
    import numpy as np

    encoding = message.encoding.lower()
    if encoding in ("rgb8", "bgr8"):
        channels = 3
    elif encoding in ("mono8",):
        channels = 1
    elif encoding in ("rgba8", "bgra8"):
        channels = 4
    else:
        raise ValueError(f"unsupported image encoding: {message.encoding}")

    array = np.frombuffer(message.data, dtype=np.uint8)
    array = array.reshape((message.height, message.width, channels))

    # Detectors expect RGB. Publishing BGR into one that expects RGB does not
    # error - it quietly costs accuracy, which is the worst kind of mistake to
    # make in a perception pipeline.
    if encoding.startswith("bgr"):
        array = array[:, :, ::-1]
    return array[:, :, :3] if channels >= 3 else array


class PerceptionNode(Node):

    def __init__(self):
        super().__init__("robot_perception")

        self.declare_parameter("backend", "auto")
        self.declare_parameter("min_confidence", 0.35)
        self.declare_parameter("semantic_map", "")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("detect_hz", 2.0)
        self.declare_parameter("camera_fov", 1.089)
        self.declare_parameter("camera_height", 0.145)
        self.declare_parameter("camera_offset_x", 0.18)
        self.declare_parameter("auto_commit", False)

        def p(name):
            return self.get_parameter(name).value

        self.map_frame = p("map_frame")
        self.base_frame = p("base_frame")
        self.auto_commit = bool(p("auto_commit"))

        self.extrinsics = CameraExtrinsics(
            x=float(p("camera_offset_x")), z=float(p("camera_height")))
        self.intrinsics = None      # built from the first frame's dimensions
        self._fov = float(p("camera_fov"))

        self.detector = load_backend(p("backend"),
                                     min_confidence=float(p("min_confidence")))
        self.get_logger().info(f"detector: {self.detector.name}")

        map_path = p("semantic_map")
        self.semantic_map = SemanticMap.load(map_path) if map_path else SemanticMap()
        self.get_logger().info(
            f"semantic map: {len(self.semantic_map)} hand-authored landmark(s)")

        self.tracker = make_tracker()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # Best-effort, keep-last-1. The detector cannot keep up with the camera
        # on a CPU, so a reliable queue would only grow, and every frame that
        # comes out of it is grounded against a staler pose than the last.
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.create_subscription(Image, "camera/image_raw", self._on_image,
                                 sensor_qos)
        self.create_subscription(Bool, "commit_landmarks", self._on_commit, 10)

        latched = QoSProfile(depth=1,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self._detections_pub = self.create_publisher(String, "detections", 10)
        self._status_pub = self.create_publisher(String, "perception_status", 10)
        self._markers_pub = self.create_publisher(MarkerArray, "landmark_proposals",
                                                  latched)

        self._latest = None
        self._lock = threading.Lock()
        self._busy = False
        self._dropped = 0
        self._processed = 0

        period = 1.0 / max(0.1, float(p("detect_hz")))
        self.create_timer(period, self._tick)
        self.create_timer(5.0, self._report)
        self.get_logger().info(
            f"running the detector at {p('detect_hz')} Hz; frames arriving "
            f"faster than that are dropped, not queued")

    # -- callbacks ----------------------------------------------------------

    def _on_image(self, message: Image) -> None:
        with self._lock:
            if self._latest is not None:
                self._dropped += 1
            self._latest = message

    def _on_commit(self, message: Bool) -> None:
        if not message.data:
            return
        outcome = propose(self.tracker.tracks, self.semantic_map)
        added = apply(outcome, self.semantic_map)
        self._say(f"committed {len(added)} landmark(s): {', '.join(added) or 'none'}")

    # -- the loop -----------------------------------------------------------

    def _tick(self) -> None:
        if self._busy:
            return
        with self._lock:
            message, self._latest = self._latest, None
        if message is None:
            return

        self._busy = True
        try:
            self._process(message)
        except Exception as error:              # noqa: BLE001
            # A bad frame must not take the node down. The node is what still
            # accumulates the rest of the patrol.
            self.get_logger().error(f"frame dropped: {type(error).__name__}: {error}")
        finally:
            self._busy = False

    def _process(self, message: Image) -> None:
        if self.intrinsics is None:
            self.intrinsics = CameraIntrinsics.from_fov(
                message.width, message.height, self._fov)
            self.get_logger().info(
                f"intrinsics from the first frame: {message.width}x"
                f"{message.height}, fx={self.intrinsics.fx:.1f}")

        pose = self._pose_at(message)
        if pose is None:
            return

        detections = self.detector.detect(image_to_array(message))
        detections = filter_detections(detections, 0.0,
                                       self.intrinsics.width,
                                       self.intrinsics.height)
        self._processed += 1
        if detections:
            self._detections_pub.publish(String(data=summarise(detections)))

        robot_x, robot_y, robot_yaw = pose
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9

        observations = []
        for detection in detections:
            if not detection.mappable:
                continue
            u, v = detection.ground_pixel
            hit = detection_to_ground(self.intrinsics, self.extrinsics, u, v)
            if hit is None:
                continue
            x, y = base_to_map(hit.as_tuple(), robot_x, robot_y, robot_yaw)
            observations.append(Observation(
                label=detection.label, confidence=detection.confidence,
                x=x, y=y, from_x=robot_x, from_y=robot_y,
                stamp=stamp, range_m=hit.range_m))

        if observations:
            self.tracker.update(observations)
            self._publish_markers()

        if self.auto_commit:
            self._on_commit(Bool(data=True))

    def _pose_at(self, message: Image):
        """Where the robot was WHEN THE IMAGE WAS TAKEN.

        Looking up "now" instead would place every landmark wherever the robot
        has driven to during inference. On a CPU that is a few hundred
        milliseconds, which at 0.45 m/s is over ten centimetres per frame - and
        it accumulates into a smear of landmarks along the route.
        """
        try:
            transform = self._tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, message.header.stamp)
        except Exception as error:              # noqa: BLE001
            self.get_logger().debug(f"no transform for this frame: {error}")
            return None

        t = transform.transform.translation
        q = transform.transform.rotation
        return (t.x, t.y, quaternion_to_yaw(q.x, q.y, q.z, q.w))

    # -- output -------------------------------------------------------------

    def _publish_markers(self) -> None:
        outcome = propose(self.tracker.tracks, self.semantic_map)
        array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        for index, proposal in enumerate(outcome.proposals):
            marker = Marker()
            marker.header.frame_id = self.map_frame
            marker.header.stamp = stamp
            marker.ns = "proposal"
            marker.id = index
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position = Point(x=float(proposal.x), y=float(proposal.y),
                                         z=0.05)
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = max(0.3, proposal.radius_m * 2)
            marker.scale.z = 0.1
            # Green shading by confidence, so a glance at RViz distinguishes a
            # well-triangulated landmark from a marginal one.
            marker.color.r = 1.0 - proposal.confidence
            marker.color.g = proposal.confidence
            marker.color.b = 0.2
            marker.color.a = 0.6
            array.markers.append(marker)

            label = Marker()
            label.header.frame_id = self.map_frame
            label.header.stamp = stamp
            label.ns = "proposal_label"
            label.id = index
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position = Point(x=float(proposal.x), y=float(proposal.y),
                                        z=0.45)
            label.pose.orientation.w = 1.0
            label.scale.z = 0.22
            label.color.r = label.color.g = label.color.b = 1.0
            label.color.a = 0.95
            label.text = f"{proposal.name} ({proposal.distinct_viewpoints}vp)"
            array.markers.append(label)

        self._markers_pub.publish(array)

    def _report(self) -> None:
        confirmed = self.tracker.summary()
        pending = len(self.tracker.pending())
        text = (f"{self._processed} frames processed, {self._dropped} dropped, "
                f"{len(self.tracker.tracks)} tracks, "
                f"{sum(confirmed.values())} confirmed {confirmed or ''}, "
                f"{pending} pending")
        self.get_logger().info(text)
        self._status_pub.publish(String(data=text))

    def _say(self, text: str) -> None:
        self.get_logger().info(text)
        self._status_pub.publish(String(data=text))


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
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
