"""Bring up the robot description, the safety governor and semantic navigation.

No simulator and no Nav2 — this is the layer that is identical on real hardware
and in simulation, which is why it is a separate launch file. `simulation.launch.py`
includes it, and so would a launch file for a physical base.

Topology this file establishes:

    /nl_command ──► semantic_nav ──► Nav2 ──► /cmd_vel_raw
                                                   │
                                    /scan ──► safety_governor
                                                   │
                                                   ▼
                                              /cmd_vel ──► base

The safety governor is the only publisher on /cmd_vel. Every other producer of
velocity — Nav2, teleop, anything added later — publishes to /cmd_vel_raw.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package = "robot_bringup"
    share = get_package_share_directory(package)

    use_sim_time = LaunchConfiguration("use_sim_time")
    semantic_map = LaunchConfiguration("semantic_map")
    use_llm = LaunchConfiguration("use_llm")
    use_nav2 = LaunchConfiguration("use_nav2")
    rviz = LaunchConfiguration("rviz")

    urdf = PathJoinSubstitution(
        [FindPackageShare(package), "urdf", "robot.urdf.xacro"]
    )

    # ParameterValue(..., value_type=str) is required. Without it the launch
    # system infers the type of the xacro output, decides a URDF that starts
    # with "<?xml" is not a string, and robot_state_publisher receives
    # something it cannot parse — with an error that does not mention types.
    robot_description = ParameterValue(
        Command(["xacro ", urdf]), value_type=str
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument(
            "semantic_map",
            default_value=os.path.join(share, "config", "semantic_map.yaml"),
            description="YAML landmark map the natural-language layer resolves against",
        ),
        DeclareLaunchArgument(
            "use_llm", default_value="true",
            description="Allow the Gemini fallback. The rule parser works regardless; "
                        "with no GEMINI_API_KEY set this degrades silently and says so.",
        ),
        DeclareLaunchArgument(
            "use_nav2", default_value="true",
            description="false uses the in-process simulated navigator instead",
        ),
        DeclareLaunchArgument("rviz", default_value="false"),
        DeclareLaunchArgument(
            "perception", default_value="true",
            description="Run the vision layer. Off, the robot navigates the "
                        "hand-authored map exactly as before.",
        ),
        DeclareLaunchArgument(
            "detector", default_value="auto",
            description="auto | yolo | synthetic | scripted. 'auto' prefers a "
                        "real model and falls back to synthetic, saying so.",
        ),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{
                "robot_description": robot_description,
                "use_sim_time": use_sim_time,
            }],
        ),

        Node(
            package="semantic_nav",
            executable="safety_node",
            name="safety_governor",
            output="screen",
            parameters=[
                PathJoinSubstitution([FindPackageShare(package), "config",
                                      "safety.yaml"]),
                {"use_sim_time": use_sim_time},
            ],
        ),

        Node(
            package="semantic_nav",
            executable="semantic_nav_node",
            name="semantic_nav",
            output="screen",
            parameters=[{
                "semantic_map": semantic_map,
                "use_llm": use_llm,
                "use_nav2": use_nav2,
                "use_sim_time": use_sim_time,
            }],
        ),

        Node(
            package="semantic_nav",
            executable="landmark_markers_node",
            name="landmark_markers",
            output="screen",
            parameters=[{
                "semantic_map": semantic_map,
                "use_sim_time": use_sim_time,
            }],
        ),

        Node(
            package="robot_perception",
            executable="perception_node",
            name="robot_perception",
            output="screen",
            condition=IfCondition(LaunchConfiguration("perception")),
            parameters=[{
                "backend": LaunchConfiguration("detector"),
                "semantic_map": semantic_map,
                # 2 Hz, not the camera's 15. On a CPU the detector takes
                # hundreds of milliseconds, so frames must be dropped rather
                # than queued - a queued frame is grounded against a stale
                # pose and smears landmarks along the robot's path.
                "detect_hz": 2.0,
                # Perception PROPOSES. Committing to the map the navigation
                # stack trusts is a separate, deliberate act:
                #   ros2 topic pub --once /commit_landmarks std_msgs/Bool "data: true"
                "auto_commit": False,
                "use_sim_time": use_sim_time,
            }],
        ),

        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            condition=IfCondition(rviz),
            arguments=["-d", os.path.join(share, "config", "robot.rviz")],
            parameters=[{"use_sim_time": use_sim_time}],
        ),
    ])
