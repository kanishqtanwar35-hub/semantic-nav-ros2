"""Gazebo + the robot + the whole stack.

    ros2 launch robot_bringup simulation.launch.py rviz:=true

Then, in another terminal:

    ros2 topic pub --once /nl_command std_msgs/String "data: 'go to the kitchen'"
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package = "robot_bringup"
    share = get_package_share_directory(package)

    world = LaunchConfiguration("world")
    rviz = LaunchConfiguration("rviz")
    use_llm = LaunchConfiguration("use_llm")

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("gazebo_ros"),
                         "launch", "gazebo.launch.py")
        ),
        launch_arguments={"world": world, "verbose": "false"}.items(),
    )

    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(share, "launch", "bringup.launch.py")
        ),
        launch_arguments={
            "use_sim_time": "true",
            "rviz": rviz,
            "use_llm": use_llm,
        }.items(),
    )

    spawn = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        name="spawn_semantic_bot",
        output="screen",
        arguments=[
            "-topic", "robot_description",
            "-entity", "semantic_bot",
            # Spawned in the lobby, matching the initial pose the Nav2 launch
            # file seeds AMCL with. A mismatch here means the robot believes it
            # is somewhere it is not, and every goal fails for reasons that
            # look like planner bugs.
            "-x", "3.0", "-y", "3.0", "-z", "0.08",
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "world", default_value=os.path.join(share, "worlds", "office.world")
        ),
        DeclareLaunchArgument("rviz", default_value="false"),
        DeclareLaunchArgument("use_llm", default_value="true"),

        gazebo,
        bringup,
        spawn,

        # Announced once Gazebo has actually accepted the model, rather than
        # printed optimistically at launch time.
        RegisterEventHandler(
            OnProcessExit(
                target_action=spawn,
                on_exit=[ExecuteProcess(
                    cmd=["echo",
                         "robot spawned. try: ros2 topic pub --once /nl_command "
                         "std_msgs/String \"data: 'go to the kitchen'\""],
                    output="screen",
                )],
            )
        ),
    ])
