"""Nav2 against the office map.

Separate from `bringup.launch.py` because Nav2 is optional. The semantic layer
runs with `use_nav2:=false` and the in-process simulated navigator, which is
what CI exercises and what makes the natural-language work reviewable without
a full navigation stack installed.

    ros2 launch robot_bringup navigation.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    share = get_package_share_directory("robot_bringup")

    map_yaml = LaunchConfiguration("map")
    params = LaunchConfiguration("params_file")
    use_sim_time = LaunchConfiguration("use_sim_time")
    autostart = LaunchConfiguration("autostart")

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("nav2_bringup"),
                         "launch", "bringup_launch.py")
        ),
        launch_arguments={
            "map": map_yaml,
            "use_sim_time": use_sim_time,
            "params_file": params,
            "autostart": autostart,
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "map", default_value=os.path.join(share, "maps", "office.yaml"),
            description="Occupancy map. Generated from the same geometry the "
                        "unit tests plan against — see scripts/generate_map.py",
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=os.path.join(share, "config", "nav2_params.yaml"),
        ),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("autostart", default_value="true"),
        nav2,
    ])
