# Development and CI image for this workspace.
#
# Based on ros:humble-ros-base rather than ros:humble-desktop on purpose. The
# desktop image is ~3.5 GB and carries RViz, Gazebo and the full Nav2 stack;
# this one is ~1.2 GB and carries none of them. If the workspace builds and
# tests here, it builds and tests without a simulator or a navigation stack
# installed - which is the property that keeps the algorithms testable and the
# CI job under a minute.
FROM ros:humble-ros-base

# xacro           - expanding the robot description
# ament-cmake-pytest - running robot_bringup's tests from CMake
# ament lint      - flake8 and copyright checks wired into colcon test
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-xacro \
      ros-humble-ament-cmake-pytest \
      ros-humble-ament-flake8 \
      ros-humble-ament-copyright \
      ros-humble-ament-pep257 \
      python3-pytest \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws
