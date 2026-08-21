"""robot_perception - vision to semantic landmarks, without ROS or a GPU.

The detector is a swappable backend; everything else here is plain Python and
runs under pytest in milliseconds. Same contract as `robot_core`, and
`test_no_ros_imports.py` in that package enforces the equivalent rule here.
"""

__version__ = "1.0.0"
