"""The architecture test, again — the library half must stay ROS-free.

`robot_core` has the same check. It is repeated here rather than shared because
the rule has a different shape: this package DOES contain a ROS node, so the
check is per-file rather than package-wide, and the list of files that are
allowed to import ROS is the thing being pinned.

An assertion is cheaper than the discipline. The property erodes one convenient
import at a time — somebody needs a Header, adds `import rclpy` to `camera.py`,
and six months later 124 tests need a ROS graph to run.
"""

import ast
import pathlib

import pytest

BANNED_ROOTS = {
    "rclpy", "rclcpp", "nav_msgs", "geometry_msgs", "sensor_msgs", "std_msgs",
    "nav2_msgs", "tf2_ros", "tf2_geometry_msgs", "visualization_msgs",
    "cv_bridge", "action_msgs", "builtin_interfaces", "ament_index_python",
}

#: The only file permitted to import ROS. Adding to this list should be a
#: deliberate act, which is why it is a literal and not a glob.
ROS_ALLOWED = {"perception_node.py"}

#: Heavy optional dependencies. Importing these at module scope anywhere else
#: would make the whole package require an ML stack.
HEAVY = {"ultralytics", "torch", "torchvision", "cv2", "transformers"}
HEAVY_ALLOWED = {"yolo.py"}

PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "robot_perception"
SOURCES = sorted(PACKAGE.rglob("*.py"))


def _module_scope_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = []
    for node in tree.body:          # top level only, not ast.walk
        if isinstance(node, ast.Import):
            names += [(a.name.split(".")[0], node.lineno) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append((node.module.split(".")[0], node.lineno))
    return names


def test_there_are_sources_to_check():
    assert len(SOURCES) >= 6


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_only_the_node_imports_ros(path):
    offenders = [(name, line) for name, line in _module_scope_imports(path)
                 if name in BANNED_ROOTS]
    if path.name in ROS_ALLOWED:
        return
    assert not offenders, (
        f"{path.name} imports ROS at module scope: {offenders}. Only "
        f"{sorted(ROS_ALLOWED)} may. Put the ROS types in the node and pass "
        f"plain values down."
    )


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_only_the_yolo_backend_imports_an_ml_stack(path):
    offenders = [(name, line) for name, line in _module_scope_imports(path)
                 if name in HEAVY]
    if path.name in HEAVY_ALLOWED:
        return
    assert not offenders, (
        f"{path.name} imports {offenders} at module scope. The detector is "
        f"optional by design: importing it here would make `pip install "
        f"pyyaml pytest` insufficient to run the suite."
    )


def test_even_the_yolo_backend_defers_its_import():
    """`load_backend` catches ImportError, so `yolo.py` must not fail at IMPORT
    time on a machine without ultralytics — only at construction."""
    path = PACKAGE / "backends" / "yolo.py"
    offenders = [n for n, _ in _module_scope_imports(path) if n in HEAVY]
    assert offenders == [], (
        f"yolo.py imports {offenders} at module scope; it must import inside "
        f"the constructor so `load_backend('auto')` can fall back cleanly"
    )


def test_the_check_would_catch_a_violation(tmp_path):
    """A guard that never fails is indistinguishable from one that cannot."""
    offending = tmp_path / "bad.py"
    offending.write_text("import rclpy\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="imports ROS"):
        test_only_the_node_imports_ros(offending)


def test_the_package_imports_with_only_the_standard_library_and_yaml():
    import importlib
    import sys
    for name in ["camera", "detection", "tracking", "mapping", "patrol",
                 "backends.synthetic"]:
        module = f"robot_perception.{name}"
        sys.modules.pop(module, None)
        importlib.import_module(module)
