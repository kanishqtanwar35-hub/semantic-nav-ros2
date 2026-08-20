"""The architecture test.

`robot_core` must remain importable without a ROS installation. That is the
whole reason the algorithms in this package are testable at all, and it is
exactly the kind of property that erodes one convenient import at a time:
someone needs a `Duration` or a `Header`, adds `import rclpy`, and six months
later nothing in the package can be exercised without a running ROS graph.

An assertion is cheaper than the discipline.
"""

import ast
import pathlib

import pytest

BANNED_ROOTS = {
    "rclpy", "rclcpp", "nav_msgs", "geometry_msgs", "sensor_msgs", "std_msgs",
    "nav2_msgs", "tf2_ros", "tf2_geometry_msgs", "action_msgs", "builtin_interfaces",
    "ament_index_python", "launch", "launch_ros", "rosidl_runtime_py",
}

PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "robot_core"
SOURCES = sorted(PACKAGE.glob("*.py"))


def test_the_package_has_sources_to_check():
    # Guards against the suite silently passing because the glob broke.
    assert len(SOURCES) >= 5


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_module_imports_no_ros(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in BANNED_ROOTS:
                    offenders.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in BANNED_ROOTS:
                offenders.append((node.lineno, node.module))

    assert not offenders, (
        f"{path.name} imports ROS: "
        + ", ".join(f"line {line}: {name}" for line, name in offenders)
        + "\nrobot_core is the ROS-free half of the stack. Put the ROS types in "
          "a node under semantic_nav and pass plain values down."
    )


def test_the_check_would_actually_catch_a_violation(tmp_path):
    """A guard test that never fails is indistinguishable from one that cannot.

    This proves the AST walk detects what it claims to.
    """
    offending = tmp_path / "bad.py"
    offending.write_text("from nav_msgs.msg import OccupancyGrid\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="imports ROS"):
        test_module_imports_no_ros(offending)


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_third_party_imports_are_deferred(path):
    """Only the standard library may be imported at module scope.

    PyYAML is imported *inside* `SemanticMap.load`/`save` rather than at the top
    of the file. That is deliberate: the algorithms — planning, safety, geometry
    — then have zero third-party dependencies, so they import on a bare
    interpreter and cannot be broken by a packaging problem in a dependency
    that only the persistence layer needs.
    """
    stdlib_ok = {
        "__future__", "ast", "dataclasses", "difflib", "heapq", "importlib",
        "math", "pathlib", "random", "re", "sys", "typing", "robot_core",
    }
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    module_level = []
    for node in tree.body:                       # top level only, not ast.walk
        if isinstance(node, ast.Import):
            module_level += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            module_level.append(node.module.split(".")[0])

    offenders = [name for name in module_level if name not in stdlib_ok]
    assert not offenders, (
        f"{path.name} imports {offenders} at module scope; defer it into the "
        "function that needs it."
    )
