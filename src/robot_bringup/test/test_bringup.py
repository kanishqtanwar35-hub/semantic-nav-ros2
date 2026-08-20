"""Tests for the things that are configuration rather than code.

Configuration is where robotics projects rot. The Python has a test suite; the
URDF, the launch files, the Nav2 parameters and the generated maps usually have
nothing, and they are edited more often than the code is. Every test here
covers a failure that would otherwise only appear as "the robot did not move"
after a two-minute Gazebo start-up.

The cross-file consistency checks are the valuable ones: they catch the class
of bug where every individual file is correct and two of them disagree.
"""

import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

PACKAGE = Path(__file__).resolve().parent.parent
URDF = PACKAGE / "urdf" / "robot.urdf.xacro"
LAUNCH = PACKAGE / "launch"
CONFIG = PACKAGE / "config"
MAPS = PACKAGE / "maps"
WORLDS = PACKAGE / "worlds"


# ---------------------------------------------------------------------------
# Robot description
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def urdf_xml():
    """Expand the xacro and parse it.

    xacro is a build-time dependency of this package. If it is missing the
    description cannot be checked at all, and a skip is more honest than a
    pass.
    """
    xacro = pytest.importorskip("xacro", reason="xacro is not installed")
    doc = xacro.process_file(str(URDF))
    return ET.fromstring(doc.toprettyxml(indent="  "))


def test_the_description_expands(urdf_xml):
    assert urdf_xml.tag == "robot"
    assert urdf_xml.get("name") == "semantic_bot"


def test_the_expected_links_exist(urdf_xml):
    links = {link.get("name") for link in urdf_xml.findall("link")}
    for required in ["base_footprint", "base_link", "left_wheel_link",
                     "right_wheel_link", "caster_link", "lidar_link",
                     "camera_link", "camera_optical_link"]:
        assert required in links, f"missing link: {required}"


def test_base_footprint_is_the_root(urdf_xml):
    """Nav2 and AMCL localise base_footprint, which sits on the ground under
    the robot's centre. Rooting the tree at base_link puts the origin at axle
    height and every distance the stack computes is wrong by the wheel radius.
    """
    children = {joint.find("child").get("link")
                for joint in urdf_xml.findall("joint")}
    roots = {link.get("name") for link in urdf_xml.findall("link")} - children
    assert roots == {"base_footprint"}


def test_the_tree_is_connected(urdf_xml):
    """Exactly one root. A second one means a link was added with no joint, and
    it silently never appears in TF."""
    parents = [joint.find("parent").get("link") for joint in urdf_xml.findall("joint")]
    children = [joint.find("child").get("link") for joint in urdf_xml.findall("joint")]
    assert len(children) == len(set(children)), "a link has two parents"
    assert set(parents) <= {link.get("name") for link in urdf_xml.findall("link")}


def test_every_physical_link_has_non_zero_inertia(urdf_xml):
    """Gazebo silently ignores links with zero or missing inertia. The robot
    looks right in RViz and falls through the floor in simulation, with no
    error message anywhere."""
    massless = []
    for link in urdf_xml.findall("link"):
        name = link.get("name")
        if link.find("visual") is None:      # pure frames carry no mass
            continue
        inertial = link.find("inertial")
        if inertial is None:
            massless.append(f"{name}: no <inertial>")
            continue
        mass = float(inertial.find("mass").get("value"))
        if mass <= 0:
            massless.append(f"{name}: mass {mass}")
            continue
        inertia = inertial.find("inertia")
        for axis in ["ixx", "iyy", "izz"]:
            if float(inertia.get(axis)) <= 0:
                massless.append(f"{name}: {axis} is not positive")
    assert not massless, "links Gazebo will ignore: " + "; ".join(massless)


def test_the_wheels_are_continuous_joints(urdf_xml):
    for name in ["left_wheel_joint", "right_wheel_joint"]:
        joint = urdf_xml.find(f".//joint[@name='{name}']")
        assert joint is not None, f"missing {name}"
        assert joint.get("type") == "continuous"


def test_the_wheel_separation_matches_the_diff_drive_plugin(urdf_xml):
    """The URDF and the plugin each carry their own copy of the wheel geometry.
    If they disagree the robot turns at the wrong rate, the odometry is wrong
    by a constant factor, and it presents as a badly tuned controller."""
    left = urdf_xml.find(".//joint[@name='left_wheel_joint']/origin")
    right = urdf_xml.find(".//joint[@name='right_wheel_joint']/origin")
    separation = abs(float(left.get("xyz").split()[1])
                     - float(right.get("xyz").split()[1]))

    plugin = urdf_xml.find(".//plugin[@name='diff_drive']")
    declared = float(plugin.find("wheel_separation").text)
    assert math.isclose(separation, declared, abs_tol=1e-6)


def test_the_wheel_diameter_matches_the_diff_drive_plugin(urdf_xml):
    radius = float(
        urdf_xml.find(".//link[@name='left_wheel_link']/visual/geometry/cylinder")
        .get("radius")
    )
    plugin = urdf_xml.find(".//plugin[@name='diff_drive']")
    assert math.isclose(2 * radius, float(plugin.find("wheel_diameter").text),
                        abs_tol=1e-6)


def test_the_base_sits_exactly_one_wheel_radius_above_the_ground(urdf_xml):
    radius = float(
        urdf_xml.find(".//link[@name='left_wheel_link']/visual/geometry/cylinder")
        .get("radius")
    )
    z = float(urdf_xml.find(".//joint[@name='base_joint']/origin")
              .get("xyz").split()[2])
    assert math.isclose(z, radius, abs_tol=1e-6)


def test_the_lidar_minimum_range_clears_the_chassis(urdf_xml):
    """A min_range inside the robot's own footprint fills the scan with returns
    off its own body, which the costmap renders as an obstacle ring that
    follows the robot everywhere."""
    box = urdf_xml.find(".//link[@name='base_link']/collision/geometry/box")
    length, width, _ = [float(v) for v in box.get("size").split()]
    circumscribed = math.hypot(length, width) / 2

    min_range = float(urdf_xml.find(".//sensor[@name='lidar']/ray/range/min").text)
    assert min_range >= circumscribed * 0.65, (
        f"lidar min range {min_range} m is inside the chassis "
        f"(circumscribed radius {circumscribed:.3f} m)"
    )


def test_the_camera_publishes_in_the_optical_frame(urdf_xml):
    """REP-103: ROS is x-forward, vision is z-forward. Publishing images in
    camera_link rotates every projection by 90 degrees, and the symptom shows
    up a long way from the cause."""
    plugin = urdf_xml.find(".//plugin[@name='camera_controller']")
    assert plugin.find("frame_name").text == "camera_optical_link"


def test_the_base_does_not_subscribe_to_cmd_vel_directly(urdf_xml):
    """The single most important line in the Gazebo description.

    The simulated base subscribes to /cmd_vel_raw. The safety governor is the
    only publisher on /cmd_vel. Remove this remapping and Nav2 reaches the
    wheels directly, bypassing the safety layer entirely - and nothing else in
    the system would notice.
    """
    remaps = [r.text for r in urdf_xml.findall(".//plugin[@name='diff_drive']/ros/remapping")]
    assert "cmd_vel:=cmd_vel_raw" in remaps


# ---------------------------------------------------------------------------
# Launch files
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "bringup.launch.py", "simulation.launch.py", "navigation.launch.py",
])
def test_launch_files_are_syntactically_valid(name):
    """A launch file with a syntax error fails two minutes into a demo. This
    costs milliseconds."""
    path = LAUNCH / name
    assert path.exists(), f"missing {name}"
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


@pytest.mark.parametrize("name", [
    "bringup.launch.py", "simulation.launch.py", "navigation.launch.py",
])
def test_launch_files_define_the_entry_point(name):
    source = (LAUNCH / name).read_text(encoding="utf-8")
    assert "def generate_launch_description(" in source


def test_bringup_launches_the_safety_governor():
    """If the governor is not launched, nothing publishes /cmd_vel and the
    robot simply does not move - a silent failure that looks like a Nav2
    problem."""
    source = (LAUNCH / "bringup.launch.py").read_text(encoding="utf-8")
    assert "safety_node" in source
    assert "semantic_nav_node" in source


def test_the_spawn_pose_matches_the_amcl_initial_pose():
    """A mismatch means the robot believes it is somewhere it is not. Every
    goal then fails for reasons that look like planner bugs."""
    source = (LAUNCH / "simulation.launch.py").read_text(encoding="utf-8")
    assert '"-x", "3.0", "-y", "3.0"' in source

    params = yaml.safe_load((CONFIG / "nav2_params.yaml").read_text(encoding="utf-8"))
    initial = params["amcl"]["ros__parameters"]["initial_pose"]
    assert (initial["x"], initial["y"]) == (3.0, 3.0)


# ---------------------------------------------------------------------------
# Cross-file consistency - the bugs no single file contains
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nav2_params():
    return yaml.safe_load((CONFIG / "nav2_params.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def safety_params():
    loaded = yaml.safe_load((CONFIG / "safety.yaml").read_text(encoding="utf-8"))
    return loaded["safety_governor"]["ros__parameters"]


def test_nav2_never_plans_faster_than_the_governor_permits(nav2_params, safety_params):
    """The consistency check that matters.

    If DWB plans trajectories at 0.8 m/s and the governor clamps to 0.45, every
    trajectory the controller evaluates is clamped on the way out. The robot
    follows none of them, oscillates, and the controller blames the path. Both
    files are individually correct; together they are a bug.
    """
    dwb = nav2_params["controller_server"]["ros__parameters"]["FollowPath"]
    assert dwb["max_vel_x"] <= safety_params["max_linear"] + 1e-9
    assert dwb["max_vel_theta"] <= safety_params["max_angular"] + 1e-9
    assert dwb["acc_lim_x"] <= safety_params["max_linear_accel"] + 1e-9
    assert dwb["acc_lim_theta"] <= safety_params["max_angular_accel"] + 1e-9


def test_the_velocity_smoother_agrees_with_the_governor(nav2_params, safety_params):
    smoother = nav2_params["velocity_smoother"]["ros__parameters"]
    assert smoother["max_velocity"][0] <= safety_params["max_linear"] + 1e-9
    assert smoother["max_velocity"][2] <= safety_params["max_angular"] + 1e-9


def test_the_robot_radius_matches_the_urdf(nav2_params, urdf_xml):
    """A costmap radius smaller than the robot plans paths it cannot fit
    through. Larger, and it refuses doorways it fits through easily."""
    box = urdf_xml.find(".//link[@name='base_link']/collision/geometry/box")
    length, width, _ = [float(v) for v in box.get("size").split()]
    circumscribed = math.hypot(length, width) / 2

    for costmap in ["local_costmap", "global_costmap"]:
        radius = nav2_params[costmap][costmap]["ros__parameters"]["robot_radius"]
        assert radius >= circumscribed * 0.9, f"{costmap} radius {radius} is too small"
        assert radius <= circumscribed * 1.3, f"{costmap} radius {radius} is too large"


def test_the_stop_distance_exceeds_the_braking_distance(safety_params):
    """v^2 / 2a. A stop distance shorter than the braking distance means the
    robot decides to stop and then keeps travelling into the obstacle."""
    v = safety_params["max_linear"]
    a = safety_params["max_linear_accel"]
    braking = v * v / (2 * a)
    assert safety_params["stop_distance"] > braking, (
        f"stop_distance {safety_params['stop_distance']} m is inside the "
        f"{braking:.3f} m braking distance at {v} m/s"
    )


def test_slow_distance_is_outside_stop_distance(safety_params):
    assert safety_params["slow_distance"] > safety_params["stop_distance"]


def test_the_inflation_radius_leaves_the_doorways_open(nav2_params):
    """The doorways are 1.2 m. Inflating by more than half of that from each
    side closes them, the planner reports 'no valid path', and the map looks
    perfectly fine to a human."""
    for costmap in ["local_costmap", "global_costmap"]:
        params = nav2_params[costmap][costmap]["ros__parameters"]
        assert params["inflation_layer"]["inflation_radius"] < 0.6


def test_the_lidar_range_limits_agree_between_the_urdf_and_amcl(nav2_params, urdf_xml):
    ray = urdf_xml.find(".//sensor[@name='lidar']/ray/range")
    amcl = nav2_params["amcl"]["ros__parameters"]
    assert amcl["laser_max_range"] <= float(ray.find("max").text) + 1e-9
    assert amcl["laser_min_range"] >= float(ray.find("min").text) - 1e-9


# ---------------------------------------------------------------------------
# Generated artefacts
# ---------------------------------------------------------------------------

def test_the_generated_files_exist():
    for path in [MAPS / "office.pgm", MAPS / "office.yaml",
                 CONFIG / "semantic_map.yaml", WORLDS / "office.world"]:
        assert path.exists(), f"missing {path.name}; run scripts/generate_map.py"


def test_the_map_yaml_points_at_the_pgm():
    meta = yaml.safe_load((MAPS / "office.yaml").read_text(encoding="utf-8"))
    assert (MAPS / meta["image"]).exists()
    assert meta["resolution"] == 0.1
    assert meta["origin"][:2] == [0.0, 0.0]


def test_the_pgm_header_matches_the_declared_resolution():
    raw = (MAPS / "office.pgm").read_bytes()
    assert raw.startswith(b"P5")
    fields = raw.split(b"\n")
    dimensions = [f for f in fields[:4] if f and not f.startswith(b"#")][1]
    width, height = [int(v) for v in dimensions.split()]

    meta = yaml.safe_load((MAPS / "office.yaml").read_text(encoding="utf-8"))
    assert width * meta["resolution"] == pytest.approx(12.0)
    assert height * meta["resolution"] == pytest.approx(9.0)


def test_the_world_file_is_valid_xml():
    tree = ET.parse(WORLDS / "office.world")
    assert tree.getroot().tag == "sdf"
    names = {m.get("name") for m in tree.getroot().findall(".//model")}
    assert "partition" in names
    assert "reception_desk" in names


def test_the_semantic_map_loads_and_has_a_dock():
    """`return_home` has nowhere to go without one, and the failure appears at
    runtime rather than at launch."""
    data = yaml.safe_load((CONFIG / "semantic_map.yaml").read_text(encoding="utf-8"))
    names = {entry["name"] for entry in data["landmarks"]}
    assert "charging dock" in names
    assert len(names) >= 5


def test_the_generated_files_are_reproducible(tmp_path):
    """Regenerating must not change anything. If it does, someone edited a
    generated file by hand and their change is one `generate_map.py` away from
    being silently reverted."""
    script = PACKAGE.parent.parent / "scripts" / "generate_map.py"
    if not script.exists():
        pytest.skip("running from an installed share directory")

    generated = [MAPS / "office.pgm", MAPS / "office.yaml",
                 CONFIG / "semantic_map.yaml", WORLDS / "office.world"]
    before = {path: path.read_bytes() for path in generated}

    result = subprocess.run([sys.executable, str(script)],
                            capture_output=True, text=True,
                            cwd=str(script.parent.parent), env={**os.environ})
    assert result.returncode == 0, result.stderr

    for path, content in before.items():
        assert path.read_bytes() == content, (
            f"{path.name} changed when regenerated - it was edited by hand, and "
            f"that edit is one run of generate_map.py away from being reverted"
        )
