import pytest

from robot_core.geometry import Pose2D
from robot_core.occupancy import OCCUPIED
from semantic_nav.commands import Command, Step, Verb, validate
from semantic_nav.demo_world import build, build_grid
from semantic_nav.grounding import Grounder
from semantic_nav.nav_client import (
    MissionRunner,
    Nav2Navigator,
    NavigationFailed,
    SimulatedNavigator,
    StepOutcome,
)


@pytest.fixture
def world():
    return build()


@pytest.fixture
def navigator(world):
    grid, _ = world
    return SimulatedNavigator(grid, start=Pose2D(3.0, 3.0, 0.0))


@pytest.fixture
def runner(navigator):
    # Sleep is injected so a 300-second wait costs nothing in CI. A test suite
    # that actually waits is a test suite people stop running.
    return MissionRunner(navigator, sleep=lambda _s: None)


def plan(semantic_map, utterance):
    return validate(Grounder(semantic_map, llm=None).ground(utterance), semantic_map)


# -- the demo world itself --------------------------------------------------

def test_every_landmark_approach_pose_is_reachable(world):
    """The map is data, and data has bugs. An approach pose inside a wall
    produces a robot that fails to reach a place that is right in front of it,
    and the failure looks like a planner bug.
    """
    grid, semantic_map = world
    inflated = grid.inflate(0.22)

    unreachable = []
    for landmark in semantic_map:
        pose = landmark.approach_pose
        if not inflated.is_free(inflated.world_to_cell(pose.x, pose.y)):
            unreachable.append(f"{landmark.name} at ({pose.x:.2f}, {pose.y:.2f})")

    assert not unreachable, "approach poses inside obstacles: " + ", ".join(unreachable)


def test_every_landmark_is_reachable_from_the_lobby(world):
    grid, semantic_map = world
    for landmark in semantic_map:
        navigator = SimulatedNavigator(grid, start=Pose2D(3.0, 3.0, 0.0))
        pose = landmark.approach_pose
        outcome = navigator.go_to(pose.x, pose.y, pose.yaw)
        assert outcome.ok, f"{landmark.name}: {outcome.detail}\n{grid.ascii()}"


def test_the_doorways_survive_inflation(world):
    """1.2 m doorways minus a 0.22 m robot inflated on both sides leaves 0.76 m.
    If someone narrows a doorway in the world file, this fails before the
    navigation tests do and says why."""
    grid, _ = world
    inflated = grid.inflate(0.22)

    for name, x in [("meeting room door", 3.0), ("kitchen door", 9.0)]:
        cell = inflated.world_to_cell(x, 5.0)
        assert inflated.is_free(cell), f"{name} is closed after inflation"


# -- simulated navigation ---------------------------------------------------

def test_navigating_to_a_landmark(navigator, world):
    _, semantic_map = world
    pose = semantic_map.get("kitchen").approach_pose
    outcome = navigator.go_to(pose.x, pose.y, pose.yaw)

    assert outcome.ok
    assert outcome.distance_m > 0
    assert navigator.pose.distance_to(pose) < 1e-9


def test_the_route_to_the_kitchen_goes_through_the_doorway(navigator, world):
    """A straight line from the lobby to the kitchen crosses the dividing wall.
    The trail must not."""
    grid, semantic_map = world
    pose = semantic_map.get("kitchen").approach_pose
    navigator.go_to(pose.x, pose.y)

    for waypoint in navigator.trail:
        cell = grid.world_to_cell(waypoint.x, waypoint.y)
        assert grid.is_free(cell), f"trail passes through a wall at {waypoint}"


def test_a_blocked_goal_fails_with_the_planners_own_reason(world):
    """"Navigation failed" tells an operator nothing. "goal is in an obstacle"
    tells them the map is wrong; "no path exists" tells them a door is shut."""
    grid, _ = world
    navigator = SimulatedNavigator(grid, start=Pose2D(3.0, 3.0))
    outcome = navigator.go_to(4.5, 0.95)          # inside the reception desk
    assert not outcome.ok
    assert "obstacle" in outcome.detail


def test_a_sealed_room_reports_no_path():
    grid = build_grid()
    for x in [3.0, 9.0]:                          # brick up both doorways
        grid.mark_rectangle(x - 1.0, 4.9, x + 1.0, 5.1, OCCUPIED)

    navigator = SimulatedNavigator(grid, start=Pose2D(3.0, 3.0))
    outcome = navigator.go_to(9.0, 6.5)
    assert not outcome.ok
    assert outcome.reason if hasattr(outcome, "reason") else True
    assert "no path" in outcome.detail


def test_the_planning_grid_is_inflated_once_not_per_goal(world):
    grid, _ = world
    navigator = SimulatedNavigator(grid, robot_radius_m=0.22)
    first = navigator.planning_grid
    navigator.go_to(3.0, 3.0)
    assert navigator.planning_grid is first


# -- the mission runner -----------------------------------------------------

def test_a_single_step_mission(runner, world):
    _, semantic_map = world
    outcome = runner.run(plan(semantic_map, "go to the kitchen"))
    assert outcome.ok
    assert len(outcome.outcomes) == 1


def test_a_multi_step_mission_runs_in_order(runner, navigator, world):
    _, semantic_map = world
    outcome = runner.run(
        plan(semantic_map, "go to the printer then wait 5 seconds then go home")
    )
    assert outcome.ok
    assert [o.step.verb for o in outcome.outcomes] == [
        Verb.GO_TO, Verb.WAIT, Verb.RETURN_HOME
    ]
    dock = semantic_map.get("charging dock").approach_pose
    assert navigator.pose.distance_to(dock) < 1e-9


def test_a_failed_step_aborts_the_rest(world):
    """Later steps were planned assuming the robot got where it was going.
    Running them from the wrong place is how one failed navigation becomes a
    robot somewhere nobody expects."""
    grid, semantic_map = world
    navigator = SimulatedNavigator(grid, start=Pose2D(3.0, 3.0))
    runner = MissionRunner(navigator, sleep=lambda _s: None)

    def fail_second(x, y, yaw=0.0):
        fail_second.calls += 1
        if fail_second.calls == 2:
            return StepOutcome(Step(Verb.GO_TO), ok=False, detail="simulated failure")
        return StepOutcome(Step(Verb.GO_TO), ok=True, detail="ok")
    fail_second.calls = 0
    navigator.go_to = fail_second

    command = validate(Command(steps=[
        Step(Verb.GO_TO, "kitchen"),
        Step(Verb.GO_TO, "lobby"),
        Step(Verb.GO_TO, "printer"),
    ]), semantic_map)

    outcome = runner.run(command)
    assert not outcome.ok
    assert outcome.aborted_at == 1
    assert len(outcome.outcomes) == 2          # the third never ran
    assert fail_second.calls == 2


def test_stop_ends_the_mission_immediately(runner, world):
    _, semantic_map = world
    command = validate(Command(steps=[
        Step(Verb.STOP),
        Step(Verb.GO_TO, "kitchen"),
    ]), semantic_map)

    outcome = runner.run(command)
    assert outcome.ok
    assert len(outcome.outcomes) == 1
    assert "stop" in outcome.reason


def test_stopping_mid_mission_prevents_the_remaining_steps(navigator, world):
    _, semantic_map = world
    runner = MissionRunner(navigator, sleep=lambda _s: None)

    original = navigator.go_to

    def stop_after_first(x, y, yaw=0.0):
        result = original(x, y, yaw)
        runner.stop()
        return result
    navigator.go_to = stop_after_first

    command = validate(Command(steps=[
        Step(Verb.GO_TO, "kitchen"),
        Step(Verb.GO_TO, "lobby"),
    ]), semantic_map)

    outcome = runner.run(command)
    assert not outcome.ok
    assert outcome.aborted_at == 1
    assert "operator" in outcome.reason


def test_stop_cancels_the_navigator_too(navigator, runner):
    runner.stop()
    assert navigator.cancelled


def test_a_wait_does_not_actually_block(world):
    _, semantic_map = world
    slept = []
    grid, _ = world
    runner = MissionRunner(SimulatedNavigator(grid), sleep=slept.append)

    runner.run(validate(Command(steps=[Step(Verb.WAIT, seconds=42)]), semantic_map))
    assert slept == [42.0]


def test_report_publishes_the_current_pose(navigator, world):
    _, semantic_map = world
    said = []
    runner = MissionRunner(navigator, sleep=lambda _s: None, on_report=said.append)

    runner.run(validate(Command(steps=[Step(Verb.REPORT)]), semantic_map))
    assert said and "3.00" in said[0]


def test_the_summary_is_readable(runner, world):
    _, semantic_map = world
    outcome = runner.run(plan(semantic_map, "go to the kitchen then go home"))
    summary = outcome.summary()
    assert "OK" in summary
    assert "go_to" in summary
    assert "m" in summary


def test_distance_accumulates_across_steps(runner, world):
    _, semantic_map = world
    outcome = runner.run(plan(semantic_map, "go to the kitchen then go home"))
    assert outcome.distance_m == pytest.approx(
        sum(o.distance_m for o in outcome.outcomes)
    )
    assert outcome.distance_m > 5.0


# -- the Nav2 adapter -------------------------------------------------------

def test_the_nav2_adapter_imports_without_ros():
    """The lazy import is what lets this whole test file exist. If Nav2Navigator
    imported rclpy at module scope, importing nav_client would fail on any
    machine without ROS and none of the mission logic above could be tested."""
    assert Nav2Navigator is not None
    Nav2Navigator()          # constructing it touches no ROS


def test_the_nav2_adapter_fails_with_a_useful_message_when_ros_is_absent():
    """On a machine without ROS the failure must name the alternative. "No
    module named rclpy" from deep inside a navigation call is a much worse
    experience than being told to use SimulatedNavigator."""
    try:
        import rclpy            # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("ROS is installed here; the fallback path cannot be exercised")

    with pytest.raises(NavigationFailed) as error:
        Nav2Navigator()._ensure_client()
    assert "SimulatedNavigator" in str(error.value)


# -- end to end -------------------------------------------------------------

@pytest.mark.parametrize("utterance", [
    "go to the kitchen",
    "take me to the meeting room",
    "patrol the lobby and the kitchen",
    "go to the printer then wait 5 seconds then go home",
    "drive to the reception desk",
    "head over to the coffee",          # alias
    "go back to the dock",
])
def test_english_in_motion_out(utterance, world):
    """The whole pipeline: text -> rules -> validation -> A* -> executed."""
    grid, semantic_map = world
    navigator = SimulatedNavigator(grid, start=Pose2D(3.0, 3.0, 0.0))
    runner = MissionRunner(navigator, sleep=lambda _s: None)

    outcome = runner.run(plan(semantic_map, utterance))

    assert outcome.ok, outcome.summary()
    assert all(o.ok for o in outcome.outcomes)


def test_the_full_pipeline_never_leaves_the_building(world):
    grid, semantic_map = world
    navigator = SimulatedNavigator(grid, start=Pose2D(3.0, 3.0, 0.0))
    runner = MissionRunner(navigator, sleep=lambda _s: None)

    runner.run(plan(semantic_map, "patrol the lobby, the kitchen and the printer"))

    for waypoint in navigator.trail:
        assert 0 <= waypoint.x <= 12.0
        assert 0 <= waypoint.y <= 9.0
