import math

import pytest

from robot_core.occupancy import OCCUPIED, Cell, OccupancyGrid
from robot_core.planners import (
    astar,
    line_of_sight,
    octile,
    rrt,
    simplify,
    step_cost,
)


@pytest.fixture
def empty():
    return OccupancyGrid(20, 20, 0.5)


@pytest.fixture
def wall():
    """A vertical wall with a single gap, so there is exactly one way through."""
    grid = OccupancyGrid(20, 20, 0.5)
    for row in range(20):
        if row != 10:                 # the gap
            grid.set(Cell(row, 10), OCCUPIED)
    return grid


# -- heuristic --------------------------------------------------------------

def test_octile_is_zero_at_the_goal():
    assert octile(Cell(3, 3), Cell(3, 3)) == 0.0


def test_octile_never_over_estimates_the_true_cost(empty):
    """Admissibility is what makes A* optimal. If the heuristic ever exceeds the
    real cost, A* still returns *a* path — just not the shortest one, silently.
    """
    goal = Cell(15, 17)
    for row in range(0, 20, 3):
        for col in range(0, 20, 3):
            start = Cell(row, col)
            result = astar(empty, start, goal)
            assert result.success
            assert octile(start, goal) <= result.cost + 1e-9


def test_step_cost_charges_more_for_a_diagonal():
    assert step_cost(Cell(0, 0), Cell(0, 1)) == 1.0
    assert math.isclose(step_cost(Cell(0, 0), Cell(1, 1)), math.sqrt(2))


# -- A* ---------------------------------------------------------------------

def test_finds_a_straight_path_in_an_empty_grid(empty):
    result = astar(empty, Cell(2, 2), Cell(2, 8))
    assert result.success
    assert result.path[0] == Cell(2, 2)
    assert result.path[-1] == Cell(2, 8)
    assert math.isclose(result.cost, 6.0)


def test_diagonal_path_costs_sqrt_two_per_step(empty):
    result = astar(empty, Cell(0, 0), Cell(5, 5))
    assert math.isclose(result.cost, 5 * math.sqrt(2), rel_tol=1e-9)


def test_four_connected_is_more_expensive_than_eight(empty):
    eight = astar(empty, Cell(0, 0), Cell(5, 5), diagonal=True)
    four = astar(empty, Cell(0, 0), Cell(5, 5), diagonal=False)
    assert four.cost > eight.cost
    assert math.isclose(four.cost, 10.0)


def test_start_equals_goal_is_a_single_cell_path(empty):
    result = astar(empty, Cell(4, 4), Cell(4, 4))
    assert result.success
    assert result.path == [Cell(4, 4)]
    assert result.cost == 0.0


def test_routes_through_the_only_gap(wall):
    result = astar(wall, Cell(5, 2), Cell(5, 18))
    assert result.success, wall.ascii()
    assert Cell(10, 10) in result.path


def test_reports_failure_when_the_wall_is_solid(wall):
    for row in range(20):
        wall.set(Cell(row, 10), OCCUPIED)

    result = astar(wall, Cell(5, 2), Cell(5, 18))
    assert not result.success
    assert result.reason == "no path exists"
    # And it explored the reachable half rather than giving up immediately.
    assert result.expanded > 50


def test_the_path_never_crosses_an_obstacle(wall):
    result = astar(wall, Cell(0, 0), Cell(19, 19))
    assert result.success
    for cell in result.path:
        assert wall.is_free(cell), f"path enters an obstacle at {cell}"


def test_consecutive_path_cells_are_adjacent(wall):
    result = astar(wall, Cell(0, 0), Cell(19, 19))
    for a, b in zip(result.path, result.path[1:]):
        assert abs(a.row - b.row) <= 1 and abs(a.col - b.col) <= 1


def test_rejects_a_start_inside_an_obstacle(wall):
    result = astar(wall, Cell(0, 10), Cell(5, 18))
    assert not result.success
    assert "start" in result.reason


def test_rejects_a_goal_inside_an_obstacle(wall):
    result = astar(wall, Cell(5, 2), Cell(0, 10))
    assert not result.success
    assert "goal" in result.reason


def test_rejects_coordinates_outside_the_map(empty):
    assert not astar(empty, Cell(-1, 0), Cell(2, 2)).success
    assert not astar(empty, Cell(0, 0), Cell(99, 99)).success


def test_honours_the_expansion_budget(empty):
    result = astar(empty, Cell(0, 0), Cell(19, 19), max_expansions=5)
    assert not result.success
    assert "gave up" in result.reason


def test_is_deterministic_across_runs(wall):
    """Equal-f nodes come off the heap in insertion order, so the same query
    returns the same path every time. Without the tie-break the result varies
    between equally optimal paths and regression tests flap."""
    first = astar(wall, Cell(0, 0), Cell(19, 19))
    for _ in range(5):
        assert astar(wall, Cell(0, 0), Cell(19, 19)).path == first.path


def test_plans_around_an_inflated_obstacle():
    grid = OccupancyGrid(20, 20, 0.1)
    grid.mark_rectangle(0.6, 0.0, 0.7, 1.4)     # wall not quite spanning
    inflated = grid.inflate(0.2)

    raw = astar(grid, Cell(2, 2), Cell(2, 15))
    safe = astar(inflated, Cell(2, 2), Cell(2, 15))

    assert raw.success and safe.success
    # Planning on the inflated grid keeps the robot's body clear, so the path
    # is longer. A path that is not longer means inflation did nothing.
    assert safe.cost > raw.cost


# -- line of sight and simplification --------------------------------------

def test_line_of_sight_is_clear_across_empty_space(empty):
    assert line_of_sight(empty, Cell(0, 0), Cell(19, 19))


def test_line_of_sight_is_blocked_by_a_wall(wall):
    assert not line_of_sight(wall, Cell(5, 2), Cell(5, 18))


def test_line_of_sight_through_the_gap(wall):
    assert line_of_sight(wall, Cell(10, 2), Cell(10, 18))


def test_simplify_collapses_a_straight_run(empty):
    result = astar(empty, Cell(3, 0), Cell(3, 19))
    assert len(result.path) == 20
    assert len(simplify(empty, result.path)) == 2


def test_simplify_keeps_the_endpoints(wall):
    result = astar(wall, Cell(0, 0), Cell(19, 19))
    reduced = simplify(wall, result.path)
    assert reduced[0] == result.path[0]
    assert reduced[-1] == result.path[-1]
    assert len(reduced) < len(result.path)


def test_simplify_never_shortcuts_through_an_obstacle(wall):
    result = astar(wall, Cell(5, 2), Cell(5, 18))
    reduced = simplify(wall, result.path)
    for a, b in zip(reduced, reduced[1:]):
        assert line_of_sight(wall, a, b), (
            f"simplified segment {a}->{b} crosses an obstacle\n{wall.ascii()}"
        )


def test_simplify_leaves_short_paths_alone(empty):
    assert simplify(empty, [Cell(0, 0)]) == [Cell(0, 0)]
    assert simplify(empty, [Cell(0, 0), Cell(0, 1)]) == [Cell(0, 0), Cell(0, 1)]


# -- RRT --------------------------------------------------------------------

def test_rrt_finds_a_path_in_open_space(empty):
    result = rrt(empty, (1.0, 1.0), (8.0, 8.0), seed=7)
    assert result.success, result.reason


def test_rrt_is_reproducible_with_a_seed(empty):
    a = rrt(empty, (1.0, 1.0), (8.0, 8.0), seed=11)
    b = rrt(empty, (1.0, 1.0), (8.0, 8.0), seed=11)
    assert a.path == b.path
    assert a.expanded == b.expanded


def test_rrt_never_tunnels_through_a_wall(wall):
    """Checking only the sampled endpoint — not the whole segment — lets the
    tree jump straight across a thin obstacle. The bug is invisible until the
    robot drives into something."""
    result = rrt(wall, (1.0, 1.0), (9.0, 1.0), seed=3, max_iterations=8000)
    if result.success:
        for cell in result.path:
            assert wall.is_free(cell)
        for a, b in zip(result.path, result.path[1:]):
            assert line_of_sight(wall, a, b)


def test_rrt_rejects_a_start_in_an_obstacle(wall):
    result = rrt(wall, (5.0, 2.5), (1.0, 1.0), seed=1)
    assert not result.success
    assert "start" in result.reason


def test_rrt_gives_up_rather_than_looping_forever(wall):
    for row in range(20):
        wall.set(Cell(row, 10), OCCUPIED)
    result = rrt(wall, (1.0, 1.0), (9.0, 1.0), seed=5, max_iterations=300)
    assert not result.success
    assert result.expanded == 300


def test_astar_beats_rrt_on_cost_in_open_space(empty):
    """The reason A* is the global planner for a floor robot and RRT is not.
    RRT is probabilistically complete but not optimal; on a 2D grid that
    difference is money."""
    optimal = astar(empty, empty.world_to_cell(1.0, 1.0),
                    empty.world_to_cell(8.0, 8.0))
    sampled = rrt(empty, (1.0, 1.0), (8.0, 8.0), seed=7)

    assert optimal.success and sampled.success
    optimal_m = optimal.cost * empty.resolution
    assert sampled.cost >= optimal_m - 1e-6
