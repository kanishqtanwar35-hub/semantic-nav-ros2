import math

import pytest

from robot_core.occupancy import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    Cell,
    OccupancyGrid,
)


def test_rejects_bad_dimensions():
    with pytest.raises(ValueError):
        OccupancyGrid(0, 10, 0.1)
    with pytest.raises(ValueError):
        OccupancyGrid(10, 10, 0.0)


def test_rejects_data_of_the_wrong_length():
    with pytest.raises(ValueError):
        OccupancyGrid(4, 4, 0.1, data=[FREE] * 15)


def test_defaults_to_all_free():
    grid = OccupancyGrid(5, 4, 0.1)
    assert len(grid.data) == 20
    assert grid.occupied_fraction() == 0.0


def test_round_trip_world_to_cell():
    grid = OccupancyGrid(10, 10, 0.5, origin_x=-2.0, origin_y=-3.0)
    cell = grid.world_to_cell(0.3, 1.1)
    x, y = grid.cell_to_world(cell)
    # cell_to_world returns the CENTRE, so the point maps back within half a
    # cell — never further.
    assert abs(x - 0.3) <= grid.resolution / 2
    assert abs(y - 1.1) <= grid.resolution / 2


def test_cell_to_world_returns_the_centre_not_the_corner():
    grid = OccupancyGrid(4, 4, 1.0)
    assert grid.cell_to_world(Cell(0, 0)) == (0.5, 0.5)


def test_out_of_bounds_reads_as_occupied_not_free():
    # If this returned FREE, a planner could route the robot off the map.
    grid = OccupancyGrid(3, 3, 0.1)
    assert grid.get(Cell(-1, 0)) == OCCUPIED
    assert grid.get(Cell(0, 99)) == OCCUPIED
    assert not grid.is_free(Cell(-1, -1))


def test_unknown_is_blocked_by_default():
    grid = OccupancyGrid(3, 3, 0.1)
    grid.set(Cell(1, 1), UNKNOWN)
    assert not grid.is_free(Cell(1, 1))
    assert grid.is_free(Cell(1, 1), treat_unknown_as_blocked=False)


def test_set_outside_bounds_is_a_no_op_not_a_crash():
    grid = OccupancyGrid(3, 3, 0.1)
    grid.set(Cell(50, 50), OCCUPIED)
    assert grid.occupied_fraction() == 0.0


def test_diagonal_neighbours_refuse_to_cut_a_blocked_corner():
    grid = OccupancyGrid(3, 3, 0.1)
    grid.set(Cell(1, 2), OCCUPIED)
    grid.set(Cell(2, 1), OCCUPIED)

    neighbours = grid.neighbours(Cell(1, 1), diagonal=True)
    assert Cell(2, 2) not in neighbours


def test_one_blocked_corner_is_enough_to_refuse_the_diagonal():
    """A point robot could squeeze past a single corner. A real one has width,
    so the strict rule is the correct one — and it is what Nav2 uses."""
    grid = OccupancyGrid(3, 3, 0.1)
    grid.set(Cell(1, 2), OCCUPIED)

    neighbours = grid.neighbours(Cell(1, 1), diagonal=True)
    assert Cell(2, 2) not in neighbours
    assert Cell(2, 0) in neighbours      # the unobstructed diagonal is fine


def test_diagonals_are_allowed_when_both_corners_are_free():
    grid = OccupancyGrid(3, 3, 0.1)
    neighbours = grid.neighbours(Cell(1, 1), diagonal=True)
    assert Cell(2, 2) in neighbours


def test_four_connected_has_no_diagonals():
    grid = OccupancyGrid(3, 3, 0.1)
    assert len(grid.neighbours(Cell(1, 1), diagonal=False)) == 4
    assert len(grid.neighbours(Cell(1, 1), diagonal=True)) == 8


def test_one_obstacle_removes_the_two_diagonals_it_guards():
    grid = OccupancyGrid(3, 3, 0.1)
    before = grid.neighbours(Cell(1, 1), diagonal=True)
    grid.set(Cell(1, 2), OCCUPIED)
    after = grid.neighbours(Cell(1, 1), diagonal=True)

    assert set(before) - set(after) == {Cell(0, 2), Cell(2, 2)}
    # The blocked cell itself is still *listed*. `neighbours` answers "which
    # cells are geometrically reachable", and `is_free` answers "may I go
    # there" — keeping them separate is what lets a caller reason about
    # unknown space differently from occupied space.
    assert Cell(1, 2) in after


def test_neighbours_are_clipped_at_the_edge():
    grid = OccupancyGrid(3, 3, 0.1)
    assert len(grid.neighbours(Cell(0, 0), diagonal=True)) == 3


def test_mark_rectangle_blocks_the_region():
    grid = OccupancyGrid(10, 10, 1.0)
    grid.mark_rectangle(2.0, 2.0, 4.0, 4.0)
    assert not grid.is_free(grid.world_to_cell(3.0, 3.0))
    assert grid.is_free(grid.world_to_cell(8.0, 8.0))


def test_mark_rectangle_accepts_reversed_corners():
    grid = OccupancyGrid(10, 10, 1.0)
    grid.mark_rectangle(4.0, 4.0, 2.0, 2.0)
    assert not grid.is_free(grid.world_to_cell(3.0, 3.0))


def test_inflate_grows_obstacles_by_the_robot_radius():
    grid = OccupancyGrid(11, 11, 0.1)
    grid.set(Cell(5, 5), OCCUPIED)

    inflated = grid.inflate(0.25)   # 0.25m / 0.1m = 3 cells

    assert not inflated.is_free(Cell(5, 8))
    assert inflated.is_free(Cell(5, 9))
    # The original must not be mutated — the planner needs both.
    assert grid.is_free(Cell(5, 8))


def test_inflation_kernel_is_circular_not_square():
    # A square kernel would block the corner at (r, r); a circular one does not,
    # because sqrt(2)*r > r. Over-inflating corners closes doorways the robot
    # actually fits through.
    grid = OccupancyGrid(11, 11, 0.1)
    grid.set(Cell(5, 5), OCCUPIED)

    inflated = grid.inflate(0.3)    # 3 cells

    assert not inflated.is_free(Cell(5, 8))       # straight out: blocked
    assert inflated.is_free(Cell(8, 8))           # diagonal corner: still free


def test_inflate_with_zero_radius_copies_rather_than_aliases():
    grid = OccupancyGrid(4, 4, 0.1)
    copy = grid.inflate(0.0)
    copy.set(Cell(0, 0), OCCUPIED)
    assert grid.is_free(Cell(0, 0))


def test_from_ros_needs_no_nav_msgs_import():
    class _P:
        x, y = -1.0, -2.0

    class _O:
        position = _P()

    class _Info:
        width, height, resolution = 3, 2, 0.25
        origin = _O()

    class _Msg:
        info = _Info()
        data = [FREE] * 6

    grid = OccupancyGrid.from_ros(_Msg())
    assert (grid.width, grid.height) == (3, 2)
    assert grid.resolution == 0.25
    assert (grid.origin_x, grid.origin_y) == (-1.0, -2.0)


def test_ascii_renders_top_row_first():
    grid = OccupancyGrid(3, 2, 1.0)
    grid.set(Cell(1, 0), OCCUPIED)   # top-left in world orientation
    rendered = grid.ascii()
    assert rendered.splitlines()[0].startswith("#")
    assert rendered.splitlines()[1] == "..."


def test_ascii_marks_the_path():
    grid = OccupancyGrid(3, 1, 1.0)
    assert grid.ascii(path=[Cell(0, 1)]) == ".*."


def test_occupied_fraction():
    grid = OccupancyGrid(4, 1, 1.0)
    grid.set(Cell(0, 0), OCCUPIED)
    assert math.isclose(grid.occupied_fraction(), 0.25)
