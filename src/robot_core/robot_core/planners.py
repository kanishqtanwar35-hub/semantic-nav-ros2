"""Path planners — A* on the grid, RRT in continuous space.

Both are here because they fail differently, and knowing which to reach for is
most of the skill:

  **A*** is complete and optimal on the grid it is given. It is the right
  choice for a differential-drive robot on a 2D floor plan, which is exactly
  what Nav2's global planner does.

  **RRT** is probabilistically complete and *not* optimal. It shines where the
  configuration space is high-dimensional enough that a grid is intractable —
  a 6-DOF arm has a 6D configuration space, and a grid over that is
  astronomically large.

Using RRT for 2D floor navigation is a common portfolio mistake: it produces
jagged, non-optimal paths for a problem A* solves exactly.

No ROS imports. Everything below is testable with plain pytest.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from robot_core.occupancy import Cell, OccupancyGrid


@dataclass
class PlanResult:
    success: bool
    path: List[Cell] = field(default_factory=list)
    cost: float = 0.0
    expanded: int = 0
    reason: str = ""

    def __bool__(self) -> bool:
        return self.success


# ---------------------------------------------------------------------------
# A*
# ---------------------------------------------------------------------------

def octile(a: Cell, b: Cell) -> float:
    """Heuristic for 8-connected grids.

    Manhattan over-estimates when diagonals are allowed, which breaks A*'s
    optimality guarantee — the heuristic must never exceed the true cost.
    Euclidean is admissible but loose, so it expands more nodes than
    necessary. Octile is the tight admissible choice for this connectivity,
    and picking it correctly is the difference between "A* works" and "A*
    works and I know why".
    """
    d_row = abs(a.row - b.row)
    d_col = abs(a.col - b.col)
    return (d_row + d_col) + (math.sqrt(2) - 2) * min(d_row, d_col)


def step_cost(a: Cell, b: Cell) -> float:
    return math.sqrt(2) if (a.row != b.row and a.col != b.col) else 1.0


def astar(grid: OccupancyGrid, start: Cell, goal: Cell,
          diagonal: bool = True, max_expansions: int = 200_000) -> PlanResult:
    if not grid.in_bounds(start):
        return PlanResult(False, reason="start is outside the map")
    if not grid.in_bounds(goal):
        return PlanResult(False, reason="goal is outside the map")
    if not grid.is_free(start):
        return PlanResult(False, reason="start is in an obstacle")
    if not grid.is_free(goal):
        return PlanResult(False, reason="goal is in an obstacle")
    if start == goal:
        return PlanResult(True, path=[start], cost=0.0)

    open_heap: List[Tuple[float, int, Cell]] = []
    # Tie-break on insertion order so the result is deterministic. Without it,
    # equal-f nodes come out in whatever order the heap happens to produce and
    # the same query can return different (equally optimal) paths between runs
    # — which makes regression tests flap.
    counter = 0
    heapq.heappush(open_heap, (octile(start, goal), counter, start))

    came_from: Dict[Cell, Cell] = {}
    g_score: Dict[Cell, float] = {start: 0.0}
    closed = set()
    expanded = 0

    while open_heap:
        _, _, current = heapq.heappop(open_heap)

        if current in closed:
            continue
        closed.add(current)
        expanded += 1

        if expanded > max_expansions:
            return PlanResult(False, expanded=expanded,
                              reason=f"gave up after {max_expansions} expansions")

        if current == goal:
            path = [current]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            path.reverse()
            return PlanResult(True, path=path, cost=g_score[current],
                              expanded=expanded)

        for neighbour in grid.neighbours(current, diagonal=diagonal):
            if neighbour in closed or not grid.is_free(neighbour):
                continue
            tentative = g_score[current] + step_cost(current, neighbour)
            if tentative < g_score.get(neighbour, math.inf):
                came_from[neighbour] = current
                g_score[neighbour] = tentative
                counter += 1
                heapq.heappush(
                    open_heap,
                    (tentative + octile(neighbour, goal), counter, neighbour),
                )

    return PlanResult(False, expanded=expanded, reason="no path exists")


def simplify(grid: OccupancyGrid, path: Sequence[Cell]) -> List[Cell]:
    """Drop waypoints that a straight line already covers.

    A* returns a cell-by-cell path. Feeding every cell to a controller produces
    a stuttering robot that stops at each one, so real stacks smooth the path
    first. This is line-of-sight simplification, the cheap version of what
    Nav2's smoother does.
    """
    if len(path) <= 2:
        return list(path)

    out = [path[0]]
    anchor = 0
    for i in range(2, len(path)):
        if not line_of_sight(grid, path[anchor], path[i]):
            out.append(path[i - 1])
            anchor = i - 1
    out.append(path[-1])
    return out


def line_of_sight(grid: OccupancyGrid, a: Cell, b: Cell) -> bool:
    """Bresenham between two cells, blocked if it crosses an obstacle."""
    row0, col0 = a.row, a.col
    row1, col1 = b.row, b.col
    d_row = abs(row1 - row0)
    d_col = abs(col1 - col0)
    step_row = 1 if row1 > row0 else -1
    step_col = 1 if col1 > col0 else -1
    error = d_col - d_row

    while True:
        if not grid.is_free(Cell(row0, col0)):
            return False
        if row0 == row1 and col0 == col1:
            return True
        doubled = error * 2
        if doubled > -d_row:
            error -= d_row
            col0 += step_col
        if doubled < d_col:
            error += d_col
            row0 += step_row


# ---------------------------------------------------------------------------
# RRT
# ---------------------------------------------------------------------------

@dataclass
class RRTNode:
    x: float
    y: float
    parent: Optional[int] = None


def rrt(grid: OccupancyGrid, start: Tuple[float, float], goal: Tuple[float, float],
        step: float = 0.5, goal_bias: float = 0.1, max_iterations: int = 5000,
        goal_tolerance: float = 0.4, seed: Optional[int] = None) -> PlanResult:
    """Rapidly-exploring Random Tree in world coordinates.

    `seed` exists so tests are deterministic. A randomised planner with no seed
    produces a test suite that fails once a fortnight for no reason, and the
    usual response — deleting the test — is worse than the flake.

    `goal_bias` is the one parameter that matters: sampling the goal
    occasionally is what turns RRT from a space-filling tree into a planner.
    At 0.0 it explores forever; at 1.0 it degenerates into a straight-line
    attempt that fails on the first obstacle.
    """
    rng = random.Random(seed)

    start_cell = grid.world_to_cell(*start)
    goal_cell = grid.world_to_cell(*goal)
    if not grid.is_free(start_cell):
        return PlanResult(False, reason="start is in an obstacle")
    if not grid.is_free(goal_cell):
        return PlanResult(False, reason="goal is in an obstacle")

    min_x = grid.origin_x
    min_y = grid.origin_y
    max_x = grid.origin_x + grid.width * grid.resolution
    max_y = grid.origin_y + grid.height * grid.resolution

    nodes: List[RRTNode] = [RRTNode(start[0], start[1])]

    for iteration in range(max_iterations):
        if rng.random() < goal_bias:
            sample = goal
        else:
            sample = (rng.uniform(min_x, max_x), rng.uniform(min_y, max_y))

        nearest_index = min(
            range(len(nodes)),
            key=lambda i: (nodes[i].x - sample[0]) ** 2 + (nodes[i].y - sample[1]) ** 2,
        )
        nearest = nodes[nearest_index]

        angle = math.atan2(sample[1] - nearest.y, sample[0] - nearest.x)
        new_x = nearest.x + step * math.cos(angle)
        new_y = nearest.y + step * math.sin(angle)

        if not (min_x <= new_x <= max_x and min_y <= new_y <= max_y):
            continue

        # Check the whole segment, not just the endpoint. Checking endpoints
        # only lets the tree tunnel straight through a thin wall — the classic
        # RRT bug, and it is invisible until the robot hits something.
        if not line_of_sight(grid, grid.world_to_cell(nearest.x, nearest.y),
                             grid.world_to_cell(new_x, new_y)):
            continue

        nodes.append(RRTNode(new_x, new_y, parent=nearest_index))

        if math.hypot(new_x - goal[0], new_y - goal[1]) <= goal_tolerance:
            path_cells: List[Cell] = []
            index: Optional[int] = len(nodes) - 1
            cost = 0.0
            previous: Optional[RRTNode] = None
            while index is not None:
                node = nodes[index]
                path_cells.append(grid.world_to_cell(node.x, node.y))
                if previous is not None:
                    cost += math.hypot(node.x - previous.x, node.y - previous.y)
                previous = node
                index = node.parent
            path_cells.reverse()
            return PlanResult(True, path=path_cells, cost=cost,
                              expanded=iteration + 1)

    return PlanResult(False, expanded=max_iterations,
                      reason=f"no path found in {max_iterations} samples")
