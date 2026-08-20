"""Occupancy grid — the world model everything else plans against.

Deliberately free of any ROS import. That is the central design decision in
this repository: **thin nodes, fat library.** Logic that lives inside a ROS node
can only be exercised by starting a ROS graph, which is why so much robotics
code is never unit-tested at all. Everything here is ordinary Python and runs
in milliseconds under pytest.

The grid follows the ROS `nav_msgs/OccupancyGrid` convention so conversion is
trivial, but it does not depend on it:

    -1  unknown
     0  free
   100  occupied

Row-major, origin at the bottom-left corner in world coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence, Tuple

UNKNOWN = -1
FREE = 0
OCCUPIED = 100

# Anything at or above this is treated as blocked. Nav2 uses a similar
# threshold; unknown cells are treated as blocked by default because driving
# confidently into unmapped space is how robots find stairs.
OCCUPIED_THRESHOLD = 50


@dataclass(frozen=True)
class Cell:
    row: int
    col: int

    def __iter__(self) -> Iterator[int]:
        yield self.row
        yield self.col


@dataclass
class OccupancyGrid:
    width: int                      # cells along x
    height: int                     # cells along y
    resolution: float               # metres per cell
    origin_x: float = 0.0
    origin_y: float = 0.0
    data: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("grid must have positive dimensions")
        if self.resolution <= 0:
            raise ValueError("resolution must be positive")
        if not self.data:
            self.data = [FREE] * (self.width * self.height)
        elif len(self.data) != self.width * self.height:
            raise ValueError(
                f"data has {len(self.data)} cells, expected "
                f"{self.width * self.height}"
            )

    # -- indexing ------------------------------------------------------------

    def index(self, cell: Cell) -> int:
        return cell.row * self.width + cell.col

    def in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell.row < self.height and 0 <= cell.col < self.width

    def get(self, cell: Cell) -> int:
        if not self.in_bounds(cell):
            # Out of bounds is not free space. Returning FREE here would let a
            # planner route the robot off the edge of the map.
            return OCCUPIED
        return self.data[self.index(cell)]

    def set(self, cell: Cell, value: int) -> None:
        if self.in_bounds(cell):
            self.data[self.index(cell)] = value

    # -- coordinate conversion ----------------------------------------------

    def world_to_cell(self, x: float, y: float) -> Cell:
        col = int(math.floor((x - self.origin_x) / self.resolution))
        row = int(math.floor((y - self.origin_y) / self.resolution))
        return Cell(row, col)

    def cell_to_world(self, cell: Cell) -> Tuple[float, float]:
        # Centre of the cell, not its corner. Returning the corner puts goals
        # half a cell off, which at 5 cm resolution is survivable and at 50 cm
        # is not.
        x = self.origin_x + (cell.col + 0.5) * self.resolution
        y = self.origin_y + (cell.row + 0.5) * self.resolution
        return x, y

    # -- occupancy queries ---------------------------------------------------

    def is_free(self, cell: Cell, treat_unknown_as_blocked: bool = True) -> bool:
        value = self.get(cell)
        if value == UNKNOWN:
            return not treat_unknown_as_blocked
        return value < OCCUPIED_THRESHOLD

    def neighbours(self, cell: Cell, diagonal: bool = True) -> List[Cell]:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if diagonal:
            steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        out: List[Cell] = []
        for d_row, d_col in steps:
            candidate = Cell(cell.row + d_row, cell.col + d_col)
            if not self.in_bounds(candidate):
                continue
            # No corner cutting: a diagonal move is allowed only if BOTH
            # orthogonal cells it passes between are free. One blocked corner
            # is enough to refuse, because the robot has width — squeezing
            # diagonally past a single obstacle corner is a path that looks
            # fine on screen and scrapes the doorframe in reality. Nav2's
            # planners apply the same rule.
            if diagonal and d_row != 0 and d_col != 0:
                if not (self.is_free(Cell(cell.row + d_row, cell.col))
                        and self.is_free(Cell(cell.row, cell.col + d_col))):
                    continue
            out.append(candidate)
        return out

    # -- editing -------------------------------------------------------------

    def mark_rectangle(self, x0: float, y0: float, x1: float, y1: float,
                       value: int = OCCUPIED) -> None:
        a = self.world_to_cell(min(x0, x1), min(y0, y1))
        b = self.world_to_cell(max(x0, x1), max(y0, y1))
        for row in range(a.row, b.row + 1):
            for col in range(a.col, b.col + 1):
                self.set(Cell(row, col), value)

    def inflate(self, radius_m: float) -> "OccupancyGrid":
        """Grow obstacles by the robot's radius.

        This is why a planner can treat the robot as a point. Without
        inflation, a path that passes exactly along a wall is 'valid' for a
        zero-width robot and a collision for a real one — the single most
        common reason a planned path fails in the physical world.
        """
        cells = max(0, int(math.ceil(radius_m / self.resolution)))
        if cells == 0:
            return OccupancyGrid(self.width, self.height, self.resolution,
                                 self.origin_x, self.origin_y, list(self.data))

        inflated = list(self.data)
        for row in range(self.height):
            for col in range(self.width):
                if self.data[row * self.width + col] < OCCUPIED_THRESHOLD:
                    continue
                for d_row in range(-cells, cells + 1):
                    for d_col in range(-cells, cells + 1):
                        # Circular, not square: a square kernel over-inflates
                        # corners by up to 41% and closes gaps the robot fits
                        # through.
                        if d_row * d_row + d_col * d_col > cells * cells:
                            continue
                        r, c = row + d_row, col + d_col
                        if 0 <= r < self.height and 0 <= c < self.width:
                            inflated[r * self.width + c] = OCCUPIED

        return OccupancyGrid(self.width, self.height, self.resolution,
                             self.origin_x, self.origin_y, inflated)

    # -- interop -------------------------------------------------------------

    @classmethod
    def from_ros(cls, msg) -> "OccupancyGrid":
        """Build from nav_msgs/OccupancyGrid without importing it."""
        return cls(
            width=msg.info.width,
            height=msg.info.height,
            resolution=msg.info.resolution,
            origin_x=msg.info.origin.position.x,
            origin_y=msg.info.origin.position.y,
            data=list(msg.data),
        )

    def occupied_fraction(self) -> float:
        blocked = sum(1 for v in self.data if v >= OCCUPIED_THRESHOLD)
        return blocked / len(self.data)

    def ascii(self, path: Optional[Sequence[Cell]] = None) -> str:
        """Render for debugging and for test failure messages.

        A failing path-planning assertion that prints a map is worth an hour of
        stepping through a debugger.
        """
        marks = {(c.row, c.col) for c in (path or [])}
        rows = []
        for row in reversed(range(self.height)):
            line = ""
            for col in range(self.width):
                if (row, col) in marks:
                    line += "*"
                    continue
                value = self.data[row * self.width + col]
                line += "#" if value >= OCCUPIED_THRESHOLD else (
                    "?" if value == UNKNOWN else "."
                )
            rows.append(line)
        return "\n".join(rows)
