"""A small office, built in code.

Exists so the whole pipeline — grounding, validation, planning, execution — can
be demonstrated and tested with one function call and no simulator, no map
server and no ROS. `robot_bringup` ships the same layout as a Gazebo world and
a YAML semantic map; this is the version that runs in CI.

Layout, 12 m x 9 m, origin at the bottom-left:

    +--------------------------------------------+  9
    |    [meeting table]   |                     |
    |    meeting room      |       kitchen       |
    |                      |                     |
    |====[door]============|============[door]===|  5
    |                                            |
    |   lobby        (dock)         [printer]    |
    |         [reception desk]                   |
    +--------------------------------------------+  0
    0                                            12
"""

from __future__ import annotations

from robot_core.occupancy import OccupancyGrid
from robot_core.semantic_map import Landmark, SemanticMap

RESOLUTION = 0.1
WIDTH_M = 12.0
HEIGHT_M = 9.0


def build_grid() -> OccupancyGrid:
    grid = OccupancyGrid(
        width=int(WIDTH_M / RESOLUTION),
        height=int(HEIGHT_M / RESOLUTION),
        resolution=RESOLUTION,
    )

    # Outer walls, 20 cm thick.
    grid.mark_rectangle(0.0, 0.0, WIDTH_M, 0.2)
    grid.mark_rectangle(0.0, HEIGHT_M - 0.2, WIDTH_M, HEIGHT_M)
    grid.mark_rectangle(0.0, 0.0, 0.2, HEIGHT_M)
    grid.mark_rectangle(WIDTH_M - 0.2, 0.0, WIDTH_M, HEIGHT_M)

    # The dividing wall at y = 5, with two 1.2 m doorways: x = 2.4..3.6 into
    # the meeting room and x = 8.4..9.6 into the kitchen.
    #
    # The doorways are the interesting part of this map. At 1.2 m wide they
    # survive inflation by a 0.22 m robot with 0.6 m to spare, so a planner bug
    # shows up as "cannot reach the kitchen" rather than as a subtly longer
    # path. The first draft of this file put a single doorway directly beneath
    # the meeting-room/kitchen partition, which split it into two 0.5 m halves
    # that inflation closed completely — every route to the top half failed
    # with "no path exists" and the *code* was correct. Map geometry is a real
    # source of navigation bugs, which is why `landmark_markers_node` exists.
    grid.mark_rectangle(0.0, 4.9, 2.4, 5.1)
    grid.mark_rectangle(3.6, 4.9, 8.4, 5.1)
    grid.mark_rectangle(9.6, 4.9, WIDTH_M, 5.1)

    # The wall between the meeting room and the kitchen.
    grid.mark_rectangle(5.9, 5.1, 6.1, HEIGHT_M)

    # Furniture.
    grid.mark_rectangle(3.4, 0.6, 5.6, 1.3)      # reception desk
    grid.mark_rectangle(9.4, 2.4, 10.2, 3.0)     # printer table
    grid.mark_rectangle(1.4, 6.4, 4.6, 7.6)      # meeting table

    return grid


def build_semantic_map() -> SemanticMap:
    return SemanticMap([
        Landmark("lobby", "room", 3.0, 3.0, aliases=["entrance", "foyer"],
                 radius_m=1.5, approach_x=3.0, approach_y=3.0, approach_yaw=0.0),
        Landmark("kitchen", "room", 9.0, 7.0,
                 aliases=["break room", "galley", "coffee"],
                 radius_m=1.5, approach_x=9.0, approach_y=6.5, approach_yaw=1.57),
        Landmark("meeting room", "room", 3.0, 7.0,
                 aliases=["boardroom", "conference room"],
                 radius_m=1.5, approach_x=3.0, approach_y=5.8, approach_yaw=1.57),
        Landmark("charging dock", "waypoint", 6.0, 2.6,
                 aliases=["dock", "home", "base", "charger"],
                 radius_m=0.3, approach_x=6.0, approach_y=2.6, approach_yaw=0.0),
        Landmark("printer", "object", 9.8, 2.7, radius_m=0.5,
                 approach_x=8.9, approach_y=2.7, approach_yaw=0.0),
        Landmark("reception desk", "furniture", 4.5, 0.95, radius_m=1.2,
                 approach_x=4.5, approach_y=2.2, approach_yaw=-1.57),
    ])


def build() -> tuple:
    """(grid, semantic_map) for the demo office."""
    return build_grid(), build_semantic_map()
