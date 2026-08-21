"""Generating a route that a perception system can actually learn from.

**The tension nobody mentions.** A route optimised for *navigation* and a route
optimised for *perception* are different routes, and the difference is not
subtle:

  * **Driving straight past an object** gives many frames and almost no
    viewpoint diversity for anything directly ahead. The bearing to an object
    in front of you barely changes as you approach it, so every frame carries
    the same projection bias and `Track.viewpoint_arc` stays near zero. Dozens
    of sightings, nothing confirmed.
  * **Stopping and rotating** gives frames from many *headings* but a single
    *position*. Rotation changes what is in view; it does not triangulate
    anything, because the camera has not moved. `distinct_viewpoints` counts 1
    no matter how long you spin.
  * **What actually works** is passing an object *to the side*, so the bearing
    sweeps while the robot translates. That is why the patrol below offsets its
    lane from the objects rather than driving at them, and why it is worth
    generating the route deliberately instead of reusing the navigation path.

The shortest path between two waypoints is rarely the one that maps the room.
An inspection robot is not commuting.

Pure Python. The output is a list of poses; the caller decides whether they come
from a simulator, from a recorded bag, or from a live robot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, List, Sequence, Tuple

from robot_core.geometry import normalise_angle

Pose = Tuple[float, float, float]      # x, y, yaw


@dataclass(frozen=True)
class PatrolSettings:
    """Sampling rates, expressed in the units the hardware is specified in.

    `step_m` derives from speed and frame rate rather than being picked: at
    0.45 m/s and 15 Hz the robot moves 3 cm between frames. Sampling much
    coarser than that in simulation would make the simulator easier than
    reality, and a confirmation policy tuned against an easy simulator fails on
    the robot.
    """

    speed_mps: float = 0.45
    camera_hz: float = 15.0
    #: Frames to skip between detections. Running a detector at the full camera
    #: rate is wasted work: consecutive frames 3 cm apart are nearly identical,
    #: and it is the *spread* of viewpoints that confirmation needs, not the
    #: count.
    detect_every: int = 8
    #: Yaw step for an in-place scan, radians.
    scan_step_rad: float = math.radians(20.0)

    @property
    def step_m(self) -> float:
        return self.speed_mps / self.camera_hz


def interpolate(a: Pose, b: Pose, step_m: float) -> Iterator[Pose]:
    """Poses along the straight line from a to b, facing the direction of travel.

    Yaw is the heading of the segment, not interpolated between the endpoints'
    yaws. A differential-drive robot points where it is going; interpolating
    orientation independently produces a crab-walk that no real base performs,
    and the resulting camera views are of places the robot never looked.
    """
    distance = math.hypot(b[0] - a[0], b[1] - a[1])
    if distance < 1e-9:
        yield a
        return

    heading = math.atan2(b[1] - a[1], b[0] - a[0])
    steps = max(1, int(distance / step_m))
    for i in range(steps + 1):
        t = i / steps
        yield (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, heading)


def scan_in_place(pose: Pose, sweep_rad: float, step_rad: float) -> Iterator[Pose]:
    """Rotate on the spot through `sweep_rad`, centred on the current heading.

    Useful for *coverage* — it brings objects into the frame that a straight
    lane would miss entirely. It is useless for *triangulation*, because the
    camera does not translate, and `Tracker` will correctly refuse to confirm
    anything seen only during a scan. Both facts are true at once and the route
    needs both kinds of motion.
    """
    x, y, yaw = pose
    steps = max(1, int(abs(sweep_rad) / step_rad))
    start = yaw - sweep_rad / 2.0
    for i in range(steps + 1):
        yield (x, y, normalise_angle(start + sweep_rad * i / steps))


def route(waypoints: Sequence[Pose], settings: PatrolSettings = PatrolSettings(),
          scan_at: Sequence[int] = (), scan_sweep_rad: float = math.pi) -> List[Pose]:
    """A dense trajectory through the waypoints, with optional scans.

    `scan_at` holds indices into `waypoints` where the robot should stop and
    look around before continuing.
    """
    if len(waypoints) < 2:
        return list(waypoints)

    poses: List[Pose] = []
    for index in range(len(waypoints) - 1):
        segment = list(interpolate(waypoints[index], waypoints[index + 1],
                                   settings.step_m))
        # Drop the shared endpoint so a waypoint is not sampled twice.
        poses.extend(segment if index == 0 else segment[1:])

        if index in scan_at:
            poses.extend(scan_in_place(poses[-1], scan_sweep_rad,
                                       settings.scan_step_rad))

    if len(waypoints) - 1 in scan_at:
        poses.extend(scan_in_place(poses[-1], scan_sweep_rad,
                                   settings.scan_step_rad))
    return poses


def sampled(poses: Sequence[Pose], every: int) -> List[Pose]:
    """Every nth pose — the frames the detector actually runs on."""
    return list(poses[::max(1, every)])


#: A lap of the demo office that sees both halves of the building.
#:
#: The lane through the lobby sits at y = 2.2 rather than through the middle,
#: so the reception desk and the printer pass down the robot's SIDE. Objects to
#: the side sweep in bearing as the robot translates, which is what breaks the
#: ground-plane bias; objects dead ahead do not.
DEMO_WAYPOINTS: List[Pose] = [
    (3.0, 3.0, 0.0),
    (6.0, 2.2, 0.0),
    (8.6, 2.2, 0.0),
    (9.0, 3.2, 1.57),      # turn north for the kitchen doorway
    (9.0, 6.2, 1.57),
    (10.4, 7.0, 0.6),      # into the kitchen
    (9.2, 7.6, 3.14),
    (6.6, 7.4, 3.14),
    (9.0, 5.6, -1.57),     # back south through the doorway
    (9.0, 3.0, -1.57),
    (5.0, 2.4, 3.14),      # west along the lobby
    (3.0, 4.2, 1.57),      # north through the meeting-room doorway
    (3.0, 6.0, 1.57),
    (1.6, 6.4, 2.4),
    (3.0, 3.0, -1.57),     # home
]

#: Waypoint indices worth a look around: the two doorways and the two room
#: centres, where a straight lane sees very little.
DEMO_SCANS: Tuple[int, ...] = (3, 5, 11, 13)


def demo_route(settings: PatrolSettings = PatrolSettings()) -> List[Pose]:
    return route(DEMO_WAYPOINTS, settings, scan_at=DEMO_SCANS,
                 scan_sweep_rad=math.radians(160))


def path_length(poses: Sequence[Pose]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1])
               for a, b in zip(poses, poses[1:]))


def describe(poses: Sequence[Pose], settings: PatrolSettings) -> str:
    length = path_length(poses)
    detections = len(sampled(poses, settings.detect_every))
    return (f"{len(poses)} poses, {length:.1f} m, "
            f"{length / settings.speed_mps:.0f} s at {settings.speed_mps} m/s, "
            f"{detections} detector frames "
            f"(1 in {settings.detect_every})")
