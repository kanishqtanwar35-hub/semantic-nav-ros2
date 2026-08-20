import math

import pytest

from robot_core.geometry import Pose2D
from robot_core.semantic_map import Landmark, SemanticMap


@pytest.fixture
def office():
    return SemanticMap([
        Landmark("kitchen", "room", 4.0, 2.0, aliases=["galley", "break room"],
                 radius_m=1.5),
        Landmark("meeting room", "room", -3.0, 5.0, aliases=["boardroom"],
                 radius_m=2.0),
        Landmark("charging dock", "waypoint", 0.0, 0.0, aliases=["dock", "home"]),
        Landmark("reception desk", "furniture", 1.0, -4.0, yaw=math.pi / 2,
                 radius_m=0.8),
        Landmark("printer", "object", 2.0, 3.0, radius_m=0.3),
    ])


# -- construction -----------------------------------------------------------

def test_length_and_names(office):
    assert len(office) == 5
    assert "kitchen" in office.names()


def test_duplicate_names_are_rejected():
    landmarks = SemanticMap([Landmark("kitchen", "room", 0, 0)])
    with pytest.raises(ValueError):
        landmarks.add(Landmark("The Kitchen", "room", 5, 5))


def test_lookup_is_case_and_article_insensitive(office):
    assert office.get("KITCHEN") is office.get("the kitchen")


# -- resolution -------------------------------------------------------------

def test_exact_match_scores_one(office):
    result = office.resolve("kitchen")
    assert result.landmark.name == "kitchen"
    assert result.score == 1.0
    assert result.confident


def test_the_definite_article_is_stripped(office):
    assert office.resolve("the kitchen").score == 1.0


def test_aliases_resolve(office):
    assert office.resolve("break room").landmark.name == "kitchen"
    assert office.resolve("boardroom").landmark.name == "meeting room"
    assert office.resolve("home").landmark.name == "charging dock"


def test_typos_still_resolve(office):
    result = office.resolve("meeting rom")
    assert result.landmark.name == "meeting room"
    assert result.confident


def test_unknown_places_resolve_to_nothing(office):
    """Refusing is the correct answer. A robot that maps 'the roof' onto the
    nearest-sounding landmark and drives there confidently is worse than one
    that says it does not know the place."""
    result = office.resolve("the helicopter pad")
    assert result.landmark is None
    assert not result.confident
    assert "no landmark resembles" in result.matched_on


def test_an_empty_query_resolves_to_nothing(office):
    assert office.resolve("   ").landmark is None


def test_resolution_reports_the_runners_up(office):
    """The alternatives are what a clarifying question is built from."""
    result = office.resolve("kitchenette sink area")
    assert result.alternatives
    assert len(result.alternatives) <= 3


def test_ambiguity_is_detected_rather_than_guessed():
    """Two candidates within a hair of each other is a coin flip. The right
    behaviour is to ask, not to pick."""
    ambiguous = SemanticMap([
        Landmark("office kitchen", "room", 1, 1),
        Landmark("staff kitchen", "room", 9, 9),
    ])
    result = ambiguous.resolve("the kitchen")
    assert result.landmark is not None
    assert result.ambiguous


def test_a_clear_winner_is_not_flagged_ambiguous(office):
    assert not office.resolve("kitchen").ambiguous


# -- approach poses ---------------------------------------------------------

def test_default_approach_stands_off_the_landmark(office):
    """The centroid of 'the desk' is inside the desk. A goal there is
    unreachable, and Nav2 will spend a minute failing to get to it."""
    desk = office.get("reception desk")
    approach = desk.approach_pose
    assert approach.distance_to(desk.pose) > desk.radius_m


def test_default_approach_faces_the_landmark(office):
    printer = office.get("printer")
    approach = printer.approach_pose
    bearing = approach.bearing_to(printer.pose)
    assert math.isclose(bearing, 0.0, abs_tol=1e-9)


def test_explicit_approach_overrides_the_default():
    landmark = Landmark("sofa", "furniture", 5.0, 5.0, radius_m=1.0,
                        approach_x=3.0, approach_y=5.0)
    approach = landmark.approach_pose
    assert (approach.x, approach.y) == (3.0, 5.0)
    # Yaw is derived so the robot arrives facing the thing it was sent to.
    assert math.isclose(approach.yaw, 0.0, abs_tol=1e-9)


def test_explicit_approach_yaw_is_respected():
    landmark = Landmark("sofa", "furniture", 5.0, 5.0,
                        approach_x=3.0, approach_y=5.0, approach_yaw=1.25)
    assert landmark.approach_pose.yaw == 1.25


# -- spatial ----------------------------------------------------------------

def test_nearest(office):
    assert office.nearest(Pose2D(3.9, 2.1)).name == "kitchen"


def test_nearest_filtered_by_category(office):
    assert office.nearest(Pose2D(3.9, 2.1), category="waypoint").name == "charging dock"


def test_nearest_returns_none_for_an_unknown_category(office):
    assert office.nearest(Pose2D(0, 0), category="spaceship") is None


def test_within_is_sorted_by_distance(office):
    found = office.within(Pose2D(0.0, 0.0), radius_m=6.0)
    distances = [Pose2D(0, 0).distance_to(lm.pose) for lm in found]
    assert distances == sorted(distances)


def test_by_category(office):
    assert {lm.name for lm in office.by_category("room")} == {
        "kitchen", "meeting room"
    }


def test_describe_lists_every_landmark_grouped(office):
    described = office.describe()
    for name in office.names():
        assert name in described
    assert "room:" in described


# -- persistence ------------------------------------------------------------

def test_round_trips_through_yaml(tmp_path, office):
    path = tmp_path / "map.yaml"
    office.save(path)
    reloaded = SemanticMap.load(path)

    assert len(reloaded) == len(office)
    for name in office.names():
        original, copy = office.get(name), reloaded.get(name)
        assert (original.x, original.y, original.yaw) == (copy.x, copy.y, copy.yaw)
        assert original.aliases == copy.aliases
        assert original.category == copy.category
        assert original.radius_m == copy.radius_m


def test_saved_yaml_is_human_editable(tmp_path, office):
    path = tmp_path / "map.yaml"
    office.save(path)
    text = path.read_text(encoding="utf-8")
    assert "landmarks:" in text
    assert "kitchen" in text


def test_explicit_approach_survives_a_round_trip(tmp_path):
    original = SemanticMap([
        Landmark("sofa", "furniture", 5.0, 5.0,
                 approach_x=3.0, approach_y=5.0, approach_yaw=0.5)
    ])
    path = tmp_path / "map.yaml"
    original.save(path)
    reloaded = SemanticMap.load(path)
    assert reloaded.get("sofa").approach_pose.yaw == 0.5
