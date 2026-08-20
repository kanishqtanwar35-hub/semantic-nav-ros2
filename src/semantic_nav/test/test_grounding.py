import pytest

from semantic_nav.commands import Verb
from semantic_nav.demo_world import build_semantic_map
from semantic_nav.grounding import (
    MAX_UTTERANCE_CHARS,
    Grounder,
    clean_target,
    from_json,
    parse_json_text,
    parse_rules,
)
from semantic_nav.llm import LLMUnavailable, ScriptedBackend


@pytest.fixture
def semantic_map():
    return build_semantic_map()


def verbs(command):
    return [step.verb for step in command.steps]


def targets(command):
    return [step.target for step in command.steps]


# -- the urgent commands ----------------------------------------------------

@pytest.mark.parametrize("utterance", [
    "stop", "STOP", "stop!", "halt", "freeze", "abort", "emergency stop",
    "e-stop", "cancel that", "stand down",
])
def test_stop_is_understood_offline_in_every_phrasing(utterance):
    """These must never depend on a network call, a key, or a model being up.
    That is the whole reason the rule parser exists and runs first."""
    assert verbs(parse_rules(utterance)) == [Verb.STOP]


def test_stop_short_circuits_the_rest_of_the_sentence():
    """"go to the kitchen, no wait, stop" must not drive to the kitchen.

    A parser that processes clauses in order and appends STOP at the end would
    execute the navigation first and stop afterwards — which is the opposite of
    what was asked, and the robot is already moving by the time anyone notices.
    """
    command = parse_rules("go to the kitchen, no wait, stop")
    assert verbs(command) == [Verb.STOP]


@pytest.mark.parametrize("utterance", [
    "go home", "return home", "go back to the dock", "dock",
    "return to base", "charge yourself",
])
def test_going_home_is_understood_offline(utterance):
    assert Verb.RETURN_HOME in verbs(parse_rules(utterance))


# -- go_to ------------------------------------------------------------------

@pytest.mark.parametrize("utterance,expected", [
    ("go to the kitchen", "kitchen"),
    ("drive to the lobby", "lobby"),
    ("navigate to the printer", "printer"),
    ("head over to the meeting room", "meeting room"),
    ("move towards the printer", "printer"),
    ("take me to the kitchen", "kitchen"),
    ("go to the kitchen please", "kitchen"),
    ("go to the lobby now", "lobby"),
    ("Go To The Kitchen.", "Kitchen"),
])
def test_goto_phrasings(utterance, expected):
    command = parse_rules(utterance)
    assert verbs(command) == [Verb.GO_TO]
    assert targets(command) == [expected]


def test_the_users_own_wording_is_preserved_until_resolution():
    """The parser extracts a phrase; it does not decide what that phrase means.
    Casing and spelling stay as spoken so a refusal can quote the user back to
    themselves, and `SemanticMap.resolve` does the normalising."""
    assert parse_rules("Go To The Kitchen.").steps[0].target == "Kitchen"


def test_trailing_politeness_is_not_part_of_the_place_name():
    assert clean_target("the kitchen please") == "kitchen"
    assert clean_target("the lobby, thanks.") == "lobby"


# -- sequences --------------------------------------------------------------

def test_a_then_b():
    command = parse_rules("go to the kitchen then go to the lobby")
    assert verbs(command) == [Verb.GO_TO, Verb.GO_TO]
    assert targets(command) == ["kitchen", "lobby"]


def test_a_then_bare_landmark():
    command = parse_rules("go to the kitchen then the printer")
    assert targets(command) == ["kitchen", "printer"]


def test_a_bare_landmark_alone_is_not_a_command():
    """"kitchen" on its own could be an answer to a question, half a sentence,
    or someone talking to a colleague. Acting on it is guessing."""
    assert parse_rules("kitchen").steps == []


def test_mixed_sequence_with_a_wait():
    command = parse_rules("go to the printer then wait 30 seconds then go home")
    assert verbs(command) == [Verb.GO_TO, Verb.WAIT, Verb.RETURN_HOME]
    assert command.steps[1].seconds == 30.0


@pytest.mark.parametrize("utterance,seconds", [
    ("wait 10 seconds", 10.0),
    ("wait 10 secs", 10.0),
    ("wait 5s", 5.0),
    ("wait for 2 minutes", 120.0),
    ("wait 1 min", 60.0),
])
def test_wait_units(utterance, seconds):
    command = parse_rules(utterance)
    assert command.steps[0].seconds == seconds


def test_a_bare_wait_gets_a_default_not_forever():
    command = parse_rules("hold on")
    assert command.steps[0].verb is Verb.WAIT
    assert command.steps[0].seconds == 5.0


# -- patrol -----------------------------------------------------------------

def test_patrol_expands_into_one_step_per_location():
    command = parse_rules("patrol the lobby and the kitchen")
    assert verbs(command) == [Verb.PATROL, Verb.PATROL]
    assert targets(command) == ["lobby", "kitchen"]


def test_patrol_with_a_comma_separated_list():
    command = parse_rules("patrol the lobby, the kitchen and the meeting room")
    assert targets(command) == ["lobby", "kitchen", "meeting room"]


def test_patrol_with_no_locations_produces_no_steps():
    command = parse_rules("patrol")
    assert command.steps == []
    assert any("named no locations" in note for note in command.notes)


# -- abstention -------------------------------------------------------------

def test_an_empty_utterance_produces_nothing():
    assert parse_rules("").steps == []
    assert parse_rules("   ").steps == []
    assert parse_rules(None).steps == []


def test_gibberish_abstains_rather_than_guessing():
    """Abstaining is what triggers escalation to the model. A rule parser that
    guesses is worse than one that gives up, because a wrong guess reaches the
    motors exactly as fast as a right one."""
    command = parse_rules("qwertyuiop asdfgh")
    assert command.steps == []
    assert command.notes


def test_very_long_input_is_truncated():
    command = parse_rules("go to the kitchen " + "x" * 5000)
    assert len(command.utterance) > MAX_UTTERANCE_CHARS   # original is kept
    assert len(command.steps) <= 1                        # parsing is capped


# -- Grounder: rules first, model second ------------------------------------

def test_the_model_is_never_called_when_the_rules_understood(semantic_map):
    """Every avoided call is latency and money. It also means the commands
    people say most often never depend on an external service."""
    backend = ScriptedBackend([{"steps": [{"verb": "go_to", "target": "lobby"}]}])
    grounder = Grounder(semantic_map, llm=backend)

    command = grounder.ground("go to the kitchen")

    assert command.source == "rules"
    assert targets(command) == ["kitchen"]
    assert backend.calls == []


def test_the_model_is_called_when_the_rules_abstain(semantic_map):
    backend = ScriptedBackend([{"steps": [{"verb": "go_to", "target": "kitchen"}]}])
    grounder = Grounder(semantic_map, llm=backend)

    command = grounder.ground("I could really use a coffee about now")

    assert command.source == "llm"
    assert targets(command) == ["kitchen"]
    assert len(backend.calls) == 1


def test_the_prompt_lists_the_known_places(semantic_map):
    """The model cannot pick a place it was never shown, which turns most
    hallucinations into a problem that never arises."""
    backend = ScriptedBackend([{"steps": []}])
    Grounder(semantic_map, llm=backend).ground("something unparseable entirely")

    prompt = backend.calls[0]["system"]
    for name in semantic_map.names():
        assert name in prompt


def test_stop_never_reaches_the_model(semantic_map):
    backend = ScriptedBackend([{"steps": [{"verb": "go_to", "target": "kitchen"}]}])
    grounder = Grounder(semantic_map, llm=backend)

    assert verbs(grounder.ground("stop")) == [Verb.STOP]
    assert backend.calls == []


def test_a_model_failure_degrades_to_not_understood(semantic_map):
    """A node that owns the robot's motion must not raise because an HTTP call
    failed. The correct degradation is 'I did not understand'."""
    backend = ScriptedBackend([LLMUnavailable("no key")])
    grounder = Grounder(semantic_map, llm=backend)

    command = grounder.ground("mumble mumble something")

    assert command.steps == []
    assert any("model unavailable" in note for note in command.notes)


def test_running_with_no_model_at_all(semantic_map):
    grounder = Grounder(semantic_map, llm=None)
    assert targets(grounder.ground("go to the kitchen")) == ["kitchen"]
    assert grounder.ground("waffle waffle").steps == []


# -- model output is untrusted ----------------------------------------------

def test_an_invented_verb_is_dropped():
    """The allowlist doing its job. `Verb(...)` is the only way a verb enters
    the system, so 'self_destruct' dies at the enum conversion."""
    command = from_json({"steps": [
        {"verb": "self_destruct"},
        {"verb": "go_to", "target": "kitchen"},
    ]})
    assert verbs(command) == [Verb.GO_TO]
    assert any("unknown verb" in note for note in command.notes)


@pytest.mark.parametrize("payload", [
    None, [], "steps", 42, {"steps": "kitchen"}, {"steps": None}, {},
])
def test_malformed_model_output_produces_no_steps(payload):
    command = from_json(payload)
    assert command.steps == []
    assert command.notes


def test_non_object_steps_are_skipped():
    command = from_json({"steps": ["kitchen", None, 7,
                                   {"verb": "go_to", "target": "lobby"}]})
    assert targets(command) == ["lobby"]


def test_non_numeric_seconds_is_noted_not_crashed():
    command = from_json({"steps": [{"verb": "wait", "seconds": "soon"}]})
    assert command.steps[0].seconds is None
    assert any("non-numeric" in note for note in command.notes)


# -- extracting JSON from whatever the model sent ---------------------------

def test_plain_json():
    assert parse_json_text('{"steps": []}') == {"steps": []}


def test_json_in_a_code_fence():
    assert parse_json_text('```json\n{"steps": []}\n```') == {"steps": []}


def test_json_with_a_chatty_preamble():
    text = 'Sure! Here you go:\n{"steps": [{"verb": "stop"}]}\nHope that helps.'
    assert parse_json_text(text)["steps"][0]["verb"] == "stop"


@pytest.mark.parametrize("text", ["", "no json here", "{unclosed", "}{"])
def test_unparseable_responses_raise(text):
    with pytest.raises(Exception):
        parse_json_text(text)
