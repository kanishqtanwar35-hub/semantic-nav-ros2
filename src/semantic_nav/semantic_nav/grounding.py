"""Natural language -> Command.

Two parsers, in a deliberate order.

**Rules first, model second.** The obvious architecture is to send everything
to an LLM. It is the wrong one here, for a reason specific to robots:

    "stop" must never depend on a network call.

A rule parser answers in microseconds, offline, deterministically, and cannot
be made to fail by a rate limit or an expired API key. The commands people say
most often and most urgently are exactly the ones with the least linguistic
variation. So the rules take those, and the model handles the long tail —
"head over to where we keep the printer and hang around for a minute".

The secondary benefits are real too: the whole system runs with no API key at
all, and the model is never in the loop for a command the rules already
understood, which is most of them.
"""

from __future__ import annotations

import re
from typing import Optional

from semantic_nav.commands import Command, Step, Verb

#: Anything longer is not an instruction to a robot. Capping input is the
#: cheapest defence against both prompt injection and an accidental paste of a
#: whole document into the command topic.
MAX_UTTERANCE_CHARS = 400

_STOP = re.compile(
    r"\b(stop|halt|freeze|abort|cancel|emergency|e-?stop|stand down)\b", re.I)
_HOME = re.compile(
    r"\b(go home|return home|come home|dock|go back to (the )?(dock|base|charger)"
    r"|charge yourself|return to base)\b", re.I)
_REPORT = re.compile(
    r"\b(where are you|what.s your (status|position|location)|report|status)\b",
    re.I)
_PATROL = re.compile(r"\b(patrol|loop|circuit|do the rounds|sweep)\b", re.I)
_GOTO = re.compile(
    r"\b(?:go|drive|navigate|head|move|travel|proceed)\s+(?:to|over to|towards?|into)\s+"
    r"|^\s*(?:take me to|bring me to|find|visit)\s+", re.I)
_WAIT = re.compile(
    r"\bwait\s+(?:for\s+)?(\d+(?:\.\d+)?)\s*(seconds?|secs?|s|minutes?|mins?|m)\b",
    re.I)
_BARE_WAIT = re.compile(r"\b(wait|hold on|hang on|pause|stay (there|here))\b", re.I)

_SPLIT = re.compile(r"\s*(?:,\s*)?\b(?:then|and then|after that|next)\b\s*", re.I)
_LIST_SPLIT = re.compile(r"\s*(?:,|\band\b)\s*", re.I)

_TRAILING = re.compile(
    r"\s*\b(please|now|right now|thanks|thank you|for me|would you|could you)\b",
    re.I)


def clean_target(text: str) -> str:
    text = _TRAILING.sub("", text)
    text = re.sub(r"^\s*(?:the|a|an|to|towards?|into|over to)\s+", "", text,
                  flags=re.I)
    return re.sub(r"[.!?,;]+\s*$", "", text).strip()


def parse_rules(utterance: str) -> Command:
    """Deterministic parse. Returns an empty Command when it does not understand.

    Emptiness is a real answer, not a failure: it is the signal to escalate to
    the model. A rule parser that guesses is worse than one that abstains,
    because a wrong guess reaches the motors just as fast as a right one.
    """
    text = (utterance or "").strip()[:MAX_UTTERANCE_CHARS]
    command = Command(utterance=utterance or "", source="rules")
    if not text:
        return command

    # STOP short-circuits the entire parse. If the word appears anywhere in the
    # sentence the answer is stop, and nothing else in the sentence matters.
    # "go to the kitchen, no wait, stop" must not drive to the kitchen.
    if _STOP.search(text):
        command.steps = [Step(Verb.STOP)]
        command.notes.append("stop short-circuits any other clause")
        return command

    for clause in _SPLIT.split(text):
        clause = clause.strip()
        if not clause:
            continue

        timed = _WAIT.search(clause)
        if timed:
            amount = float(timed.group(1))
            unit = timed.group(2).lower()
            seconds = amount * 60 if unit.startswith("m") and unit != "s" else amount
            command.steps.append(Step(Verb.WAIT, seconds=seconds))
            continue

        if _HOME.search(clause):
            command.steps.append(Step(Verb.RETURN_HOME))
            continue

        if _REPORT.search(clause):
            command.steps.append(Step(Verb.REPORT))
            continue

        if _PATROL.search(clause):
            remainder = _PATROL.sub("", clause, count=1)
            targets = [clean_target(part) for part in _LIST_SPLIT.split(remainder)]
            targets = [t for t in targets if t]
            for target in targets:
                command.steps.append(Step(Verb.PATROL, target=target))
            if not targets:
                command.notes.append("patrol named no locations")
            continue

        match = _GOTO.search(clause)
        if match:
            target = clean_target(clause[match.end():])
            if target:
                command.steps.append(Step(Verb.GO_TO, target=target))
                continue

        if _BARE_WAIT.search(clause):
            command.steps.append(Step(Verb.WAIT, seconds=5.0))
            continue

        # A bare landmark name after a "then" is a goto: "go to the kitchen
        # then the printer". Only valid as a continuation — a bare noun on its
        # own is too ambiguous to act on.
        if command.steps and len(clause.split()) <= 4:
            target = clean_target(clause)
            if target:
                command.steps.append(Step(Verb.GO_TO, target=target))
                continue

        command.notes.append(f"did not understand: {clause!r}")

    return command


# ---------------------------------------------------------------------------
# Model-backed parsing
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You convert instructions for an indoor mobile robot into JSON.

Reply with ONLY a JSON object, no prose and no code fence:
{"steps": [{"verb": "...", "target": "...", "seconds": 0}]}

Allowed verbs: go_to, patrol, return_home, stop, wait, report.
- go_to and patrol require "target" — the name of a place.
- wait takes "seconds".
- return_home, stop and report take neither.

"target" MUST be one of the known place names listed below, copied exactly.
If the instruction refers to a place that is not in the list, return
{"steps": []}. Do not invent a place. Do not output coordinates.

Known places:
{places}
"""


class Grounder:
    """Rules first, model second, refusal third."""

    def __init__(self, semantic_map, llm=None, use_llm: bool = True):
        self.semantic_map = semantic_map
        self.llm = llm
        self.use_llm = use_llm

    def ground(self, utterance: str) -> Command:
        command = parse_rules(utterance)
        if command.steps:
            return command

        if not (self.use_llm and self.llm is not None):
            return command

        try:
            steps = self.llm.parse(
                utterance[:MAX_UTTERANCE_CHARS],
                system=SYSTEM_PROMPT.replace(
                    "{places}",
                    "\n".join(f"- {n}" for n in sorted(self.semantic_map.names())),
                ),
            )
        except Exception as error:                       # noqa: BLE001
            # A model failure must degrade to "I did not understand", never to
            # an unhandled exception in a node that owns the robot's motion.
            command.notes.append(f"model unavailable: {type(error).__name__}")
            return command

        return from_json(steps, utterance)


def from_json(payload, utterance: str = "", source: str = "llm") -> Command:
    """Convert model JSON into a Command, discarding anything malformed.

    Every field is treated as hostile. The model is a text generator that was
    asked nicely for a schema, not an API that guarantees one, and this function
    is the boundary where that distinction is enforced.
    """
    command = Command(utterance=utterance, source=source)
    if not isinstance(payload, dict):
        command.notes.append("model did not return an object")
        return command

    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list):
        command.notes.append("model returned no step list")
        return command

    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            command.notes.append(f"step {index}: not an object")
            continue

        verb_text = str(raw.get("verb", "")).strip().lower()
        try:
            verb = Verb(verb_text)
        except ValueError:
            # An invented verb is dropped here rather than reaching validation.
            # This is the allowlist doing its job.
            command.notes.append(f"step {index}: unknown verb {verb_text!r}")
            continue

        target = raw.get("target")
        target = str(target).strip() if isinstance(target, (str, int, float)) else None

        seconds: Optional[float] = None
        if raw.get("seconds") is not None:
            try:
                seconds = float(raw["seconds"])
            except (TypeError, ValueError):
                command.notes.append(f"step {index}: non-numeric seconds")

        command.steps.append(Step(verb, target=target or None, seconds=seconds))

    return command


def parse_json_text(text: str):
    """Pull a JSON object out of whatever the model actually sent.

    Models wrap JSON in code fences, prefix it with "Sure!", and occasionally
    append an explanation. Insisting on clean output and erroring otherwise
    means a working robot fails on a stylistic whim of the model, so this is
    forgiving about packaging and strict about content.
    """
    import json

    if not text:
        raise ValueError("empty response")

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object in response")

    return json.loads(text[start:end + 1])
