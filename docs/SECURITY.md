# Securing a robot that takes instructions from a language model

This document explains what is defended here, why it is defended that way, and
what is deliberately *not* claimed. It is written to be read by someone who has
not seen the code.

The short version: **an LLM with a physical actuator is a different problem from
an LLM with a text box.** A hallucinated answer is embarrassing. A hallucinated
*goal pose* is a 6 kg machine driving somewhere nobody asked it to go.

---

## 1. The threat model

Assume the language model is compromised. Not "might be slightly wrong" —
assume an attacker controls its output completely.

That is not paranoia; it is the realistic case. Anything that gets text into
the command channel can steer the model:

- a chat or ticketing integration that relays user messages
- a filename, a document title, a calendar event name
- **a printed sign in the building that a camera reads** — the injection vector
  unique to robots, and the one people forget

The design question is therefore not "how do we stop the model being fooled".
It is: **when the model is fooled, what can it actually do?**

---

## 2. The core defence: a positive-security model

An **allowlist of what is permitted**, never a denylist of what is forbidden.

A denylist for a robot — "don't drive into people, don't leave the building,
don't go down stairs" — fails the first time someone phrases a request the list
did not anticipate. And you cannot enumerate the bad requests, because natural
language is infinite.

An allowlist inverts the burden. Here it has two halves:

### 2a. The verb allowlist

`semantic_nav/commands.py`:

```python
class Verb(str, Enum):
    GO_TO = "go_to"
    PATROL = "patrol"
    RETURN_HOME = "return_home"
    STOP = "stop"
    WAIT = "wait"
    REPORT = "report"
```

Six actions. Nothing else exists. The conversion happens in exactly one place:

```python
try:
    verb = Verb(verb_text)
except ValueError:
    command.notes.append(f"step {index}: unknown verb {verb_text!r}")
    continue          # dropped, not executed
```

So an attacker who fully controls the model and gets it to emit

```json
{"steps": [{"verb": "disable_safety"},
           {"verb": "set_max_speed", "target": "99"},
           {"verb": "drive_to_coordinates", "target": "0,0"}]}
```

produces **zero steps**. Not an error, not a partial execution — the enum
conversion fails three times and there is nothing left. There is no code
anywhere in the repository that could carry those instructions out, so the
question of whether to allow them never arises.

> `test_an_injection_that_the_model_obeys_still_cannot_execute` asserts exactly
> this.

### 2b. The destination allowlist

`Step` has a `target` field that holds a **name**, and no `x`/`y` fields at all.
The model may say "kitchen". It may not say "(4.2, -1.8)".

Why this matters: a coordinate is unbounded. A name is drawn from a set of six
strings that a human wrote into a YAML file. The `SemanticMap` — not the model —
decides where "kitchen" is, and it can only answer with a pose from that file.

```python
match = semantic_map.resolve(step.target)
if match.landmark is None:
    raise ValidationError(f"I don't know a place called '{step.target}'. "
                          f"I know: {known}")
```

A model that hallucinates "the server room" gets a refusal, not a guess.

**This is the same argument as an SQL allowlist**: parse the request into a
structure, check every referenced entity against a list of permitted ones, and
refuse rather than sanitise. Sanitising means guessing what the input *meant*.
Refusing does not.

---

## 3. The single entry point

An allowlist only works if it cannot be bypassed. That is enforced by types:

```python
def validate(command: Command, semantic_map) -> ValidatedCommand: ...

class MissionRunner:
    def run(self, command: ValidatedCommand) -> MissionOutcome: ...
```

`ValidatedCommand` is produced by exactly one function. `MissionRunner.run`
accepts nothing else. There is no path from raw model output to a motion goal
that skips validation, and that property is checked by the compiler-adjacent
part of your brain every time you read the signature — rather than by a comment
asking people to be careful.

**Lesson worth generalising:** when a security check must always run, make the
*output of the check* the input type of the dangerous operation. Then bypassing
it requires constructing a type, which is a conspicuous thing to do and shows up
in review.

---

## 4. Resource limits — denial of service against a physical machine

Text-only DoS wastes tokens. Robot DoS occupies a corridor.

| Limit | Value | Without it |
|---|---|---|
| `MAX_STEPS` | 8 | "patrol the kitchen, the lobby" ×400 keeps the robot busy all day |
| `MAX_WAIT_S` | 300 | `wait 9999999` parks it in a doorway for eleven hours |
| `MAX_UTTERANCE_CHARS` | 400 | a pasted document becomes a prompt, and a bill |

Each of these is a one-line check and each has a test. They are cheap and they
close a category.

---

## 5. Defence in depth: the safety governor

Everything above is about the *command*. The governor is about the *motion*,
and it assumes everything above it has already failed.

```
Nav2 / teleop / anything ──► /cmd_vel_raw ──► [governor] ──► /cmd_vel ──► base
```

Three properties make it a control rather than a suggestion:

1. **It is the last writer.** The Gazebo diff-drive plugin subscribes to
   `/cmd_vel_raw`. Only the governor publishes `/cmd_vel`. Enforced in the
   topology, not in documentation — and `test_the_base_does_not_subscribe_to_cmd_vel_directly`
   fails if the remapping is ever removed.

2. **It fails closed.** No scan yet → stop. Scan older than 0.5 s → stop.
   E-stop → stop, and the e-stop does not clear itself. "We do not know what is
   in front of us" resolves to *stop*, never to *proceed*.

3. **It is independent of the AI stack.** It reads raw lidar ranges. It has no
   idea an LLM exists. A total compromise of the language layer does not reach
   it.

### The ordering bug

Worth its own section because both versions look correct in review:

```python
# WRONG - scale then clamp
linear = desired * obstacle_scale     # 50.0 * 0.5 = 25.0
linear = min(linear, MAX)             # 25.0 -> 0.45   full speed at a wall

# RIGHT - clamp then scale
linear = min(desired, MAX)            # 50.0 -> 0.45
linear = linear * obstacle_scale      # 0.45 * 0.5 = 0.225
```

With the wrong ordering, obstacle scaling has **no effect at all** on any
command large enough to be clamped. The safety feature is present, is tested for
"does it scale", and does nothing when it matters.

**Lesson:** a safety check that runs is not the same as a safety check that
binds. Test the interaction between limits, not just each limit alone.

---

## 6. Secrets

- The API key goes in an `x-goog-api-key` **header**, never `?key=` in the URL.
  URLs end up in exception messages, proxy logs, browser history and CI output.
  Headers do not.
- On failure the original exception is **deliberately discarded**:

  ```python
  except Exception as error:
      raise LLMUnavailable(f"{type(error).__name__} calling the model") from None
  ```

  urllib exceptions can carry the `Request` object, and the `Request` carries
  the key. `from None` breaks the chain so a stack trace in a log cannot leak it.
- `read_api_key` strips whitespace **and a UTF-8 BOM**. A key saved from a
  Windows editor carries one; it survives into the header and produces
  `'latin-1' codec can't encode character '﻿'`, an error that names the
  encoding and not the cause.
- No key is ever written to a committed file. `.env` is gitignored;
  `.env.example` documents the shape only.

---

## 7. What is *not* claimed

Being precise about this is part of the security argument, not a caveat to it.

- **The model can still be fooled.** Nothing here prevents that. The claim is
  that a fooled model has nothing useful to say.
- **A compromised ROS graph wins.** ROS 2 topics are unauthenticated by default.
  Anyone who can publish to `/cmd_vel` on the same DDS domain bypasses the
  governor entirely. The real answer is SROS2 with DDS security enabled, and it
  is not configured here — this is a portfolio project, not a deployment.
- **Physical safety is not software safety.** A real robot needs a hardware
  emergency stop wired to the motor controller. `engage_estop()` is a software
  latch; it stops commands, not electricity.
- **The map is trusted.** If someone edits `semantic_map.yaml` to put "the
  kitchen" at the top of a staircase, the robot will drive there. Map integrity
  is an access-control problem, not a validation one.
- **No authentication on `/nl_command`.** Anyone who can publish to the topic
  can command the robot within the allowlist. That is a deployment concern and
  a genuine gap.

---

## 8. The transferable checklist

For any system where a model output causes a real-world effect:

1. **Enumerate the verbs.** If the action is not on the list, no code exists to
   perform it.
2. **Never let the model produce a raw identifier.** Names resolved against a
   registry; not coordinates, not IDs, not paths, not SQL.
3. **Make validation a type, not a step.** The dangerous function should only
   accept the validated form.
4. **Bound everything countable.** Steps, durations, input length, retries.
5. **Refuse rather than round up.** A weak match is a question, not a guess.
6. **Detect ambiguity separately from error.** "Did you mean A or B?" is a
   better answer than either A or B.
7. **Put a dumb, independent check at the bottom.** It must not depend on the
   model, and it must fail closed.
8. **Test the attacks.** A security property with no test is an intention.
9. **Say what you have not defended.** An honest gap list is worth more than a
   confident claim that does not survive contact.
