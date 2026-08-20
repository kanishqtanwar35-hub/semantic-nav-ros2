"""Optional language-model backend.

Optional is the operative word. The robot navigates, patrols, docks and stops
with this file absent, and every test in the package runs without a key. The
model widens the range of phrasings understood; it is not load-bearing.

Uses `urllib` rather than `requests` on purpose: a ROS node that pulls a third
-party HTTP stack into the same process as the motion controller has added a
dependency to a safety-relevant executable to save four lines.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional

DEFAULT_MODEL = "gemini-flash-latest"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def read_api_key(explicit: Optional[str] = None) -> Optional[str]:
    """Read the key from the argument, then the environment, then a file.

    `.strip("\\ufeff")` is not paranoia. A key saved from a Windows editor
    carries a UTF-8 BOM, which survives into the HTTP header and produces
    `'latin-1' codec can't encode character '\\ufeff'` — an error that names the
    encoding and not the cause, and costs an hour the first time.
    """
    key = explicit or os.environ.get("GEMINI_API_KEY")
    if not key:
        path = os.environ.get("GEMINI_API_KEY_FILE")
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8-sig") as handle:
                key = handle.read()
    if not key:
        return None
    return key.strip().lstrip("﻿").strip() or None


class LLMUnavailable(RuntimeError):
    """No key, no network, or the service refused. Always recoverable."""


class GeminiBackend:
    """Minimal Gemini client returning parsed JSON steps."""

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL,
                 timeout_s: float = 6.0):
        self.api_key = read_api_key(api_key)
        self.model = model
        # Short by design. A robot waiting on an HTTP round trip is a robot not
        # responding to its operator; six seconds is already generous, and the
        # rule parser has already handled anything urgent.
        self.timeout_s = timeout_s

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def parse(self, utterance: str, system: str) -> dict:
        from semantic_nav.grounding import parse_json_text

        if not self.api_key:
            raise LLMUnavailable("no GEMINI_API_KEY set")

        body = json.dumps({
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": utterance}]}],
            "generationConfig": {
                "temperature": 0.0,          # this is parsing, not writing
                "maxOutputTokens": 256,
                "responseMimeType": "application/json",
            },
        }).encode("utf-8")

        request = urllib.request.Request(
            ENDPOINT.format(model=self.model),
            data=body,
            headers={
                "Content-Type": "application/json",
                # The key goes in a header, never in the query string. A URL
                # ends up in exception messages, proxy logs and CI output; a
                # header does not.
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise LLMUnavailable(f"HTTP {error.code} from the model") from None
        except Exception as error:                        # noqa: BLE001
            # Deliberately swallowing the original: urllib exceptions can carry
            # the request object, and the request carries the API key.
            raise LLMUnavailable(f"{type(error).__name__} calling the model") from None

        try:
            text = payload["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            raise LLMUnavailable("model returned no usable candidate") from None

        return parse_json_text(text)


class ScriptedBackend:
    """A backend that replays fixed responses. For tests and for demos.

    Exists so the grounding pipeline — including the malformed-output handling —
    is exercised on every CI run without a key, a network, or a bill.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    @property
    def available(self) -> bool:
        return True

    def parse(self, utterance: str, system: str) -> dict:
        self.calls.append({"utterance": utterance, "system": system})
        if not self.responses:
            raise LLMUnavailable("scripted backend exhausted")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, str):
            from semantic_nav.grounding import parse_json_text
            return parse_json_text(response)
        return response


def default_backend() -> Optional[GeminiBackend]:
    """A backend if a key is configured, otherwise None. Never raises."""
    backend = GeminiBackend()
    return backend if backend.available else None
