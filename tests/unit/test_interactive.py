"""Tests for InteractiveREPL settings handling."""

from __future__ import annotations

from rich.console import Console

from fundamentallm.cli.interactive import InteractiveREPL


class FakeGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, float]] = []

    def generate(self, prompt: str, max_tokens: int, temperature: float, top_k=None, top_p=None):
        self.calls.append((prompt, max_tokens, temperature))
        return prompt + "x"


def test_handle_settings_updates_values():
    repl = InteractiveREPL(FakeGenerator())
    repl.console = Console(record=True)

    repl._handle_settings("/set temperature=0.5")
    repl._handle_settings("/set max_tokens=10")
    repl._handle_settings("/set top_k=3")
    repl._handle_settings("/set top_p=0.8")

    assert repl.temperature == 0.5
    assert repl.max_tokens == 10
    assert repl.top_k == 3
    assert repl.top_p == 0.8


def test_generate_response_appends_history():
    generator = FakeGenerator()
    repl = InteractiveREPL(generator, max_tokens=2, temperature=1.0)
    repl.console = Console(record=True)

    repl._generate_response("hi")

    assert repl.history
    assert generator.calls[0][0] == "hi"
