"""Interactive REPL for text generation."""

from __future__ import annotations

from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from fundamentallm.generation.generator import TextGenerator


class InteractiveREPL:
    """Simple interactive loop for multi-turn generation."""

    def __init__(
        self,
        generator: TextGenerator,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> None:
        self.generator = generator
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.console = Console()
        self.history: list[tuple[str, str]] = []

    def run(self) -> None:
        banner = Panel.fit(
            "[bold cyan]FundamentaLLM Interactive Mode[/bold cyan]\n"
            "Type 'help' for commands, 'quit' to exit",
            border_style="cyan",
        )
        self.console.print(banner)

        while True:
            try:
                prompt = Prompt.ask("[bold cyan]You[/bold cyan]")
                lowered = prompt.lower()
                if lowered in {"quit", "exit", "q", "/quit", "/exit"}:
                    self.console.print("[yellow]Goodbye![/yellow]")
                    break
                if lowered in {"help", "/help"}:
                    self._show_help()
                    continue
                if lowered in {"history", "/history"}:
                    self._show_history()
                    continue
                if lowered in {"/clear", "clear"}:
                    self.console.clear()
                    continue
                if lowered in {"/status", "status"}:
                    self._show_status()
                    continue
                if prompt.startswith("/set "):
                    self._handle_settings(prompt)
                    continue
                self._generate_response(prompt)
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Interrupted[/yellow]")
                continue
            except Exception as exc:  # pragma: no cover - defensive
                self.console.print(f"[red]Error: {exc}[/red]")

    def _generate_response(self, prompt: str) -> None:
        self.console.print("[yellow]Generating...[/yellow]", end="")
        response = self.generator.generate(
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )
        generated = response[len(prompt) :]
        self.console.print(f"\r[bold green]Model[/bold green]: {generated}")
        self.history.append((prompt, generated))

    def _show_help(self) -> None:
        help_text = (
            "[bold]Commands:[/bold]\n"
            "  quit, exit, q    - Exit interactive mode\n"
            "  /help            - Show this help\n"
            "  /status          - Show current generation settings\n"
            "  /clear           - Clear the screen\n"
            "  history          - Show conversation history\n"
            "  /set param=val   - Set parameters (temperature, max_tokens, top_k, top_p)\n\n"
            "[bold]Examples:[/bold]\n"
            "  /set temperature=0.5\n"
            "  /set max_tokens=500\n"
        )
        self.console.print(Panel(help_text, title="Help", border_style="cyan"))

    def _show_history(self) -> None:
        if not self.history:
            self.console.print("[yellow]No history yet[/yellow]")
            return
        lines = []
        for idx, (prompt, response) in enumerate(self.history, start=1):
            lines.append(f"\n[bold cyan]Turn {idx}[/bold cyan]")
            lines.append(f"[cyan]You:[/cyan] {prompt}")
            lines.append(f"[green]Model:[/green] {response}")
        history_text = "\n".join(lines)
        self.console.print(Panel(history_text, title="History", border_style="cyan"))

    def _handle_settings(self, command: str) -> None:
        parts = command[5:].split("=", maxsplit=1)
        if len(parts) != 2:
            self.console.print("[red]Invalid format. Use: /set param=value[/red]")
            return
        param, value = parts[0].strip(), parts[1].strip()
        try:
            if param == "temperature":
                self.temperature = float(value)
                self.console.print(f"[green]temperature set to {self.temperature}[/green]")
            elif param == "max_tokens":
                self.max_tokens = int(value)
                self.console.print(f"[green]max_tokens set to {self.max_tokens}[/green]")
            elif param == "top_k":
                self.top_k = int(value) if value.lower() != "none" else None
                self.console.print(f"[green]top_k set to {self.top_k}[/green]")
            elif param == "top_p":
                self.top_p = float(value) if value.lower() != "none" else None
                self.console.print(f"[green]top_p set to {self.top_p}[/green]")
            else:
                self.console.print(f"[red]Unknown parameter: {param}[/red]")
        except ValueError as exc:
            self.console.print(f"[red]Invalid value: {exc}[/red]")

    def _show_status(self) -> None:
        status_lines = [
            f"temperature = {self.temperature}",
            f"max_tokens = {self.max_tokens}",
            f"top_k = {self.top_k}",
            f"top_p = {self.top_p}",
        ]
        self.console.print(Panel("\n".join(status_lines), title="Current Settings", border_style="cyan"))
