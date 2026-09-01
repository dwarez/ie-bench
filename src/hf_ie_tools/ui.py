from __future__ import annotations

from rich.console import Console

console = Console()


def info(message: str) -> None:
    console.print(f"[bold cyan]›[/] {message}")


def success(message: str) -> None:
    console.print(f"[bold green]✓[/] {message}")


def warning(message: str) -> None:
    console.print(f"[bold yellow]![/] {message}")


def error(message: str) -> None:
    console.print(f"[bold red]error:[/] {message}", highlight=False)
