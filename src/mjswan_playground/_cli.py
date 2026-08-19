"""CLI entry point for the playground: list, run and build the tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from mjswan_playground.registry import ALL_TASKS, load

app = typer.Typer(
    name="mjswan-playground",
    help="Browser-ready tasks built on mjswan.",
    no_args_is_help=True,
)

TaskId = Annotated[str, typer.Argument(help=f"One of: {', '.join(ALL_TASKS)}.")]


def _build(task_id: str, output_dir: Optional[Path]):
    """Always an absolute path: ``Builder.build`` resolves a relative one against its
    *caller's* directory, which from here is wherever this package is installed."""
    try:
        builder = load(task_id)
    except KeyError as exc:
        typer.echo(str(exc.args[0]), err=True)
        raise typer.Exit(1) from exc
    path = (output_dir or Path("dist") / task_id).resolve()
    return builder.build(output_dir=path), path


@app.command("list")
def list_cmd() -> None:
    """List the available tasks."""
    for task_id in ALL_TASKS:
        typer.echo(task_id)


@app.command("run")
def run_cmd(
    task_id: TaskId,
    port: Annotated[int, typer.Option(help="HTTP server port.")] = 8080,
    host: Annotated[str, typer.Option(help="HTTP server host.")] = "localhost",
    no_open: Annotated[
        bool, typer.Option("--no-open", help="Do not open the browser automatically.")
    ] = False,
    output_dir: Annotated[
        Optional[Path], typer.Option(help="Where to write the built app.")
    ] = None,
) -> None:
    """Build a task and serve it in the browser."""
    built, _ = _build(task_id, output_dir)
    built.launch(host=host, port=port, open_browser=not no_open)


@app.command("build")
def build_cmd(
    task_id: TaskId,
    output_dir: Annotated[
        Optional[Path], typer.Option(help="Where to write the built app.")
    ] = None,
) -> None:
    """Build a task into a dist directory without launching it."""
    _, path = _build(task_id, output_dir)
    typer.echo(str(path))
