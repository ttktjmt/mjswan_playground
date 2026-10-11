"""CLI entry point for the playground: list, run and build the tasks, and the site."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

from mjswan_playground import _site
from mjswan_playground.registry import ALL_PARTS, ALL_TASKS, load

app = typer.Typer(
    name="mjswan-playground",
    help="Browser-ready tasks built on mjswan.",
    no_args_is_help=True,
)

TaskId = Annotated[
    str,
    typer.Argument(
        help=f"One of: {', '.join(ALL_TASKS)}; or a part to upload alone: "
        f"{', '.join(ALL_PARTS)}."
    ),
]


def _build(task_id: str, output_dir: Optional[Path]):
    """Always an absolute path: ``Builder.build`` resolves a relative one against its
    *caller's* directory, which from here is wherever this package is installed."""
    try:
        builder = load(task_id)
    except KeyError as exc:
        typer.echo(str(exc.args[0]), err=True)
        raise typer.Exit(1) from exc
    path = (output_dir or Path("dist") / task_id).resolve()
    built = builder.build(output_dir=path)
    problems = _unservable_command_slots(
        json.loads((path / "manifest.json").read_text())
    )
    if problems:
        typer.echo(
            "These graph inputs read a command field the browser does not serve, so "
            "their graphs would never run:",
            err=True,
        )
        for problem in problems:
            typer.echo(f"  {problem}", err=True)
        typer.echo(
            "Read a command with get_command(name) or a field it serves; a command the "
            "MDP lacks goes in the policy's commands=.",
            err=True,
        )
        raise typer.Exit(1)
    return built, path


#: Fields mjswan's own command classes serve in the browser (their ``getStateField``).
_NATIVE_COMMAND_FIELDS = {
    "UiCommand": ("command",),
    "TrackingCommand": (
        "command",
        "is_ready",
        "ref_root_pos_w",
        "ref_root_quat_w",
        "ref_joint_pos",
        "anchor_pos_w",
        "anchor_quat_w",
        "anchor_lin_vel_w",
        "anchor_ang_vel_w",
        "ref_base_height",
        "ref_base_lin_vel_b",
        "ref_base_ang_vel_b",
        "ref_gravity_b",
        "joint_pos",
        "tracked_joint_pos",
        "body_pos_w",
        "robot_anchor_pos_w",
        "robot_anchor_quat_w",
        "robot_body_pos_w",
        "body_pos_relative_w",
    ),
}


def _unservable_command_slots(manifest: dict) -> list[str]:
    """``{command, field}`` slots the browser cannot serve; parity misses them."""
    problems = []
    for project in manifest.get("projects", []):
        for scene in project.get("scenes", []):
            for mdp in scene.get("mdps", []):
                commands = mdp.get("commands") or {}
                for where, slot in _command_slots(mdp):
                    read = (
                        f"{scene.get('id')}/{mdp.get('id')}{where}: "
                        f"{slot['command']}.{slot['field']}"
                    )
                    command = commands.get(slot["command"])
                    if command is None:
                        problems.append(f"{read} (no such command in this MDP)")
                        continue
                    if command.get("name") == "OnnxCommand":
                        # `command` is what get_command() reads.
                        served = ["command"] + [
                            field["name"] for field in command.get("state_fields", [])
                        ]
                    else:
                        served = _NATIVE_COMMAND_FIELDS.get(command.get("name"))
                    if served is not None and slot["field"] not in served:
                        problems.append(f"{read} (serves {', '.join(served)})")
    return problems


def _command_slots(node, where: str = ""):
    if isinstance(node, dict):
        if "command" in node and "field" in node:
            yield where, node
        for key, value in node.items():
            yield from _command_slots(value, f"{where}/{key}")
    elif isinstance(node, list):
        for value in node:
            yield from _command_slots(value, where)


@app.command("list")
def list_cmd() -> None:
    """List the available tasks, then the parts of them built to upload alone."""
    for task_id in ALL_TASKS + ALL_PARTS:
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


@app.command("site")
def site_cmd(
    task_ids: Annotated[
        Optional[list[str]],
        typer.Argument(
            metavar="TASK_ID...",
            help="Only these tasks. Default: every task on the site.",
            show_default=False,
        ),
    ] = None,
    no_build: Annotated[
        bool,
        typer.Option("--no-build", help="Merge the builds already in --dist-dir."),
    ] = False,
    base_path: Annotated[
        str, typer.Option(help="URL path the site is served from.")
    ] = "/",
    dist_dir: Annotated[
        Path, typer.Option(help="Where each task's build is, as <dist-dir>/<task-id>.")
    ] = Path("dist"),
    output_dir: Annotated[
        Optional[Path],
        typer.Option(help="Where to write the site. Default: <dist-dir>/_site."),
    ] = None,
) -> None:
    """Build the GitHub Pages site: every task on it, merged into one app."""
    if not (base_path.startswith("/") and base_path.endswith("/")):
        raise typer.BadParameter(
            "starts and ends with /, e.g. /mjswan_playground/", param_hint="--base-path"
        )
    try:
        order = _site.task_order()
    except _site.SiteError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    if task_ids:
        if left_out := [task for task in task_ids if task in _site.NOT_ON_SITE]:
            for task in left_out:
                typer.echo(
                    f"{task} is not on the site: {_site.NOT_ON_SITE[task]}.", err=True
                )
            raise typer.Exit(1)
        if unknown := [task for task in task_ids if task not in order]:
            typer.echo(
                f"Unknown task {', '.join(unknown)}. Available: {', '.join(order)}",
                err=True,
            )
            raise typer.Exit(1)
        order = [task for task in order if task in task_ids]

    dist_dir = dist_dir.resolve()
    if not no_build:
        for task_id in order:
            # A process per task: pacman and bipedhrl both import a top-level `src`.
            command = [sys.executable, "-m", "mjswan_playground", "build", task_id]
            command += ["--output-dir", str(dist_dir / task_id)]
            if subprocess.run(command).returncode:
                typer.echo(
                    f"Building {task_id} failed, so the site was not made.", err=True
                )
                raise typer.Exit(1)

    site = (output_dir or dist_dir / "_site").resolve()
    try:
        projects = _site.merge([dist_dir / task_id for task_id in order], site)
        _site.install_engine(site, base_path)
    except _site.SiteError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    for project in projects:
        megabytes = _site.size(site / project["id"]) / 1e6
        typer.echo(f"{project['id']:<28}{megabytes:8.1f} MB")
    total = _site.size(site)
    typer.echo(f"{'site, engine included':<28}{total / 1e6:8.1f} MB")
    if total > _site.SIZE_LIMIT:
        typer.echo("GitHub Pages serves at most 1 GB.", err=True)
        raise typer.Exit(1)
    if total > _site.SIZE_WARNING:
        typer.echo("Warning: the site is nearing GitHub Pages' 1 GB.", err=True)
    typer.echo(str(site))
