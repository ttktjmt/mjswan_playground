"""The GitHub Pages site: the tasks' builds merged into one mjswan app."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from mjswan_playground.registry import ALL_PARTS, ALL_TASKS

README = Path(__file__).resolve().parents[2] / "README.md"
#: GitHub Pages refuses a published site larger than 1 GB.
SIZE_LIMIT = 1_000_000_000
SIZE_WARNING = 800_000_000
#: Tasks the site leaves out, and why. They still build and run on their own.
NOT_ON_SITE = {
    "musclemimic": "its walking clip comes from a gated Hugging Face dataset, and the "
    "deploy workflow has no Hub login",
    "msm": "its clips are AMASS motions, whose license forbids redistribution, and "
    "MyoFullBody's arm and hand are non-commercial: its author publishes it",
}

_ROW = re.compile(
    r"^\| \[`([a-z][a-z0-9_]*)`\]\(src/mjswan_playground/\1/README\.md\) \|", re.M
)


class SiteError(Exception):
    """The site cannot be made from the README or the builds."""


def task_order(readme: Path = README) -> list[str]:
    """The site's builds in README's order, each a task or the part `site_build` picks.
    Every registry task needs exactly one row."""
    if not readme.exists():
        raise SiteError(
            f"{readme} is missing: the site takes its task order from the repository's "
            "README, so build it from a checkout."
        )
    ids = _ROW.findall(readme.read_text())
    problems = []
    if missing := [task for task in ALL_TASKS if task not in ids]:
        problems.append(f"no row for {', '.join(missing)}")
    if unknown := [task for task in ids if task not in ALL_TASKS]:
        problems.append(f"a row for {', '.join(unknown)}, which the registry lacks")
    if repeated := sorted({task for task in ids if ids.count(task) > 1}):
        problems.append(f"more than one row for {', '.join(repeated)}")
    if problems:
        raise SiteError(
            f"README's Tasks table sets the site's order, and it has {'; '.join(problems)}."
        )
    return [site_build(task) for task in ids if task not in NOT_ON_SITE]


def site_build(task: str) -> str:
    """What the site carries of ``task``: its ``-open`` part, the scenes no non-commercial
    license covers, if it has one, as mjswan Cloud does; else the whole."""
    return f"{task}-open" if f"{task}-open" in ALL_PARTS else task


def merge(builds: list[Path], output_dir: Path) -> list[dict]:
    """Write the projects of ``builds`` side by side under one manifest at
    ``output_dir``, the first project flagged to open the app. Returns the projects."""
    header: dict | None = None
    projects: list[tuple[Path, dict]] = []
    for build in builds:
        path = build / "manifest.json"
        if not path.exists():
            raise SiteError(f"{build} has no manifest.json: build that task first.")
        manifest = json.loads(path.read_text())
        if manifest.get("uses_custom_js"):
            raise SiteError(
                f"{build} was built with custom-TS terms, whose plugins.js is one per app "
                "and cannot sit beside another build's."
            )
        rest = {key: value for key, value in manifest.items() if key != "projects"}
        if header is None:
            header, first = rest, build
        elif rest != header:
            keys = sorted(
                k for k in rest.keys() | header.keys() if rest.get(k) != header.get(k)
            )
            raise SiteError(
                f"{path} differs from {first / 'manifest.json'} in {', '.join(keys)}: "
                "build every task with the same mjswan."
            )
        for project in manifest["projects"]:
            if any(taken["id"] == project["id"] for _, taken in projects):
                raise SiteError(
                    f"{build} has project {project['id']!r}, which an earlier build has too."
                )
            projects.append((build, project))
    if header is None or not projects:
        raise SiteError("No builds to merge.")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    merged = []
    for index, (build, project) in enumerate(projects):
        shutil.copytree(build / project["id"], output_dir / project["id"])
        # Only the first project may carry `default`, the flag the app opens on.
        entry = {key: value for key, value in project.items() if key != "default"}
        merged.append({**entry, "default": True} if index == 0 else entry)
    (output_dir / "manifest.json").write_text(
        json.dumps({**header, "projects": merged}, indent=2)
    )
    return merged


def install_engine(output_dir: Path, base_path: str) -> None:
    """Lay mjswan's engine over the merged data. The engine bakes in the path it is
    served from, while every path in the data is relative, so only it is rebuilt."""
    from mjswan.build.frontend import TEMPLATE_DIR, ClientBuilder, install_spa

    ClientBuilder(TEMPLATE_DIR).build(base_path=base_path)
    if not install_spa(output_dir):
        raise SiteError(f"mjswan built no engine at {TEMPLATE_DIR / 'dist'}.")


def size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
