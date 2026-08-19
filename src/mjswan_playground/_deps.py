"""Pinned checkouts of the repositories the tasks pull their assets from. They land in
``.cache/`` at the repo root, so a checkout stays self-contained."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
# ponytail: installed copies have no pyproject.toml next to them and site-packages may
# be read-only, so those fall back to the user cache.
CACHE_DIR = (
    _ROOT / ".cache"
    if (_ROOT / "pyproject.toml").exists()
    else Path.home() / ".cache" / "mjswan_playground"
)


def _git(args: list[str], cwd: Path) -> None:
    env = os.environ.copy()
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    try:
        subprocess.run(["git", *args], cwd=cwd, env=env, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("git is required to fetch the task assets") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to run `git {' '.join(args)}` in {cwd}") from exc


def _head(repo: Path) -> str:
    """The checked-out commit, or "" if git cannot say (a half-finished clone)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
        )
    except FileNotFoundError as exc:
        raise RuntimeError("git is required to fetch the task assets") from exc
    return out.stdout.strip() if out.returncode == 0 else ""


def ensure_repo(
    *, name: str, url: str, commit: str, marker: str, root_env_var: str
) -> Path:
    """Return a checkout of ``url`` at ``commit``, cloning into the cache if needed, or
    whatever ``root_env_var`` points at if ``marker`` is present there."""
    configured = os.getenv(root_env_var)
    if configured:
        root = Path(configured).expanduser().resolve()
        if not (root / marker).exists():
            raise FileNotFoundError(
                f"{root_env_var}={root} does not look like a {name} checkout "
                f"({marker} is missing)."
            )
        return root

    repo = CACHE_DIR / name
    if not (repo / ".git").exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        _git(["clone", url, str(repo)], cwd=repo.parent)
    elif _head(repo) == commit:
        # Already parked on the pinned commit: fetching would only add a network
        # round-trip, and would fail a build that could have run offline.
        return repo
    else:
        _git(["remote", "set-url", "origin", url], cwd=repo)
        _git(["fetch", "--tags", "origin"], cwd=repo)
    # Detached at a pinned commit: a branch name would leave the checkout stale
    # after a fetch, and the assets are what the policy was trained against.
    _git(["checkout", "--detach", commit], cwd=repo)
    return repo
