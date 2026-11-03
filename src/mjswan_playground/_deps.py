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
    *,
    name: str,
    url: str,
    commit: str,
    marker: str,
    root_env_var: str | None = None,
    paths: tuple[str, ...] = (),
) -> Path:
    """Return a checkout of ``url`` at ``commit``, fetching into the cache if needed, or
    whatever ``root_env_var`` points at if ``marker`` is present there. ``paths`` limits
    the checkout to those directories, for an upstream whose other files are large."""
    configured = os.getenv(root_env_var) if root_env_var else None
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
        repo.mkdir(parents=True, exist_ok=True)
        _git(["init", "-q"], cwd=repo)
        if paths:
            _git(["sparse-checkout", "set", *paths], cwd=repo)
    elif _head(repo) == commit:
        # Already parked on the pinned commit: fetching would only add a network
        # round-trip, and would fail a build that could have run offline.
        return repo
    # --depth 1: some upstreams' history is many times the size of their files.
    # A sparse checkout fetches no blob outside its paths.
    fetch = ["fetch", "--depth", "1", *(["--filter=blob:none"] if paths else [])]
    _git([*fetch, url, commit], cwd=repo)
    _git(["checkout", "--detach", commit], cwd=repo)
    return repo
