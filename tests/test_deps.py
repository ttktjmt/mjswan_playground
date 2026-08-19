"""``ensure_repo`` leaves a checkout already parked on the pinned commit alone, so a
second build needs no network."""

import subprocess

from mjswan_playground import _deps


def _run(args, cwd):
    subprocess.run(args, cwd=cwd, check=True, capture_output=True)


def test_pinned_checkout_is_not_refetched(tmp_path, monkeypatch):
    origin = tmp_path / "origin"
    origin.mkdir()
    _run(["git", "init", "-q", "-b", "main"], origin)
    _run(
        [
            "git",
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "one",
        ],
        origin,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=origin, capture_output=True, text=True
    ).stdout.strip()

    cache = tmp_path / "cache"
    monkeypatch.setattr(_deps, "CACHE_DIR", cache)
    kwargs = dict(
        name="origin", commit=commit, marker=".git", root_env_var="UNSET_ROOT"
    )

    assert _deps.ensure_repo(url=str(origin), **kwargs) == cache / "origin"
    # A url that cannot be fetched: only reachable if the second call skips the fetch.
    assert (
        _deps.ensure_repo(url="file:///nonexistent.git", **kwargs) == cache / "origin"
    )
