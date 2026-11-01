"""The build refuses a graph input the browser cannot serve."""

import importlib.util
import json
import re
from pathlib import Path

from typer.testing import CliRunner

from mjswan_playground import _cli


def _manifest(slots: list[dict], motions: tuple[list[str], ...] = ()) -> dict:
    """One MDP reading `slots`, and a policy on it with a motion per `frame_fields`."""
    commands = {
        "clock": {
            "name": "OnnxCommand",
            "state_fields": [{"name": "step_count"}],
            "command_field": "step_count",
        },
        "motion": {"name": "TrackingCommand"},
        "pad": {"name": "UiCommand"},
    }
    mdp = {
        "id": "a0",
        "commands": commands,
        "observations": {"actor": {"input_slots": slots}},
    }
    policy = {"mdp": "a0", "motions": [{"frame_fields": f} for f in motions]}
    return {
        "projects": [{"scenes": [{"id": "s", "mdps": [mdp], "policies": [policy]}]}]
    }


def _refused(*slots: tuple[str, str], motions: tuple[list[str], ...] = ()) -> list[str]:
    manifest = _manifest([{"command": c, "field": f} for c, f in slots], motions)
    return _cli._unservable_command_slots(manifest)


def test_served_slots_pass() -> None:
    assert not _refused(
        ("clock", "step_count"),
        ("clock", "command"),
        ("motion", "anchor_quat_w"),
        ("motion", "command"),
        ("pad", "command"),
    )


def test_unserved_slots_are_refused() -> None:
    for slot in [
        ("clock", "x"),
        ("motion", "x"),
        ("pad", "x"),
        ("gone", "x"),
    ]:
        assert len(_refused(slot)) == 1, slot


def test_frame_fields_every_motion_lists_are_served() -> None:
    motions = (["latent"], ["latent", "phase"])
    assert not _refused(("motion", "latent"), motions=motions)
    assert len(_refused(("motion", "phase"), motions=motions)) == 1
    assert len(_refused(("clock", "latent"), motions=motions)) == 1


def _get_state_field(command: str) -> str:
    """The installed mjswan's browser source for `<command>.getStateField`."""
    package = Path(importlib.util.find_spec("mjswan").submodule_search_locations[0])
    source = (package / f"template/src/core/command/{command}.ts").read_text()
    return source.split("getStateField(field: string)", 1)[1].split("\n  }\n", 1)[0]


def test_native_fields_match_the_installed_mjswan() -> None:
    served = set(re.findall(r"case '(\w+)'", _get_state_field("TrackingCommand")))
    assert served == set(_cli._NATIVE_COMMAND_FIELDS["TrackingCommand"])
    assert "field === 'command'" in _get_state_field("OnnxCommand")


def test_build_exits_on_an_unservable_slot(tmp_path, monkeypatch) -> None:
    class Builder:
        def build(self, output_dir: Path) -> None:
            output_dir.mkdir(parents=True)
            manifest = _manifest([{"command": "clock", "field": "phase"}])
            (output_dir / "manifest.json").write_text(json.dumps(manifest))

    monkeypatch.setattr(_cli, "load", lambda task_id: Builder())
    result = CliRunner().invoke(
        _cli.app, ["build", "x", "--output-dir", str(tmp_path / "d")]
    )
    assert result.exit_code == 1
    assert "clock.phase" in result.output
