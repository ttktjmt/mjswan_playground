"""The registry points at modules that exist and expose ``setup_builder``. Import-free on
purpose: importing a task pulls in mjlab and torch, and building one clones a repo."""

import ast
import importlib.util

import pytest

from mjswan_playground.registry import _PARTS, _TASKS, ALL_PARTS, ALL_TASKS, load


def _module(name: str) -> ast.Module:
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.origin, f"{name} is not importable"
    return ast.parse(open(spec.origin).read())


@pytest.mark.parametrize("task_id", ALL_TASKS)
def test_task_module_defines_setup_builder(task_id: str) -> None:
    tree = _module(_TASKS[task_id])
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "setup_builder" in functions


@pytest.mark.parametrize("part", ALL_PARTS)
def test_part_is_in_its_modules_parts(part: str) -> None:
    (parts,) = [
        node.value
        for node in _module(_PARTS[part]).body
        if isinstance(node, ast.Assign)
        and [getattr(t, "id", None) for t in node.targets] == ["PARTS"]
    ]
    assert part in {key.value for key in parts.keys}


def test_unknown_task_lists_the_known_ones() -> None:
    with pytest.raises(KeyError, match="husky"):
        load("nope")
