"""The registry points at modules that exist and expose ``setup_builder``. Import-free on
purpose: importing a task pulls in mjlab and torch, and building one clones a repo."""

import ast
import importlib.util

import pytest

from mjswan_playground.registry import _TASKS, ALL_TASKS, load


@pytest.mark.parametrize("task_id", ALL_TASKS)
def test_task_module_defines_setup_builder(task_id: str) -> None:
    spec = importlib.util.find_spec(_TASKS[task_id])
    assert spec is not None and spec.origin, f"{_TASKS[task_id]} is not importable"

    tree = ast.parse(open(spec.origin).read())
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "setup_builder" in functions


def test_unknown_task_lists_the_known_ones() -> None:
    with pytest.raises(KeyError, match="husky-skater"):
        load("nope")
