"""The Modal deployment serves requests concurrently (integration review M5).

Read statically, because the dev environment has no modal package. With
max_containers=1 and Modal's default of one input per container, every /v1
request (turn posts, challenge polls, authorize, the review dashboard) would
queue behind the last, and the 2.2 s deadline does not count queue time.
"""

from __future__ import annotations

import ast
from pathlib import Path

MODAL_APP = Path(__file__).resolve().parents[2] / "detector" / "modal_app.py"


def _entrypoint() -> ast.FunctionDef:
    tree = ast.parse(MODAL_APP.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "fastapi_app"
    )


def _name(node: ast.expr) -> str:
    target = node.func if isinstance(node, ast.Call) else node
    return ast.unparse(target)


def test_the_entrypoint_takes_32_concurrent_inputs() -> None:
    decorators = _entrypoint().decorator_list
    names = [_name(d) for d in decorators]
    assert "modal.concurrent" in names
    concurrent = decorators[names.index("modal.concurrent")]
    assert isinstance(concurrent, ast.Call)
    assert {k.arg: ast.literal_eval(k.value) for k in concurrent.keywords} == {
        "max_inputs": 32
    }
    # Modal's order: app.function outermost, then concurrent, then asgi_app.
    assert names == ["app.function", "modal.concurrent", "modal.asgi_app"]


def test_still_pinned_to_one_container() -> None:
    function = _entrypoint().decorator_list[0]
    assert isinstance(function, ast.Call)
    pins = {
        k.arg: ast.literal_eval(k.value)
        for k in function.keywords
        if k.arg in {"min_containers", "max_containers"}
    }
    assert pins == {"min_containers": 1, "max_containers": 1}
