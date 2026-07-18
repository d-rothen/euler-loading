"""Static checks for constructs evaluated at the Python version floor."""

from __future__ import annotations

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[1] / "euler_loading"


def test_no_runtime_pep604_type_aliases() -> None:
    """PEP 604 unions in assignment expressions fail during Python 3.9 imports."""
    offenders: list[str] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for statement in tree.body:
            if not isinstance(statement, ast.Assign):
                continue
            if any(
                isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr)
                for node in ast.walk(statement.value)
            ):
                offenders.append(f"{path.relative_to(PACKAGE_ROOT)}:{statement.lineno}")

    assert offenders == [], (
        "Runtime PEP 604 aliases are not Python 3.9 compatible; use "
        f"typing.Union/Optional instead: {offenders}"
    )
