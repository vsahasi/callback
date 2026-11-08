"""detector/v1 serves callback_shared, never the internal shared.contract.

The two packages define different models under the same names (ErrorBody,
Decision, Scenario...). One stray import would put the internal shape on the
agent's wire.
"""

from __future__ import annotations

from pathlib import Path

V1_DIR = Path(__file__).resolve().parents[2] / "detector" / "v1"


def test_v1_package_exists() -> None:
    assert (V1_DIR / "__init__.py").is_file()


def test_no_v1_module_imports_the_internal_contract() -> None:
    offenders = [
        path.name
        for path in sorted(V1_DIR.glob("*.py"))
        if "from shared." in path.read_text(encoding="utf-8")
        or "import shared." in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
