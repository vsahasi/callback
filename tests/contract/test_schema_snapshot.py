"""The contract is frozen by a snapshot. Any shape change must be deliberate.

To accept an intended change:

    CALLBACK_UPDATE_SNAPSHOT=1 uv run pytest tests/contract/test_schema_snapshot.py

then bump CONTRACT_VERSION in shared/contract.py and tell Veer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from shared.contract import CONTRACT_VERSION, schema_bundle

SNAPSHOT = Path(__file__).resolve().parents[2] / "schemas" / "contract.schema.json"


def _render(bundle: dict) -> str:
    return json.dumps(bundle, indent=2, sort_keys=True) + "\n"


def test_schema_snapshot_matches() -> None:
    current = _render(schema_bundle())

    if os.environ.get("CALLBACK_UPDATE_SNAPSHOT") == "1":
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(current, encoding="utf-8")
        pytest.skip(f"snapshot rewritten at {SNAPSHOT}")

    assert SNAPSHOT.exists(), (
        f"no contract snapshot at {SNAPSHOT}. Create it with "
        "CALLBACK_UPDATE_SNAPSHOT=1 uv run pytest tests/contract/test_schema_snapshot.py"
    )

    recorded = SNAPSHOT.read_text(encoding="utf-8")
    if recorded != current:
        recorded_obj = json.loads(recorded)
        current_obj = json.loads(current)
        added = sorted(set(current_obj["models"]) - set(recorded_obj["models"]))
        removed = sorted(set(recorded_obj["models"]) - set(current_obj["models"]))
        changed = sorted(
            name
            for name in set(current_obj["models"]) & set(recorded_obj["models"])
            if current_obj["models"][name] != recorded_obj["models"][name]
        )
        enum_changed = sorted(
            name
            for name in set(current_obj["enums"]) | set(recorded_obj["enums"])
            if current_obj["enums"].get(name) != recorded_obj["enums"].get(name)
        )
        pytest.fail(
            "contract schema drifted from the snapshot.\n"
            f"  version recorded: {recorded_obj['contract_version']}\n"
            f"  version current : {current_obj['contract_version']}\n"
            f"  models added    : {added}\n"
            f"  models removed  : {removed}\n"
            f"  models changed  : {changed}\n"
            f"  enums changed   : {enum_changed}\n"
            "If this change is intended, rewrite the snapshot with "
            "CALLBACK_UPDATE_SNAPSHOT=1 and bump CONTRACT_VERSION."
        )


def test_snapshot_records_the_current_contract_version() -> None:
    assert SNAPSHOT.exists(), "run the snapshot test first"
    recorded = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert recorded["contract_version"] == CONTRACT_VERSION, (
        "snapshot contract_version is stale. Bump CONTRACT_VERSION and rewrite "
        "the snapshot together."
    )


def test_bundle_covers_every_endpoint_object() -> None:
    bundle = schema_bundle()
    for required in (
        "EventRequest",
        "EventResponse",
        "FactorsResponse",
        "ErrorResponse",
        "HealthResponse",
    ):
        assert required in bundle["models"], f"{required} missing from the bundle"

    assert bundle["headers"]["scenario"] == "X-Callback-Scenario"
    assert bundle["client_policy"]["on_timeout"] == "deny"
    assert bundle["enums"]["Scenario"] == [
        "real_cfo",
        "cloned_voice",
        "replay",
        "ai_agent",
        "human_impostor",
    ]
