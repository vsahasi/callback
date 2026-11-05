#!/usr/bin/env python3
"""Drive all five mock scenarios against a running detector and print a table.

This is the fastest way to see what the detector does. Start the service, run
this, read the table. It is also a usable smoke test: it knows what each
scenario is supposed to do and exits non-zero if reality disagrees, so it can
sit in a CI job or a pre-demo check.

    # terminal 1
    CALLBACK_API_TOKEN=kit-test uv run uvicorn detector.app:app --port 8021

    # terminal 2
    uv run python examples/drive_mock.py --base-url http://127.0.0.1:8021 \
        --token kit-test

Useful flags:

    --wrong-otp          fail every step-up instead of passing it
    --show-fail-closed   also prove the client denies when the detector is gone
    --scenario replay    drive one scenario (repeatable)
    --no-retry           single attempt per event, no bounded retry

Everything here goes through shared/client.py, so what you see is what Veer's
agent will see.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow `python examples/drive_mock.py` from a checkout, not just `uv run`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.client import (
    CallSession,
    DetectorClient,
    Outcome,
    RetryPolicy,
)
from shared.contract import (
    MOCK_OTP_CODE,
    AudioRef,
    Decision,
    Scenario,
    SessionState,
)

WRONG_OTP_CODE = "000000"

# Two different ways for a detector to be absent, because they exercise two
# different halves of the retry policy for real rather than in a mock.
#
#   refused:    nothing is bound to the port, so connect() is refused in
#               microseconds. Cheap failure, so the bounded retry fires.
#   blackholed: 192.0.2.0/24 is reserved for documentation and is not routed,
#               so connect() hangs until the timeout. The first attempt eats
#               the whole latency budget, so there is no retry. Deliberate:
#               out of time means deny now, not deny later.
REFUSED_URL = "http://127.0.0.1:9"
BLACKHOLED_URL = "http://192.0.2.1:9"


@dataclass(frozen=True)
class Expected:
    """What each scenario is supposed to do, so the script can grade itself."""

    decision_after_speech: Decision
    state_after_speech: SessionState
    offers_otp: bool
    decision_with_correct_otp: Decision | None
    decision_with_wrong_otp: Decision | None


EXPECTED: dict[Scenario, Expected] = {
    Scenario.REAL_CFO: Expected(
        decision_after_speech=Decision.ALLOW,
        state_after_speech=SessionState.ALLOWED,
        offers_otp=False,
        decision_with_correct_otp=None,
        decision_with_wrong_otp=None,
    ),
    Scenario.CLONED_VOICE: Expected(
        decision_after_speech=Decision.CHALLENGE,
        state_after_speech=SessionState.CHALLENGING,
        offers_otp=True,
        decision_with_correct_otp=Decision.ALLOW,
        decision_with_wrong_otp=Decision.DENY,
    ),
    Scenario.REPLAY: Expected(
        decision_after_speech=Decision.DENY,
        state_after_speech=SessionState.DENIED,
        offers_otp=False,
        decision_with_correct_otp=None,
        decision_with_wrong_otp=None,
    ),
    Scenario.AI_AGENT: Expected(
        decision_after_speech=Decision.DENY,
        state_after_speech=SessionState.DENIED,
        offers_otp=False,
        decision_with_correct_otp=None,
        decision_with_wrong_otp=None,
    ),
    Scenario.HUMAN_IMPOSTOR: Expected(
        decision_after_speech=Decision.CHALLENGE,
        state_after_speech=SessionState.CHALLENGING,
        offers_otp=True,
        decision_with_correct_otp=Decision.ALLOW,
        decision_with_wrong_otp=Decision.DENY,
    ),
}


@dataclass
class Row:
    """One check's result, ready to print."""

    label: str
    final: Outcome
    after_speech: Outcome
    otp_note: str
    verdict: str
    notes: list[str]


def _audio(start_ms: int) -> AudioRef:
    """A plausible audio reference. Audio is referenced, never inlined."""

    return AudioRef(
        track_id="TR_drive_mock",
        start_ms=start_ms,
        end_ms=start_ms + 3000,
        sample_rate_hz=16000,
    )


async def drive_one(
    client: DetectorClient, scenario: Scenario, *, wrong_otp: bool
) -> Row:
    """Run one scenario start to finish and grade it."""

    expected = EXPECTED[scenario]
    notes: list[str] = []
    call = CallSession(client, scenario=scenario, claimed_user_id=scenario.value)

    # 1. Call connected.
    outcome = await call.start()
    if outcome.failed_closed:
        return _failed_row(scenario, outcome, "start failed")

    # 2. The caller said who they are.
    outcome = await call.claim_identity(scenario.value)
    if outcome.failed_closed:
        return _failed_row(scenario, outcome, "claim failed")

    # 3. Speech, until the detector commits. Two segments is the configured
    #    minimum; this keeps going in case that is tuned up.
    for index in range(4):
        outcome = await call.submit_speech(_audio(1000 + index * 3000))
        if outcome.failed_closed or not outcome.should_continue_listening:
            break

    after_speech = outcome
    if outcome.failed_closed:
        return _failed_row(scenario, outcome, "speech failed")

    if outcome.decision is not expected.decision_after_speech:
        notes.append(
            f"expected {expected.decision_after_speech.value} after speech, "
            f"got {outcome.decision.value}"
        )
    if outcome.state is not expected.state_after_speech:
        notes.append(
            f"expected state {expected.state_after_speech.value}, "
            f"got {outcome.state.value}"
        )

    # 4. Step-up, if one was offered. A hard deny never offers one, and this
    #    is exactly the place a client must not assume `otp` is present.
    otp_note = "not offered"
    if outcome.should_request_otp:
        if not expected.offers_otp:
            notes.append("step-up offered on a scenario that should hard-deny")
        if outcome.otp is None:
            notes.append("next_action is request_otp but otp is null")
        else:
            otp_note = (
                f"{outcome.otp.code_length} digits, "
                f"{outcome.otp.attempts_remaining} attempts"
            )

        if wrong_otp:
            attempts = 0
            # Bounded: three attempts is the policy, so this cannot be a
            # long loop unless something is badly wrong.
            while not call.closed and attempts < 6:
                attempts += 1
                outcome = await call.submit_otp(WRONG_OTP_CODE)
                if outcome.failed_closed:
                    return _failed_row(scenario, outcome, "otp submit failed")
            otp_note = f"{attempts} wrong codes, exhausted"
            wanted = expected.decision_with_wrong_otp
        else:
            outcome = await call.submit_otp(MOCK_OTP_CODE)
            if outcome.failed_closed:
                return _failed_row(scenario, outcome, "otp submit failed")
            otp_note = (
                "correct code accepted" if outcome.is_allow else "correct code rejected"
            )
            wanted = expected.decision_with_correct_otp

        if wanted is not None and outcome.decision is not wanted:
            notes.append(
                f"expected {wanted.value} after step-up, got {outcome.decision.value}"
            )
    else:
        if expected.offers_otp:
            notes.append("expected a step-up challenge, none was offered")
        if outcome.otp is not None:
            notes.append("otp challenge present on a scenario that cannot use one")
        otp_note = "not required" if outcome.is_allow else "not offered (hard deny)"

    # 5. The guard: a finished session must refuse further driving.
    if not call.closed:
        notes.append("session is not closed after a terminal decision")

    # 6. Always close the call out. session_end is the one event a terminal
    #    session still accepts.
    await call.end_call()

    verdict = "ok" if not notes else "MISMATCH"
    return Row(
        label=scenario.value,
        final=outcome,
        after_speech=after_speech,
        otp_note=otp_note,
        verdict=verdict,
        notes=notes,
    )


def _failed_row(scenario: Scenario, outcome: Outcome, where: str) -> Row:
    return Row(
        label=scenario.value,
        final=outcome,
        after_speech=outcome,
        otp_note="n/a",
        verdict="FAILED CLOSED",
        notes=[f"{where}: {outcome.failure_reason}"],
    )


async def demo_fail_closed(label: str, url: str, timeout_ms: int) -> Row:
    """Prove the headline rule against a detector that is not there."""

    async with DetectorClient(
        base_url=url, token="irrelevant", timeout_ms=timeout_ms
    ) as dead:
        call = CallSession(dead)
        outcome = await call.start()

    ok = outcome.is_deny and outcome.failed_closed and outcome.response is None
    notes = [] if ok else ["THE CLIENT DID NOT FAIL CLOSED"]
    print(
        f"  {label:<11} {outcome.attempts} attempt(s) inside the {timeout_ms}ms "
        f"budget: {outcome.failure_reason}"
    )
    return Row(
        label=label,
        final=outcome,
        after_speech=outcome,
        otp_note="n/a",
        verdict="ok" if ok else "MISMATCH",
        notes=notes,
    )


# ------------------------------------------------------------------- printing


def _factor_cell(outcome: Outcome) -> str:
    if outcome.response is None:
        return "no response"
    factors = outcome.response.factors
    if not factors:
        return "none yet"
    # Abbreviated to the first word so the table stays inside a terminal.
    # Run with --detail for the full names, scores and explanations.
    failing = [f.name.value.split("_")[0] for f in factors if f.verdict.value == "fail"]
    if not failing:
        return f"{len(factors)} factors, all pass"
    return f"{len(factors)} factors, fail: {'+'.join(failing)}"


def _risk_cell(outcome: Outcome) -> str:
    if outcome.risk_score is None:
        return "-"
    return f"{outcome.risk_score:.3f}"


def print_table(rows: list[Row]) -> None:
    headers = ("scenario", "decision", "state", "risk", "factors", "otp", "verdict")
    table = [
        (
            row.label,
            row.final.decision.value,
            row.final.state.value,
            _risk_cell(row.final),
            _factor_cell(row.final),
            row.otp_note,
            row.verdict,
        )
        for row in rows
    ]
    widths = [
        max(len(headers[i]), *(len(line[i]) for line in table))
        for i in range(len(headers))
    ]

    def render(cells: tuple[str, ...]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

    print(render(headers))
    print("  ".join("-" * w for w in widths))
    for line in table:
        print(render(line))


def print_detail(rows: list[Row]) -> None:
    print()
    print("factor detail")
    print("=============")
    for row in rows:
        response = row.final.response
        print()
        if response is None:
            print(f"{row.label}  (no response)")
            print(f"  failed closed: {row.final.failure_reason}")
            continue
        print(f"{row.label}  (confidence {response.confidence:.2f})")
        print(f"  reason: {response.reason}")
        print(
            f"  next_action: {response.next_action.value}  "
            f"latency_ms: {response.latency_ms}  "
            f"attempts: {row.final.attempts}"
        )
        for factor in response.factors:
            print(
                f"    {factor.name.value:<18} score={factor.score:.2f} "
                f"weight={factor.weight:.2f} {factor.verdict.value:<13} "
                f"{factor.detail or ''}"
            )


# ----------------------------------------------------------------------- main


async def run(args: argparse.Namespace) -> int:
    retry = RetryPolicy.none() if args.no_retry else RetryPolicy()
    scenarios = args.scenario or list(Scenario)

    async with DetectorClient(
        base_url=args.base_url,
        token=args.token,
        timeout_ms=args.timeout_ms,
        retry=retry,
    ) as client:
        # Two checks, not one. /healthz has no auth, so it cannot tell you
        # anything about the token.
        health = await client.health()
        if health is None:
            print(
                f"detector at {args.base_url} is not answering /healthz.\n"
                "Start it with:\n"
                "  CALLBACK_API_TOKEN=kit-test uv run uvicorn detector.app:app "
                "--port 8021",
                file=sys.stderr,
            )
            return 2

        credentials = await client.verify_credentials()
        if credentials is False:
            print(
                f"detector at {args.base_url} is up but rejected the token.\n"
                "Pass the one the service was started with, via --token or "
                "$CALLBACK_API_TOKEN.",
                file=sys.stderr,
            )
            return 2
        if credentials is None:
            print(
                "could not check the token: the detector answered /healthz but "
                "not /factors. Carrying on, expect denials.",
                file=sys.stderr,
            )

        print(f"detector: {args.base_url}")
        print(
            f"contract: {health.contract_version}   mock_mode: {health.mock_mode}   "
            f"timeout: {args.timeout_ms}ms   "
            f"retry: {retry.max_attempts} attempt(s) inside "
            f"{retry.total_budget_ms}ms"
        )
        if not health.mock_mode:
            print(
                "warning: mock_mode is false, so the scripted scenarios will not "
                "apply and every expectation below will mismatch.",
                file=sys.stderr,
            )
        print(f"otp mode: {'wrong code' if args.wrong_otp else 'correct code'}")
        print()

        rows = [
            await drive_one(client, scenario, wrong_otp=args.wrong_otp)
            for scenario in scenarios
        ]

    print_table(rows)

    if args.show_fail_closed:
        print()
        print("fail-closed check (detector deliberately unreachable)")
        print("=====================================================")
        dead_rows = [
            await demo_fail_closed("refused", REFUSED_URL, args.timeout_ms),
            await demo_fail_closed("blackholed", BLACKHOLED_URL, args.timeout_ms),
        ]
        print()
        print_table(dead_rows)
        rows.extend(dead_rows)

    if args.detail:
        print_detail([r for r in rows if r.final.response is not None])

    print()
    mismatches = [row for row in rows if row.verdict != "ok"]
    noun = "check" if len(rows) == 1 else "checks"
    if not mismatches:
        print(f"all {len(rows)} {noun} behaved as documented.")
        return 0

    print(f"{len(mismatches)} of {len(rows)} {noun} did not behave as documented:")
    for row in mismatches:
        for note in row.notes:
            print(f"  {row.label}: {note}")
    return 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drive the five mock scenarios and print a result table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("CALLBACK_BASE_URL", "http://127.0.0.1:8021"),
        help="Detector base URL. Defaults to $CALLBACK_BASE_URL.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("CALLBACK_API_TOKEN"),
        help="Bearer token. Defaults to $CALLBACK_API_TOKEN.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=800,
        help="Per-attempt client timeout.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        type=Scenario,
        choices=list(Scenario),
        metavar="NAME",
        help="Drive only this scenario. Repeatable.",
    )
    parser.add_argument(
        "--wrong-otp",
        action="store_true",
        help="Fail every step-up instead of passing it.",
    )
    parser.add_argument(
        "--show-fail-closed",
        action="store_true",
        help="Also drive a session against an unreachable detector.",
    )
    parser.add_argument(
        "--no-retry",
        action="store_true",
        help="One attempt per event, no bounded retry.",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Print the full factor breakdown under the table.",
    )
    return parser.parse_args(argv)


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
