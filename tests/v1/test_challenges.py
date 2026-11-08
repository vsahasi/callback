"""Voice challenges: formats, lazy timing, supersede, scripts (ruling 2.7, 2.11)."""

from __future__ import annotations

import io
import re

import httpx
import pytest
from callback_shared.contract import ChallengeKind, ChallengeState, Scenario
from fastapi import FastAPI

from detector.v1.challenges import (
    CHALLENGE_TTL_S,
    FREE_ANSWER_QUESTIONS,
    WORDS,
    make_say,
)
from detector.v1.scenarios import SCRIPTS, SPOOF_RISE_AT_S, spoof_at
from tests.v1.conftest import (
    FakeClock,
    base_settings,
    build_app,
    client_for,
    create_session,
    turn_body,
)

DIGITS_RE = re.compile(r"\d \d \d, \d \d \d")
PHRASE_RE = re.compile(r"(\d \d \d, \d \d \d), ([a-z]+), ([a-z]+), ([a-z]+)")


def levenshtein(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


async def issue(client: httpx.AsyncClient, session_id: str, kind: str) -> dict:
    response = await client.post(
        f"/v1/sessions/{session_id}/challenges", json={"kind": kind}
    )
    assert response.status_code == 200, response.text
    return response.json()


async def status_of(client: httpx.AsyncClient, session_id: str, cid: str) -> dict:
    response = await client.get(f"/v1/sessions/{session_id}/challenges/{cid}")
    assert response.status_code == 200, response.text
    return response.json()


async def answer(client: httpx.AsyncClient, session_id: str, clock: FakeClock) -> None:
    response = await client.post(
        f"/v1/sessions/{session_id}/turns", json=turn_body(clock)
    )
    assert response.status_code == 204


# ------------------------------------------------------------- formats


def test_the_word_list_follows_the_rules() -> None:
    assert len(WORDS) >= 64
    assert len(set(WORDS)) == len(WORDS)
    assert all(re.fullmatch(r"[a-z]+", word) for word in WORDS)
    homophones = {"for", "four", "to", "two", "too", "won", "one", "ate", "eight", "oh"}
    assert not homophones & set(WORDS)
    close = [
        (a, b)
        for i, a in enumerate(WORDS)
        for b in WORDS[i + 1 :]
        if levenshtein(a, b) < 2
    ]
    assert close == []


async def test_digits_and_phrase_formats(
    v1: httpx.AsyncClient, clock: FakeClock
) -> None:
    session_id = await create_session(v1)
    digits = await issue(v1, session_id, "digits")
    assert re.fullmatch(r"chl_[0-9a-f]{16}", digits["challenge_id"])
    assert DIGITS_RE.fullmatch(digits["say"])
    assert digits["expires_at"] == clock.ms() + 45_000

    phrase = await issue(v1, session_id, "phrase")
    match = PHRASE_RE.fullmatch(phrase["say"])
    assert match is not None
    words = match.groups()[1:]
    assert len(set(words)) == 3
    assert all(word in WORDS for word in words)

    liveness = await issue(v1, session_id, "liveness")
    assert DIGITS_RE.fullmatch(liveness["say"])


def test_enroll_prompt_formats() -> None:
    for prompt in (1, 2, 3):
        say, expected = make_say(ChallengeKind.ENROLL, enroll_prompt=prompt)
        assert re.fullmatch(r"Please repeat after me: \d \d \d, \d \d \d\.", say)
        assert len(expected) == 6
    for prompt in (4, 5, 6):
        say, expected = make_say(ChallengeKind.ENROLL, enroll_prompt=prompt)
        match = re.fullmatch(
            r"Please repeat after me: ([a-z]+), ([a-z]+), ([a-z]+)\.", say
        )
        assert match is not None
        assert all(word in WORDS for word in match.groups())
        assert expected == match.groups()
    say, expected = make_say(ChallengeKind.ENROLL, enroll_prompt=7)
    assert say in FREE_ANSWER_QUESTIONS
    assert expected == ()


# -------------------------------------------------------------- timing


async def test_pending_until_700ms_after_the_first_user_turn(
    v1: httpx.AsyncClient, clock: FakeClock
) -> None:
    session_id = await create_session(v1)
    issued = await issue(v1, session_id, "phrase")
    cid = issued["challenge_id"]
    assert (await status_of(v1, session_id, cid))["status"] == "pending"
    clock.advance(3)
    assert (await status_of(v1, session_id, cid))["status"] == "pending"
    await answer(v1, session_id, clock)
    answered_at = clock.now
    clock.now = answered_at + 0.699
    assert (await status_of(v1, session_id, cid)) == {
        "status": "pending",
        "phrase_match": None,
        "speaker_score": None,
        "reason": None,
    }
    clock.now = answered_at + 0.7
    assert (await status_of(v1, session_id, cid)) == {
        "status": "passed",
        "phrase_match": True,
        "speaker_score": 0.83,
        "reason": None,
    }


async def test_an_agent_turn_does_not_count(
    v1: httpx.AsyncClient, clock: FakeClock
) -> None:
    session_id = await create_session(v1)
    cid = (await issue(v1, session_id, "digits"))["challenge_id"]
    await v1.post(
        f"/v1/sessions/{session_id}/turns",
        json=turn_body(clock, role="agent", text="please say the digits"),
    )
    clock.advance(5)
    assert (await status_of(v1, session_id, cid))["status"] == "pending"


async def test_a_user_turn_before_issue_does_not_count(
    v1: httpx.AsyncClient, clock: FakeClock
) -> None:
    session_id = await create_session(v1)
    await answer(v1, session_id, clock)
    clock.advance(0.1)
    cid = (await issue(v1, session_id, "digits"))["challenge_id"]
    clock.advance(5)
    assert (await status_of(v1, session_id, cid))["status"] == "pending"


async def test_expired_no_answer_after_45s(
    v1: httpx.AsyncClient, clock: FakeClock
) -> None:
    session_id = await create_session(v1)
    cid = (await issue(v1, session_id, "phrase"))["challenge_id"]
    clock.advance(CHALLENGE_TTL_S - 0.01)
    assert (await status_of(v1, session_id, cid))["status"] == "pending"
    clock.advance(0.01)
    assert (await status_of(v1, session_id, cid)) == {
        "status": "expired",
        "phrase_match": None,
        "speaker_score": None,
        "reason": "no_answer",
    }
    # A late answer does not revive it.
    await answer(v1, session_id, clock)
    clock.advance(1)
    assert (await status_of(v1, session_id, cid))["reason"] == "no_answer"


async def test_a_new_challenge_supersedes_a_pending_one(
    v1: httpx.AsyncClient, clock: FakeClock
) -> None:
    session_id = await create_session(v1)
    first = (await issue(v1, session_id, "phrase"))["challenge_id"]
    clock.advance(1)
    second = (await issue(v1, session_id, "phrase"))["challenge_id"]
    assert (await status_of(v1, session_id, first)) == {
        "status": "expired",
        "phrase_match": None,
        "speaker_score": None,
        "reason": "superseded",
    }
    await answer(v1, session_id, clock)
    clock.advance(0.7)
    assert (await status_of(v1, session_id, second))["status"] == "passed"
    # The superseded one stays superseded.
    assert (await status_of(v1, session_id, first))["reason"] == "superseded"


async def test_a_resolved_challenge_is_not_superseded(
    v1: httpx.AsyncClient, clock: FakeClock
) -> None:
    session_id = await create_session(v1)
    first = (await issue(v1, session_id, "digits"))["challenge_id"]
    await answer(v1, session_id, clock)
    clock.advance(0.7)
    await issue(v1, session_id, "digits")
    assert (await status_of(v1, session_id, first))["status"] == "passed"


async def test_superseded_while_resolving_is_not_refused(
    v1: httpx.AsyncClient, v1_app: FastAPI, clock: FakeClock
) -> None:
    session_id = await create_session(v1)
    first = (await issue(v1, session_id, "digits"))["challenge_id"]
    await answer(v1, session_id, clock)
    clock.advance(0.3)
    unanswered = (await issue(v1, session_id, "digits"))["challenge_id"]
    clock.advance(0.1)
    await issue(v1, session_id, "digits")
    session = v1_app.state.v1.registry.get(session_id)
    assert session.challenges[first].resolve(clock.now).reason == "superseded"
    assert session.challenges[first].refused(clock.now) is False
    assert session.challenges[unanswered].refused(clock.now) is True


# -------------------------------------------------------------- errors


async def test_an_unknown_challenge_is_404(v1: httpx.AsyncClient) -> None:
    session_id = await create_session(v1, room="a")
    other = await create_session(v1, room="b")
    cid = (await issue(v1, other, "digits"))["challenge_id"]
    for missing in ("chl_0000000000000000", cid):
        response = await v1.get(f"/v1/sessions/{session_id}/challenges/{missing}")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "challenge_not_found"


async def test_enroll_without_an_enrollment_is_409(v1: httpx.AsyncClient) -> None:
    session_id = await create_session(v1)
    response = await v1.post(
        f"/v1/sessions/{session_id}/challenges", json={"kind": "enroll"}
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "no_open_enrollment"


async def test_a_locked_session_refuses_new_challenges(
    v1: httpx.AsyncClient, v1_app: FastAPI, clock: FakeClock
) -> None:
    session_id = await create_session(v1)
    cid = (await issue(v1, session_id, "digits"))["challenge_id"]
    v1_app.state.v1.registry.get(session_id).locked_code = "synthetic_speech"
    response = await v1.post(
        f"/v1/sessions/{session_id}/challenges", json={"kind": "phrase"}
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "session_locked"
    # Polling an existing challenge still works.
    assert (await status_of(v1, session_id, cid))["status"] == "pending"


async def test_an_unknown_kind_is_a_422(v1: httpx.AsyncClient) -> None:
    session_id = await create_session(v1)
    response = await v1.post(
        f"/v1/sessions/{session_id}/challenges", json={"kind": "riddle"}
    )
    assert response.status_code == 422


async def test_after_end_challenges_are_409(v1: httpx.AsyncClient) -> None:
    session_id = await create_session(v1)
    cid = (await issue(v1, session_id, "digits"))["challenge_id"]
    await v1.post(f"/v1/sessions/{session_id}/end")
    response = await v1.get(f"/v1/sessions/{session_id}/challenges/{cid}")
    assert response.status_code == 409
    response = await v1.post(
        f"/v1/sessions/{session_id}/challenges", json={"kind": "digits"}
    )
    assert response.status_code == 409


async def test_end_drops_say_and_expected_answers(
    v1: httpx.AsyncClient, v1_app: FastAPI
) -> None:
    session_id = await create_session(v1)
    cid = (await issue(v1, session_id, "phrase"))["challenge_id"]
    await v1.post(f"/v1/sessions/{session_id}/end")
    record = v1_app.state.v1.registry.lookup(session_id).challenges[cid]
    assert record.say is None
    assert record.expected is None


# ------------------------------------------------------------- scripts


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        ("real_cfo", {"status": "passed", "phrase_match": True, "speaker_score": 0.83, "reason": None}),
        ("cloned_voice", {"status": "passed", "phrase_match": True, "speaker_score": 0.78, "reason": None}),
        ("replay", {"status": "failed", "phrase_match": False, "speaker_score": 0.80, "reason": "phrase_mismatch"}),
        ("ai_agent", {"status": "passed", "phrase_match": True, "speaker_score": 0.70, "reason": None}),
        ("human_impostor", {"status": "failed", "phrase_match": True, "speaker_score": 0.29, "reason": "speaker_mismatch"}),
    ],
)  # fmt: skip
@pytest.mark.parametrize("kind", ["digits", "phrase", "liveness"])
async def test_each_script_resolves_as_the_table_says(
    v1: httpx.AsyncClient, clock: FakeClock, scenario: str, kind: str, expected: dict
) -> None:
    session_id = await create_session(v1, scenario=scenario)
    cid = (await issue(v1, session_id, kind))["challenge_id"]
    await answer(v1, session_id, clock)
    clock.advance(0.7)
    assert await status_of(v1, session_id, cid) == expected


def test_the_script_table() -> None:
    rows = {
        s: (x.voiceprint, x.spoof, x.agent, x.text, x.text_flags)
        for s, x in SCRIPTS.items()
    }
    assert rows == {
        Scenario.REAL_CFO: (0.83, 0.04, 0.05, 0.10, ()),
        Scenario.CLONED_VOICE: (0.78, 0.30, 0.10, 0.20, ()),
        Scenario.REPLAY: (0.80, 0.45, 0.10, 0.20, ()),
        Scenario.AI_AGENT: (0.70, 0.85, 0.90, 0.40, ()),
        Scenario.HUMAN_IMPOSTOR: (
            0.29,
            0.05,
            0.05,
            0.80,
            ("urgency", "verification_refusal"),
        ),
    }
    assert all(x.enroll.status is ChallengeState.PASSED for x in SCRIPTS.values())


def test_the_cloned_voice_spoof_rises_at_6s() -> None:
    assert SPOOF_RISE_AT_S == 6.0
    assert spoof_at(Scenario.CLONED_VOICE, 5.999) == 0.30
    assert spoof_at(Scenario.CLONED_VOICE, 6.0) == 0.92
    assert spoof_at(Scenario.REAL_CFO, 600) == 0.04


async def test_real_mode_fails_with_no_audio(clock: FakeClock) -> None:
    app = build_app(base_settings(mock_mode=False), clock, io.StringIO())
    async with client_for(app) as client:
        session_id = await create_session(client)
        cid = (await issue(client, session_id, "phrase"))["challenge_id"]
        await answer(client, session_id, clock)
        clock.advance(0.7)
        assert await status_of(client, session_id, cid) == {
            "status": "failed",
            "phrase_match": None,
            "speaker_score": None,
            "reason": "no_audio",
        }


async def test_digits_and_phrase_carry_the_pending_bind(
    v1: httpx.AsyncClient, v1_app: FastAPI
) -> None:
    session_id = await create_session(v1)
    session = v1_app.state.v1.registry.get(session_id)
    session.pending_bind = "k" * 64
    phrase = (await issue(v1, session_id, "phrase"))["challenge_id"]
    digits = (await issue(v1, session_id, "digits"))["challenge_id"]
    assert session.challenges[phrase].bind == "k" * 64
    assert session.challenges[digits].bind == "k" * 64


async def test_say_never_reaches_the_log(
    v1: httpx.AsyncClient, clock: FakeClock, log_stream: io.StringIO
) -> None:
    session_id = await create_session(v1)
    issued = [await issue(v1, session_id, kind) for kind in ("digits", "phrase")]
    await answer(v1, session_id, clock)
    clock.advance(0.7)
    for item in issued:
        await status_of(v1, session_id, item["challenge_id"])
    text = log_stream.getvalue()
    assert "v1_challenge_issued" in text
    for item in issued:
        say = item["say"]
        assert say not in text
        for group in re.findall(r"\d \d \d", say):
            assert group not in text
        for word in re.findall(r"[a-z]+", say):
            assert f'"{word}"' not in text
