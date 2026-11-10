"""The users_on_file directory and routes 1 and 2 (ruling section 2.9)."""

from __future__ import annotations

import dataclasses
import io
import json
from difflib import SequenceMatcher
from pathlib import Path

import httpx
import pytest
from callback_shared.contract import UserOnFile, UserSummary

from detector.config import Settings
from detector.v1.users import (
    AMBIGUITY_MARGIN,
    FUZZY_CUTOFF,
    UserDirectory,
    UsersFileError,
    normalise_name,
)
from tests.v1.conftest import EXAMPLE_USERS, FakeClock, base_settings, build_app

REPO_ROOT = Path(__file__).resolve().parents[2]

TWO_USERS = {
    "users": [
        {
            "user_id": "u_eric",
            "display_name": "Eric Catalano",
            "aliases": ["Eric C"],
            "role": "CFO",
            "phone_e164": "+15555550100",
        },
        {
            "user_id": "u_eric_2",
            "display_name": "Erica Lane",
            "aliases": ["eric catalano"],
            "role": "Controller",
            "phone_e164": "+15555550101",
        },
        {
            "user_id": "u_sam",
            "display_name": "Sam Ortiz",
            "aliases": ["Samuel Ortiz"],
            "role": "AP lead",
            "phone_e164": "+15555550102",
        },
    ]
}


def directory(doc: dict[str, object]) -> UserDirectory:
    return UserDirectory.from_json_text(json.dumps(doc))


# ------------------------------------------------------------ the file


def test_the_example_file_is_the_ruling_example_plus_aliases() -> None:
    text = (REPO_ROOT / "users_on_file.example.json").read_text(encoding="utf-8")
    assert json.loads(text) == EXAMPLE_USERS
    loaded = UserDirectory.from_json_text(text)
    assert len(loaded) == 1
    user = loaded.get("u_eric")
    assert user is not None
    assert user.phone_last_four == "0100"
    # A reserved, fictional 555-01xx number.
    assert user.phone_e164.startswith("+1555555")


def test_the_real_file_is_gitignored() -> None:
    ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "users_on_file.json" in ignored


def test_json_beats_a_path(tmp_path: Path) -> None:
    path = tmp_path / "users.json"
    path.write_text(json.dumps(TWO_USERS), encoding="utf-8")
    settings = base_settings(users_on_file_path=str(path))
    assert len(UserDirectory.from_settings(settings)) == 1  # the inline example
    from_path = dataclasses.replace(settings, users_on_file_json=None)
    assert len(UserDirectory.from_settings(from_path)) == 3


def test_neither_source_is_an_empty_directory_with_a_warning() -> None:
    warnings: list[str] = []
    settings = base_settings(users_on_file_json=None)
    loaded = UserDirectory.from_settings(settings, warn=warnings.append)
    assert len(loaded) == 0
    assert len(warnings) == 1
    assert "USERS_ON_FILE_PATH" in warnings[0]


def test_the_empty_warning_is_printed_by_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    UserDirectory.from_settings(base_settings(users_on_file_json=None))
    assert "WARNING" in capsys.readouterr().out


def test_a_bad_phone_number_raises_at_startup() -> None:
    bad = {"users": [{**EXAMPLE_USERS["users"][0], "phone_e164": "5550100"}]}
    settings = base_settings(users_on_file_json=json.dumps(bad))
    with pytest.raises(UsersFileError) as caught:
        build_app(settings, FakeClock(), io.StringIO())
    message = str(caught.value)
    assert "phone_e164" in message
    assert "5550100" not in message
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    "doc",
    [
        "not json at all",
        json.dumps({"people": []}),
        json.dumps({"users": [{"user_id": "u_x"}]}),
        json.dumps({"users": [EXAMPLE_USERS["users"][0], EXAMPLE_USERS["users"][0]]}),
        json.dumps({"users": [{**EXAMPLE_USERS["users"][0], "phone_e164": "+0123"}]}),
    ],
)
def test_an_invalid_file_raises(doc: str) -> None:
    with pytest.raises(UsersFileError) as caught:
        UserDirectory.from_settings(base_settings(users_on_file_json=doc))
    assert "+15555550100" not in str(caught.value)
    assert "Eric Catalano" not in str(caught.value)


def test_an_unreadable_path_raises(tmp_path: Path) -> None:
    settings = base_settings(
        users_on_file_json=None, users_on_file_path=str(tmp_path / "missing.json")
    )
    with pytest.raises(UsersFileError):
        UserDirectory.from_settings(settings)


def test_unknown_fields_in_the_file_are_ignored() -> None:
    doc = {"users": [{**EXAMPLE_USERS["users"][0], "note": "x"}], "version": 2}
    assert len(directory(doc)) == 1


# ---------------------------------------------------------- name search


@pytest.mark.parametrize(
    ("raw", "normalised"),
    [
        ("Eric Catalano", "eric catalano"),
        ("  eric   CATALANO  ", "eric catalano"),
        ("Eric Catalano.", "eric catalano"),
        ("?Eric Catalano!", "eric catalano"),
        ("Eric\tCatalano,", "eric catalano"),
        ("...", ""),
    ],
)
def test_normalise_name(raw: str, normalised: str) -> None:
    assert normalise_name(raw) == normalised


def test_an_exact_match() -> None:
    found = directory(EXAMPLE_USERS).search("Eric Catalano")
    assert found == [
        UserSummary(
            user_id="u_eric",
            display_name="Eric Catalano",
            role="CFO",
            phone_last_four="0100",
        )
    ]


def test_a_case_and_whitespace_variant() -> None:
    assert [
        u.user_id for u in directory(EXAMPLE_USERS).search(" eric  catalano. ")
    ] == ["u_eric"]


def test_an_alias() -> None:
    assert [u.user_id for u in directory(TWO_USERS).search("samuel ortiz")] == ["u_sam"]


def test_no_match() -> None:
    assert directory(TWO_USERS).search("Sabrina Rao") == []
    assert directory(TWO_USERS).search("") == []
    assert directory(TWO_USERS).search("...") == []


def test_two_matches_come_back_in_file_order() -> None:
    found = directory(TWO_USERS).search("Eric Catalano")
    assert [u.user_id for u in found] == ["u_eric", "u_eric_2"]


# -------------------------------------------- speech-to-text tolerance

# The display name alone, no aliases: the tolerance has to come from the
# matcher, not from the file.
NO_ALIASES = {
    "users": [
        {
            "user_id": "u_eric",
            "display_name": "Eric Catalano",
            "role": "CFO",
            "phone_e164": "+15555550100",
        },
        {
            "user_id": "u_sabrina",
            "display_name": "Sabrina Rao",
            "role": "Controller",
            "phone_e164": "+15555550101",
        },
    ]
}


@pytest.mark.parametrize(
    "heard",
    [
        # Deepgram's two spellings of the CFO's surname on a live call.
        "Eric Catalado",
        "Eric Catawano",
        "Erik Catalano",
    ],
)
def test_a_misheard_surname_still_finds_the_one_closest_user(heard: str) -> None:
    ratio = SequenceMatcher(None, normalise_name(heard), "eric catalano").ratio()
    assert ratio >= FUZZY_CUTOFF, ratio
    found = directory(NO_ALIASES).search(heard)
    assert [u.user_id for u in found] == ["u_eric"]


def test_a_different_person_is_not_close_enough() -> None:
    ratio = SequenceMatcher(None, "sabrina rao", "eric catalano").ratio()
    assert ratio < FUZZY_CUTOFF
    assert [u.user_id for u in directory(NO_ALIASES).search("Sabrina Rao")] == [
        "u_sabrina"
    ]


def test_a_first_name_alone_finds_the_only_person_with_it() -> None:
    assert [u.user_id for u in directory(NO_ALIASES).search("Eric")] == ["u_eric"]
    assert [u.user_id for u in directory(NO_ALIASES).search("Catalano")] == ["u_eric"]


def test_a_first_name_shared_by_several_people_finds_all_of_them() -> None:
    # "eric" is inside "eric catalano", "erica lane" and u_eric_2's alias, so
    # the agent sees several and identifies nobody.
    found = directory(TWO_USERS).search("Eric")
    assert [u.user_id for u in found] == ["u_eric", "u_eric_2"]


def test_a_containing_match_beats_a_close_spelling() -> None:
    # "eric catalan" is inside two names; the fuzzy step never runs.
    found = directory(TWO_USERS).search("Eric Catalan")
    assert [u.user_id for u in found] == ["u_eric", "u_eric_2"]


def test_a_close_spelling_two_people_share_is_ambiguous() -> None:
    # u_eric_2 carries "eric catalano" as an alias, so the misheard name is
    # equally close to both: both come back, never a guess.
    found = directory(TWO_USERS).search("Eric Catalado")
    assert [u.user_id for u in found] == ["u_eric", "u_eric_2"]


def test_two_close_spellings_within_the_margin_are_ambiguous() -> None:
    doc = {
        "users": [
            {
                "user_id": "u_a",
                "display_name": "Eric Catalano",
                "role": "CFO",
                "phone_e164": "+15555550100",
            },
            {
                "user_id": "u_b",
                "display_name": "Eric Catalago",
                "role": "Controller",
                "phone_e164": "+15555550101",
            },
        ]
    }
    first = SequenceMatcher(None, "eric catalado", "eric catalano").ratio()
    second = SequenceMatcher(None, "eric catalado", "eric catalago").ratio()
    assert min(first, second) >= FUZZY_CUTOFF
    assert abs(first - second) < AMBIGUITY_MARGIN
    found = directory(doc).search("Eric Catalado")
    assert [u.user_id for u in found] == ["u_a", "u_b"]


def test_a_clear_winner_over_a_second_candidate_outside_the_margin() -> None:
    doc = {
        "users": [
            {
                "user_id": "u_a",
                "display_name": "Eric Catalano",
                "role": "CFO",
                "phone_e164": "+15555550100",
            },
            {
                "user_id": "u_b",
                "display_name": "Erica Catalina",
                "role": "Controller",
                "phone_e164": "+15555550101",
            },
        ]
    }
    first = SequenceMatcher(None, "eric catalado", "eric catalano").ratio()
    second = SequenceMatcher(None, "eric catalado", "erica catalina").ratio()
    assert first - second >= AMBIGUITY_MARGIN, (first, second)
    assert [u.user_id for u in directory(doc).search("Eric Catalado")] == ["u_a"]


async def test_route_search_tolerates_a_misheard_surname(v1: httpx.AsyncClient) -> None:
    response = await v1.get("/v1/users", params={"name": "Eric Catawano"})
    assert response.status_code == 200
    assert [item["user_id"] for item in response.json()] == ["u_eric"]


# --------------------------------------------------------------- routes


async def test_route_search(v1: httpx.AsyncClient) -> None:
    response = await v1.get("/v1/users", params={"name": "eric catalano"})
    assert response.status_code == 200
    body = [UserSummary.model_validate(item) for item in response.json()]
    assert [u.user_id for u in body] == ["u_eric"]
    assert body[0].phone_last_four == "0100"
    assert "phone_e164" not in response.json()[0]


async def test_route_search_with_no_match_is_an_empty_list(
    v1: httpx.AsyncClient,
) -> None:
    response = await v1.get("/v1/users", params={"name": "Nobody Here"})
    assert response.status_code == 200
    assert response.json() == []


async def test_route_get_user(v1: httpx.AsyncClient) -> None:
    response = await v1.get("/v1/users/u_eric")
    assert response.status_code == 200
    user = UserOnFile.model_validate(response.json())
    assert user.phone_e164 == "+15555550100"
    assert user.phone_last_four == "0100"


async def test_route_unknown_user_is_404(v1: httpx.AsyncClient) -> None:
    response = await v1.get("/v1/users/u_nobody")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "user_not_found"
    assert "u_nobody" not in response.text


async def test_a_lookup_leaves_no_name_or_number_in_the_log(
    v1_settings: Settings, clock: FakeClock
) -> None:
    stream = io.StringIO()
    app = build_app(v1_settings, clock, stream)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://detector.test",
        headers={
            "Authorization": f"Bearer {v1_settings.api_token}",
            "X-Contract-Version": "1.1.0",
        },
    ) as client:
        assert (await client.get("/v1/users", params={"name": "Eric Catalano"})).json()
        assert (await client.get("/v1/users/u_eric")).status_code == 200
    text = stream.getvalue()
    assert text  # something was logged
    for needle in ("Eric", "Catalano", "+15555550100", "5555550100"):
        assert needle not in text, needle


@pytest.mark.parametrize(
    "heard",
    ["Erik Catalano", "Eric Catalono", "erik catalono", "Eric Catalano."],
)
def test_the_example_file_carries_likely_stt_spellings(heard: str) -> None:
    text = (REPO_ROOT / "users_on_file.example.json").read_text(encoding="utf-8")
    found = UserDirectory.from_json_text(text).search(heard)
    assert [u.user_id for u in found] == ["u_eric"]
