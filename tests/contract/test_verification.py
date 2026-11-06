"""The details check: normalisation, the match policy, the combination table,
the attempt cap, and a dedicated group of leak tests.

These tests always run the router in process. Unlike the rest of this
directory they are not pointed at a deployed URL by CALLBACK_BASE_URL, because
the router is mounted by the app owner and a given deployment may not carry it
yet.

The leak group is the important one. It drives both the happy and the failure
paths and asserts that the expected account number never appears in a
serialised model, a log record, an error body, an exception string or a repr.
To accept an intended change to the schema snapshot:

    CALLBACK_UPDATE_SNAPSHOT=1 uv run pytest tests/contract/test_verification.py

then bump VERIFICATION_VERSION in shared/verification.py.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import random
import re
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from detector.config import Settings
from detector.state import DetectorEngine
from detector.verification import (
    COMBINATION_TABLE,
    MAX_ATTEMPTS_PER_SESSION,
    MAX_ATTEMPTS_PER_TARGET,
    PAYMENT_ACTION_BY_VERDICT,
    AttemptLedger,
    build_verification_router,
    combine_verdicts,
    reconcile_voice,
    run_verification,
)
from shared import verification as shared_verification
from shared.contract import Decision, EventRequest, EventType
from shared.verification import (
    CALLER_SCRIPT,
    LEAK_SUBSTRING_DIGITS,
    NEAR_MISS_CLASSES,
    NEAR_MISS_MIN_DIGITS,
    REDACTED_PLACEHOLDER,
    TARGET_DIGIT_RULES,
    VERIFICATION_VERSION,
    CallerDisclosure,
    LeakGuardError,
    MatchClass,
    PaymentAction,
    RedactedDigits,
    Verdict,
    VerificationRequest,
    VerificationResult,
    VerificationTarget,
    build_result,
    classify_match,
    disclosure_for,
    join_utterances,
    leaks,
    normalise_expected,
    parse_spoken_digits,
    safe_validation_detail,
    scrub,
    verdict_for_match,
    verification_schema_bundle,
    verify_request,
)
from tests.contract.conftest import TEST_TOKEN

# An eight digit account number whose four digit runs cannot collide with the
# small integers and version digits that legitimately appear in a response.
SECRET = "98765432"
SESSION = "sess-verification-1"


def make_request(
    *,
    transcript: str,
    expected: str = SECRET,
    session_id: str = SESSION,
    target: VerificationTarget = VerificationTarget.ACCOUNT_NUMBER,
    voice_decision: Decision | None = None,
    voice_risk_score: float | None = None,
) -> VerificationRequest:
    return VerificationRequest(
        session_id=session_id,
        target=target,
        expected=expected,
        transcript=transcript,
        voice_decision=voice_decision,
        voice_risk_score=voice_risk_score,
    )


# ------------------------------------------------------- spoken digit parsing

SPOKEN_FORMS: tuple[tuple[str, str], ...] = (
    # Plain digit words.
    ("four five six", "456"),
    ("nine eight seven six five four three two", "98765432"),
    # Bare digits and digit groups, however the transcriber grouped them.
    ("9876 5432", "98765432"),
    ("98-76-54-32", "98765432"),
    ("9,876,543", "9876543"),
    # Zero, every way it gets said.
    ("zero oh nought naught nil", "00000"),
    ("oh oh seven", "007"),
    # Teens arrive as one word and are two digits.
    ("nineteen eighty four", "1984"),
    ("ten eleven twelve", "101112"),
    # Tens compound with a following unit, and keep their zero without one.
    ("forty five", "45"),
    ("forty five six", "456"),
    ("sixty", "60"),
    ("twenty three ninety", "2390"),
    # A tens word followed by a zero word stays three digits rather than
    # losing one.
    ("forty oh", "400"),
    # Repeat words.
    ("double seven", "77"),
    ("triple eight", "888"),
    ("treble four", "444"),
    ("double oh", "00"),
    ("quadruple one", "1111"),
    ("double 5", "55"),
    # Filler, politeness and field names are dropped.
    ("um, er, the account number is, uh, four five six", "456"),
    ("so my sort code is nine eight, hold on, seven six five four", "987654"),
    ("yeah okay it ends four five six seven please", "4567"),
    # Separators and pauses.
    ("four - five ... six / seven", "4567"),
    ("four, pause, five, break, six", "456"),
    # Case and punctuation do not matter.
    ("FOUR Five SIX", "456"),
    # Digits split across utterance boundaries, joined before parsing.
    ("nine eight seven six\nfive four three two", "98765432"),
    # Homophones stay ordinary English rather than becoming digits.
    ("this is for you too", ""),
    ("one for the money two for the show", "12"),
    # Words no list anticipates are dropped, not guessed at.
    ("four flibbertigibbet five", "45"),
)


@pytest.mark.parametrize(("transcript", "expected_digits"), list(SPOKEN_FORMS))
def test_parses_spoken_forms(transcript: str, expected_digits: str) -> None:
    assert parse_spoken_digits(transcript).digits == expected_digits


def test_tens_compound_matches_a_digit_by_digit_reading() -> None:
    # "forty five" and "four five" are the same two digits. That is why the
    # tens word is combined with its unit instead of expanding to 40.
    assert (
        parse_spoken_digits("forty five").digits
        == parse_spoken_digits("four five").digits
    )


def test_ignored_tokens_are_counted() -> None:
    parsed = parse_spoken_digits("four flibbertigibbet five whatsit")
    assert parsed.digits == "45"
    assert parsed.ignored_token_count == 2


AMBIGUOUS_FORMS: tuple[str, ...] = (
    "four hundred and twenty",
    "nine hundred seven",
    "twelve hundred thirty four",
    "two thousand one",
    "five grand",
    # A repeat word with nothing single to repeat.
    "double",
    "double forty",
    "triple 45",
    "the account number is double",
)


@pytest.mark.parametrize("transcript", list(AMBIGUOUS_FORMS))
def test_ambiguous_wording_is_refused_rather_than_guessed(transcript: str) -> None:
    parsed = parse_spoken_digits(transcript)
    assert parsed.ambiguity is not None
    assert parsed.digits == ""
    assert not parsed.is_usable


def test_empty_transcript_is_unusable_but_not_ambiguous() -> None:
    for transcript in ("", "   ", "hello, is this the right number"):
        parsed = parse_spoken_digits(transcript)
        assert parsed.digits == ""
        assert parsed.ambiguity is None
        assert not parsed.is_usable


def test_join_utterances_lets_a_number_cross_a_fragment_boundary() -> None:
    # "forty" and "five" arriving as separate fragments are one number. Parsed
    # separately they would be 40 then 5.
    assert parse_spoken_digits(join_utterances(["forty", "five"])).digits == "45"
    assert join_utterances("already joined") == "already joined"


def test_expected_side_accepts_bank_formatting_and_rejects_words() -> None:
    assert normalise_expected("98-76-54-32") == SECRET
    assert normalise_expected("9876 5432") == SECRET
    assert normalise_expected("9876.5432") == SECRET
    with pytest.raises(ValueError, match="only digits and separators"):
        normalise_expected("GB29 NWBK 6016 1331 9268 19")


# -------------------------------------------------------------- match policy


def test_exact_match_is_verified() -> None:
    result = verify_request(
        make_request(transcript="nine eight seven six five four three two"),
        attempts_used=1,
        attempts_remaining=2,
    )
    assert result.match_class is MatchClass.EXACT
    assert result.verdict is Verdict.VERIFIED
    assert result.caller_disclosure is CallerDisclosure.CONFIRMED


def test_exact_match_through_messy_wording_is_still_verified() -> None:
    result = verify_request(
        make_request(
            transcript=(
                "um so the new account number is, uh, nine eight, "
                "seven six, five four, three two, yeah"
            )
        ),
        attempts_used=1,
        attempts_remaining=2,
    )
    assert result.verdict is Verdict.VERIFIED


CLEAR_DIFFERENCES: tuple[str, ...] = (
    "12345678",  # same length, nothing in common
    "98765431098",  # much longer
    "9876",  # much shorter
    "98765234",  # two substitutions
    "38765492",  # a swap of two digits that are not neighbours
    "89765431",  # an adjacent transposition plus a substitution
)


@pytest.mark.parametrize("spoken", list(CLEAR_DIFFERENCES))
def test_clear_differences_are_mismatch(spoken: str) -> None:
    match = classify_match(RedactedDigits(SECRET), spoken)
    assert match is MatchClass.UNRELATED
    assert verdict_for_match(match) is Verdict.MISMATCH


NEAR_MISSES: tuple[tuple[str, MatchClass], ...] = (
    ("98765433", MatchClass.SINGLE_DIGIT_SUBSTITUTION),
    ("08765432", MatchClass.SINGLE_DIGIT_SUBSTITUTION),
    ("98765342", MatchClass.ADJACENT_TRANSPOSITION),
    ("89765432", MatchClass.ADJACENT_TRANSPOSITION),
    ("9876542", MatchClass.MISSING_DIGIT),
    ("8765432", MatchClass.MISSING_DIGIT),
    ("987654329", MatchClass.EXTRA_DIGIT),
    ("598765432", MatchClass.EXTRA_DIGIT),
)


@pytest.mark.parametrize(("spoken", "expected_class"), list(NEAR_MISSES))
def test_every_near_miss_class_lands_in_needs_review(
    spoken: str, expected_class: MatchClass
) -> None:
    match = classify_match(RedactedDigits(SECRET), spoken)
    assert match is expected_class
    assert match in NEAR_MISS_CLASSES
    assert verdict_for_match(match) is Verdict.NEEDS_REVIEW


def test_a_near_miss_is_never_silently_forgiven() -> None:
    for spoken, _ in NEAR_MISSES:
        assert verdict_for_match(
            classify_match(RedactedDigits(SECRET), spoken)
        ) is not (Verdict.VERIFIED)


def test_transposition_only_counts_when_the_digits_are_neighbours() -> None:
    # 9876 54 32 -> swap positions 0 and 7: not neighbours, so two
    # substitutions and a mismatch rather than a forgiven slip.
    assert classify_match(RedactedDigits(SECRET), "28765439") is MatchClass.UNRELATED


def test_short_numbers_admit_no_near_miss() -> None:
    # A sort code is exactly at the floor, so it still gets near misses.
    assert len(SECRET) >= NEAR_MISS_MIN_DIGITS
    short = "9876"
    assert len(short) < NEAR_MISS_MIN_DIGITS
    match = classify_match(RedactedDigits(short), "9875")
    assert match is MatchClass.UNRELATED
    assert verdict_for_match(match) is Verdict.MISMATCH


def test_unparseable_transcript_is_needs_review_not_mismatch() -> None:
    result = verify_request(
        make_request(transcript="um, hold on, let me find the letter"),
        attempts_used=1,
        attempts_remaining=2,
    )
    assert result.match_class is MatchClass.UNPARSEABLE
    assert result.verdict is Verdict.NEEDS_REVIEW
    assert result.transcript_ambiguous is False


def test_ambiguous_transcript_is_flagged_as_ambiguous() -> None:
    result = verify_request(
        make_request(transcript="it is nine eight seven six five four hundred"),
        attempts_used=1,
        attempts_remaining=2,
    )
    assert result.match_class is MatchClass.UNPARSEABLE
    assert result.verdict is Verdict.NEEDS_REVIEW
    assert result.transcript_ambiguous is True


def test_only_exact_maps_to_verified() -> None:
    verified = [m for m in MatchClass if verdict_for_match(m) is Verdict.VERIFIED]
    assert verified == [MatchClass.EXACT]


def test_sort_code_and_routing_number_are_compared_too() -> None:
    sort_code = verify_request(
        make_request(
            transcript="double two, three three, four four",
            expected="223344",
            target=VerificationTarget.SORT_CODE,
        ),
        attempts_used=1,
        attempts_remaining=2,
    )
    assert sort_code.verdict is Verdict.VERIFIED

    routing = verify_request(
        make_request(
            transcript="one two one, oh oh oh, three five eight",
            expected="121000358",
            target=VerificationTarget.ROUTING_NUMBER,
        ),
        attempts_used=1,
        attempts_remaining=2,
    )
    assert routing.verdict is Verdict.VERIFIED


def test_target_digit_rules_are_enforced_on_the_expected_value() -> None:
    for target, (low, high) in TARGET_DIGIT_RULES.items():
        with pytest.raises(ValueError, match="must have between"):
            make_request(transcript="one", expected="1" * (low - 1), target=target)
        if high < 20:
            with pytest.raises(ValueError, match="must have between"):
                make_request(transcript="one", expected="1" * (high + 1), target=target)


# ------------------------------------------------------ caller disclosure


def test_failing_callers_hear_the_same_words_whatever_went_wrong() -> None:
    assert disclosure_for(Verdict.MISMATCH) is CallerDisclosure.NOT_CONFIRMED
    assert disclosure_for(Verdict.NEEDS_REVIEW) is CallerDisclosure.NOT_CONFIRMED
    assert disclosure_for(Verdict.VERIFIED) is CallerDisclosure.CONFIRMED
    # Two values, so there is no third thing to accidentally say.
    assert len(CallerDisclosure) == 2


def test_caller_script_contains_no_digits_and_no_position_words() -> None:
    for text in CALLER_SCRIPT.values():
        assert not re.search(r"\d", text)
        for giveaway in ("digit", "last four", "length", "first", "position"):
            assert giveaway not in text.lower()


def test_result_exposes_only_the_caller_safe_wording() -> None:
    result = verify_request(
        make_request(transcript="98765433"), attempts_used=1, attempts_remaining=2
    )
    assert result.verdict is Verdict.NEEDS_REVIEW
    assert result.caller_script() == CALLER_SCRIPT[CallerDisclosure.NOT_CONFIRMED]


# ----------------------------------------------------- the combination table


def _details(verdict: Verdict) -> VerificationResult:
    """A details result with the requested verdict, built the normal way."""

    match = {
        Verdict.VERIFIED: MatchClass.EXACT,
        Verdict.MISMATCH: MatchClass.UNRELATED,
        Verdict.NEEDS_REVIEW: MatchClass.SINGLE_DIGIT_SUBSTITUTION,
    }[verdict]
    return build_result(
        session_id=SESSION,
        target=VerificationTarget.ACCOUNT_NUMBER,
        expected=RedactedDigits(SECRET),
        match=match,
        attempts_used=1,
        attempts_remaining=2,
    )


EXPECTED_TABLE: tuple[tuple[Decision, Verdict, Verdict, PaymentAction], ...] = (
    (
        Decision.ALLOW,
        Verdict.VERIFIED,
        Verdict.VERIFIED,
        PaymentAction.RELEASE_FOR_APPROVAL,
    ),
    (
        Decision.ALLOW,
        Verdict.NEEDS_REVIEW,
        Verdict.NEEDS_REVIEW,
        PaymentAction.HOLD_FOR_REVIEW,
    ),
    (Decision.ALLOW, Verdict.MISMATCH, Verdict.MISMATCH, PaymentAction.BLOCK),
    (
        Decision.CHALLENGE,
        Verdict.VERIFIED,
        Verdict.NEEDS_REVIEW,
        PaymentAction.HOLD_FOR_REVIEW,
    ),
    (
        Decision.CHALLENGE,
        Verdict.NEEDS_REVIEW,
        Verdict.NEEDS_REVIEW,
        PaymentAction.HOLD_FOR_REVIEW,
    ),
    (Decision.CHALLENGE, Verdict.MISMATCH, Verdict.MISMATCH, PaymentAction.BLOCK),
    (
        Decision.PENDING,
        Verdict.VERIFIED,
        Verdict.NEEDS_REVIEW,
        PaymentAction.HOLD_FOR_REVIEW,
    ),
    (
        Decision.PENDING,
        Verdict.NEEDS_REVIEW,
        Verdict.NEEDS_REVIEW,
        PaymentAction.HOLD_FOR_REVIEW,
    ),
    (Decision.PENDING, Verdict.MISMATCH, Verdict.MISMATCH, PaymentAction.BLOCK),
    (Decision.DENY, Verdict.VERIFIED, Verdict.MISMATCH, PaymentAction.BLOCK),
    (Decision.DENY, Verdict.NEEDS_REVIEW, Verdict.MISMATCH, PaymentAction.BLOCK),
    (Decision.DENY, Verdict.MISMATCH, Verdict.MISMATCH, PaymentAction.BLOCK),
)


@pytest.mark.parametrize(
    ("voice", "details_verdict", "product", "action"), list(EXPECTED_TABLE)
)
def test_combination_truth_table(
    voice: Decision,
    details_verdict: Verdict,
    product: Verdict,
    action: PaymentAction,
) -> None:
    combined = combine_verdicts(
        session_id=SESSION,
        details=_details(details_verdict),
        voice_decision=voice,
    )
    assert combined.verdict is product
    assert combined.payment_action is action
    assert combined.voice_decision is voice
    assert combined.details_verdict is details_verdict
    assert combined.caller_disclosure is disclosure_for(product)


def test_the_table_is_total() -> None:
    assert set(COMBINATION_TABLE) == {
        (decision, verdict) for decision in Decision for verdict in Verdict
    }
    assert len(COMBINATION_TABLE) == len(Decision) * len(Verdict)
    assert set(PAYMENT_ACTION_BY_VERDICT) == set(Verdict)


def test_exactly_one_cell_verifies() -> None:
    verifying = [k for k, v in COMBINATION_TABLE.items() if v is Verdict.VERIFIED]
    assert verifying == [(Decision.ALLOW, Verdict.VERIFIED)]


def test_a_mismatch_dominates_any_voice_result() -> None:
    for decision in Decision:
        assert COMBINATION_TABLE[(decision, Verdict.MISMATCH)] is Verdict.MISMATCH


def test_a_genuine_voice_saying_the_wrong_number_is_never_verified() -> None:
    combined = combine_verdicts(
        session_id=SESSION,
        details=_details(Verdict.MISMATCH),
        voice_decision=Decision.ALLOW,
        voice_risk_score=0.01,
    )
    assert combined.verdict is Verdict.MISMATCH
    assert combined.payment_action is PaymentAction.BLOCK


def test_a_denied_voice_is_never_verified_however_good_the_read_back() -> None:
    for verdict in Verdict:
        assert COMBINATION_TABLE[(Decision.DENY, verdict)] is not Verdict.VERIFIED


def test_an_absent_voice_decision_is_treated_as_pending() -> None:
    combined = combine_verdicts(
        session_id=SESSION,
        details=_details(Verdict.VERIFIED),
        voice_decision=None,
    )
    assert combined.voice_decision is Decision.PENDING
    assert combined.verdict is Verdict.NEEDS_REVIEW


def test_risk_score_inconsistent_with_allow_is_downgraded() -> None:
    # An allow carrying a risk the detector's own thresholds would not allow
    # is a contradiction, so the stricter reading wins.
    assert reconcile_voice(Decision.ALLOW, 0.45)[0] is Decision.CHALLENGE
    assert reconcile_voice(Decision.ALLOW, 0.90)[0] is Decision.DENY
    assert reconcile_voice(Decision.CHALLENGE, 0.90)[0] is Decision.DENY
    assert reconcile_voice(Decision.ALLOW, 0.05)[0] is Decision.ALLOW
    assert reconcile_voice(Decision.ALLOW, None)[0] is Decision.ALLOW
    assert reconcile_voice(None, None)[0] is Decision.PENDING

    combined = combine_verdicts(
        session_id=SESSION,
        details=_details(Verdict.VERIFIED),
        voice_decision=Decision.ALLOW,
        voice_risk_score=0.45,
    )
    assert combined.verdict is Verdict.NEEDS_REVIEW


# ------------------------------------------------------------- attempt cap


def test_attempts_are_capped_per_call() -> None:
    ledger = AttemptLedger()
    for attempt in range(1, MAX_ATTEMPTS_PER_SESSION + 1):
        answer = run_verification(
            make_request(transcript="1 2 3 4 5 6 7 8"), ledger=ledger
        )
        assert answer.details.rate_limited is False
        assert answer.details.attempts_used == attempt
        assert answer.details.attempts_remaining == MAX_ATTEMPTS_PER_SESSION - attempt

    refused = run_verification(
        make_request(transcript="1 2 3 4 5 6 7 8"), ledger=ledger
    )
    assert refused.details.rate_limited is True
    assert refused.details.match_class is MatchClass.NOT_COMPARED
    assert refused.details.verdict is Verdict.NEEDS_REVIEW
    assert refused.details.attempts_remaining == 0


def test_the_cap_refuses_even_the_correct_number() -> None:
    """Fail closed: at the cap nothing is compared, so nothing is disclosed."""

    ledger = AttemptLedger()
    for _ in range(MAX_ATTEMPTS_PER_SESSION):
        run_verification(make_request(transcript="1 2 3 4 5 6 7 8"), ledger=ledger)

    correct = run_verification(
        make_request(
            transcript="nine eight seven six five four three two",
            voice_decision=Decision.ALLOW,
            voice_risk_score=0.01,
        ),
        ledger=ledger,
    )
    assert correct.details.verdict is Verdict.NEEDS_REVIEW
    assert correct.combined.verdict is Verdict.NEEDS_REVIEW
    assert correct.combined.payment_action is PaymentAction.HOLD_FOR_REVIEW


def test_retrying_past_the_cap_does_not_reset_the_counter() -> None:
    ledger = AttemptLedger()
    for _ in range(MAX_ATTEMPTS_PER_SESSION + 3):
        answer = run_verification(
            make_request(transcript="1 2 3 4 5 6 7 8"), ledger=ledger
        )
    assert answer.details.rate_limited is True
    assert answer.details.attempts_used > MAX_ATTEMPTS_PER_SESSION


def test_each_field_gets_its_own_allowance() -> None:
    ledger = AttemptLedger()
    for _ in range(MAX_ATTEMPTS_PER_SESSION):
        run_verification(make_request(transcript="1 2 3 4 5 6 7 8"), ledger=ledger)

    sort_code = run_verification(
        make_request(
            transcript="double two three three four four",
            expected="223344",
            target=VerificationTarget.SORT_CODE,
        ),
        ledger=ledger,
    )
    assert sort_code.details.rate_limited is False
    assert sort_code.details.verdict is Verdict.VERIFIED


def test_one_expected_value_is_capped_across_sessions() -> None:
    """Otherwise opening a new session resets the guesser's budget."""

    ledger = AttemptLedger()
    refusals = 0
    for index in range(MAX_ATTEMPTS_PER_TARGET + 2):
        answer = run_verification(
            make_request(
                transcript="1 2 3 4 5 6 7 8", session_id=f"sess-fresh-{index}"
            ),
            ledger=ledger,
        )
        refusals += int(answer.details.rate_limited)
    assert refusals == 2


def test_a_different_expected_value_has_its_own_cross_session_budget() -> None:
    ledger = AttemptLedger()
    for index in range(MAX_ATTEMPTS_PER_TARGET):
        run_verification(
            make_request(transcript="1 2 3 4 5 6 7 8", session_id=f"sess-a-{index}"),
            ledger=ledger,
        )
    other = run_verification(
        make_request(
            transcript="1 2 3 4 5 6 7 8",
            expected="11223344",
            session_id="sess-b-0",
        ),
        ledger=ledger,
    )
    assert other.details.rate_limited is False


def test_ledger_evicts_rather_than_growing_without_bound() -> None:
    ledger = AttemptLedger(max_tracked=4)
    for index in range(20):
        run_verification(
            make_request(transcript="1 2 3", session_id=f"sess-evict-{index}"),
            ledger=ledger,
        )
    assert len(ledger._per_session) <= 4
    assert len(ledger._per_target) <= 4


# --------------------------------------------------------------- leak tests
#
# The expected account number must not reach any surface a caller, an operator
# reading logs, or a client handling an error can see. Each test below drives a
# real path and then searches the output.


def _secret() -> RedactedDigits:
    return RedactedDigits(SECRET)


def test_repr_str_and_format_all_redact() -> None:
    value = _secret()
    assert SECRET not in repr(value)
    assert SECRET not in str(value)
    assert SECRET not in f"{value}"
    assert SECRET not in f"{value!r}"
    assert SECRET not in f"{value!s}"
    assert SECRET not in format(value, ">40")
    assert REDACTED_PLACEHOLDER in repr(value)


def test_request_repr_redacts_the_expected_value() -> None:
    request = make_request(transcript="four five six")
    assert SECRET not in repr(request)
    assert not leaks(repr(request), _secret())


def test_request_serialisation_never_carries_the_expected_value() -> None:
    request = make_request(transcript="four five six")
    as_json = request.model_dump_json()
    assert not leaks(as_json, _secret())
    assert json.loads(as_json)["expected"] == REDACTED_PLACEHOLDER
    # Python mode hands back the wrapper, whose repr is redacted.
    assert SECRET not in str(request.model_dump())
    assert not leaks(str(request.model_dump(mode="json")), _secret())


@pytest.mark.parametrize(
    "transcript",
    [
        "nine eight seven six five four three two",  # exact
        "nine eight seven six five four three three",  # substitution
        "nine eight seven six five four two three",  # transposition
        "nine eight seven six five four three",  # missing digit
        "nine eight seven six five four three two one",  # extra digit
        "one two three four five six seven eight",  # unrelated
        "um, hold on",  # unparseable
        "nine eight seven hundred",  # ambiguous
    ],
)
def test_no_result_on_any_path_discloses_the_expected_value(transcript: str) -> None:
    answer = run_verification(
        make_request(transcript=transcript, voice_decision=Decision.ALLOW),
        ledger=AttemptLedger(),
    )
    rendered = answer.model_dump_json()
    assert not leaks(rendered, _secret())
    assert SECRET not in repr(answer)
    for text in (answer.details.reason, answer.combined.reason):
        assert not re.search(r"\d", text), text


def test_the_refused_path_discloses_nothing_either() -> None:
    ledger = AttemptLedger()
    for _ in range(MAX_ATTEMPTS_PER_SESSION + 1):
        answer = run_verification(
            make_request(transcript="nine eight seven six five four three two"),
            ledger=ledger,
        )
    assert answer.details.rate_limited is True
    assert not leaks(answer.model_dump_json(), _secret())


def test_results_carry_no_length_or_position_information() -> None:
    """A near miss may name the shape of the error, never where or how long.

    The shape alone already tells the caller they are within one edit, which
    is why the caller-facing disclosure is two valued and the attempt cap
    exists. Position and length would narrow it much further.
    """

    answer = run_verification(
        make_request(transcript="nine eight seven six five four three three"),
        ledger=AttemptLedger(),
    )
    rendered = answer.model_dump_json()
    for giveaway in ("last four", "position", "index", "length", "digits long"):
        assert giveaway not in rendered.lower()
    # No integer in the payload is derived from the expected value: the only
    # numbers are the attempt counters and the version string.
    numbers = set(re.findall(r"\d+", rendered))
    assert numbers <= {"0", "1", "2", "3"}


def test_nothing_logged_contains_the_expected_value(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    ledger = AttemptLedger()
    for transcript in (
        "nine eight seven six five four three two",
        "nine eight seven six five four three three",
        "one two three four five six seven eight",
        "um",
        "nine eight seven six five four three two",
    ):
        run_verification(make_request(transcript=transcript), ledger=ledger)

    assert caplog.records, "expected the verification path to log something"
    for record in caplog.records:
        assert not leaks(record.getMessage(), _secret())
        assert not leaks(str(record.args), _secret())


def test_exception_strings_do_not_quote_the_bad_value() -> None:
    with pytest.raises(ValueError, match="only digits and separators") as exc_info:
        RedactedDigits("98765432x")
    assert "98765432" not in str(exc_info.value)

    with pytest.raises(ValueError, match="more than") as too_long:
        RedactedDigits("1" * 100)
    assert "1111" not in str(too_long.value)


def test_a_validation_error_must_be_rendered_by_the_safe_helper() -> None:
    """str(ValidationError) is a leak, which is why the helper exists.

    pydantic attaches the offending input to every error: the raw string for a
    field failure, and the whole raw request dict for the model-level target
    rule. `str(exc)` only hides it by truncating long inputs, so nothing may
    put a ValidationError into a response or a log.
    """

    bad_value = "7654321987"
    with pytest.raises(ValidationError, match="must have between") as bad_shape:
        VerificationRequest(
            session_id=SESSION,
            target=VerificationTarget.SORT_CODE,
            expected=bad_value,
            transcript="one",
        )

    # The hazard, asserted rather than assumed: the raw value really is in
    # there. If a future pydantic stops doing this, this test should be
    # relaxed, not the renderer.
    raw_input = bad_shape.value.errors()[0]["input"]
    assert bad_value in str(raw_input)

    detail = safe_validation_detail(bad_shape.value)
    assert bad_value not in detail
    assert "must have between" in detail
    assert not leaks(detail, RedactedDigits(bad_value))


def test_the_safe_helper_strips_a_bad_expected_value() -> None:
    bad_value = "98765432x"
    with pytest.raises(ValidationError) as exc_info:
        VerificationRequest(
            session_id=SESSION,
            target=VerificationTarget.ACCOUNT_NUMBER,
            expected=bad_value,
            transcript="one",
        )
    assert "98765432" in str(exc_info.value), "the hazard this helper exists for"
    detail = safe_validation_detail(exc_info.value)
    assert "98765432" not in detail
    assert not leaks(detail, _secret())


def test_the_leak_detector_recognises_the_obvious_dodges() -> None:
    secret = _secret()
    assert leaks(f"the number is {SECRET}", secret)
    assert leaks("ends 9876", secret)
    assert leaks("ends 9 8 7 6", secret)
    assert leaks("ends 9-8-7-6", secret)
    assert not leaks("no digits here", secret)
    assert not leaks("attempt 2 of 3", secret)
    assert SECRET not in scrub(f"the number is {SECRET}", secret)
    assert REDACTED_PLACEHOLDER in scrub(f"the number is {SECRET}", secret)


def test_a_reason_line_that_would_leak_raises_instead_of_answering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard is a regression test for a future edit, not for today.

    Every reason line is a constant with no digits in it, so the only way to
    prove the guard is wired in is to put a disclosure in one of those
    constants and watch the build refuse.
    """

    patched = dict(shared_verification._REASON_BY_MATCH)
    patched[MatchClass.EXACT] = "the account ends 9876"
    monkeypatch.setattr(shared_verification, "_REASON_BY_MATCH", patched)

    with pytest.raises(LeakGuardError):
        build_result(
            session_id=SESSION,
            target=VerificationTarget.ACCOUNT_NUMBER,
            expected=_secret(),
            match=MatchClass.EXACT,
            attempts_used=1,
            attempts_remaining=2,
        )


def test_the_secret_cannot_be_hashed_or_pickled() -> None:
    value = _secret()
    with pytest.raises(TypeError):
        hash(value)
    with pytest.raises(TypeError):
        pickle.dumps(value)
    # deepcopy still works, because pydantic's model_copy(deep=True) needs it.
    assert deepcopy(value) == value
    assert SECRET not in repr(deepcopy(value))


def test_the_secret_is_immutable_and_compares_in_constant_time() -> None:
    value = _secret()
    with pytest.raises(AttributeError):
        value._digits = "1"  # type: ignore[misc]
    assert value == SECRET
    assert value == RedactedDigits("98-76-54-32")
    assert value != "98765431"
    assert (value == 98765432) is False


def test_no_length_accessor_is_exposed() -> None:
    value = _secret()
    with pytest.raises(TypeError):
        len(value)  # type: ignore[arg-type]
    assert not hasattr(value, "digit_count")


# ------------------------------------------------------------- the endpoint


def make_app(
    settings: Settings,
    *,
    ledger: AttemptLedger | None = None,
    engine: DetectorEngine | None = None,
) -> FastAPI:
    app = FastAPI()
    app.state.settings = settings
    if engine is not None:
        app.state.engine = engine
    app.include_router(build_verification_router(settings, ledger=ledger))
    return app


def make_client(app: FastAPI, *, token: str | None = TEST_TOKEN) -> httpx.AsyncClient:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://detector.test",
        headers=headers,
    )


def drive_engine(engine: DetectorEngine, session_id: str, scenario: str) -> None:
    """Run one session through the detector far enough to reach a decision."""

    events = (
        EventType.SESSION_START,
        EventType.CALLER_CLAIM,
        EventType.SPEECH_SEGMENT,
        EventType.SPEECH_SEGMENT,
    )
    for sequence, event in enumerate(events, start=1):
        engine.handle_event(
            EventRequest(
                session_id=session_id,
                event=event,
                sequence=sequence,
                occurred_at=datetime.now(UTC),
                claimed_user_id=scenario,
            ),
            scenario_header=scenario,
            idempotency_key=None,
        )


def body(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "session_id": SESSION,
        "target": "account_number",
        "expected": SECRET,
        "transcript": "nine eight seven six five four three two",
    }
    payload.update(overrides)
    return payload


async def test_endpoint_verifies_a_clean_read_back(settings: Settings) -> None:
    async with make_client(make_app(settings)) as client:
        response = await client.post(
            "/verify/bank-details", json=body(voice_decision="allow")
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["details"]["verdict"] == "verified"
    assert payload["combined"]["verdict"] == "verified"
    assert payload["combined"]["payment_action"] == "release_for_approval"
    assert payload["verification_version"] == VERIFICATION_VERSION
    assert not leaks(response.text, _secret())


async def test_endpoint_needs_a_token(settings: Settings) -> None:
    async with make_client(make_app(settings), token=None) as client:
        response = await client.post("/verify/bank-details", json=body())
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"

    async with make_client(make_app(settings), token="wrong") as client:
        response = await client.post("/verify/bank-details", json=body())
    assert response.status_code == 401


async def test_endpoint_rejects_unknown_fields(settings: Settings) -> None:
    async with make_client(make_app(settings)) as client:
        response = await client.post(
            "/verify/bank-details", json=body(typo_field="oops")
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


async def test_a_malformed_expected_value_is_not_echoed_back(
    settings: Settings,
) -> None:
    """The reason the body is validated here instead of by FastAPI.

    pydantic attaches the offending input to every error, and the app's own
    handler stringifies those errors into the response body.
    """

    async with make_client(make_app(settings)) as client:
        response = await client.post(
            "/verify/bank-details", json=body(expected="98765432x")
        )
    assert response.status_code == 422
    assert "98765432" not in response.text
    assert not leaks(response.text, _secret())


async def test_a_bad_body_is_a_clean_422(settings: Settings) -> None:
    async with make_client(make_app(settings)) as client:
        for payload in ("not json at all", "[1, 2, 3]", "null"):
            response = await client.post(
                "/verify/bank-details",
                content=payload,
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code == 422, payload
            assert response.json()["error"]["code"] == "validation_error"


async def test_endpoint_enforces_the_attempt_cap(settings: Settings) -> None:
    app = make_app(settings, ledger=AttemptLedger())
    async with make_client(app) as client:
        for _ in range(MAX_ATTEMPTS_PER_SESSION):
            await client.post(
                "/verify/bank-details",
                json=body(transcript="one two three four five six seven eight"),
            )
        response = await client.post("/verify/bank-details", json=body())
    payload = response.json()
    assert payload["details"]["rate_limited"] is True
    assert payload["details"]["verdict"] == "needs_review"
    assert payload["combined"]["payment_action"] == "hold_for_review"


async def test_endpoint_reads_the_voice_verdict_from_the_detector(
    settings: Settings,
) -> None:
    """No voice_decision in the body: the engine's own verdict is used."""

    engine = DetectorEngine(settings)
    session_id = "sess-engine-allow-1"
    drive_engine(engine, session_id, "real_cfo")
    assert engine.factors_for(session_id).decision is Decision.ALLOW

    async with make_client(make_app(settings, engine=engine)) as client:
        response = await client.post(
            "/verify/bank-details", json=body(session_id=session_id)
        )
    payload = response.json()
    assert payload["combined"]["voice_decision"] == "allow"
    assert payload["combined"]["verdict"] == "verified"


async def test_an_unknown_session_falls_back_to_pending(settings: Settings) -> None:
    engine = DetectorEngine(settings)
    async with make_client(make_app(settings, engine=engine)) as client:
        response = await client.post(
            "/verify/bank-details", json=body(session_id="sess-never-seen-1")
        )
    payload = response.json()
    assert payload["combined"]["voice_decision"] == "pending"
    assert payload["combined"]["verdict"] == "needs_review"


async def test_a_cloned_voice_reading_the_right_number_is_blocked(
    settings: Settings,
) -> None:
    engine = DetectorEngine(settings)
    session_id = "sess-engine-replay-1"
    drive_engine(engine, session_id, "replay")
    assert engine.factors_for(session_id).decision is Decision.DENY

    async with make_client(make_app(settings, engine=engine)) as client:
        response = await client.post(
            "/verify/bank-details", json=body(session_id=session_id)
        )
    payload = response.json()
    assert payload["details"]["verdict"] == "verified"
    assert payload["combined"]["verdict"] == "mismatch"
    assert payload["combined"]["payment_action"] == "block"


# -------------------------------------------------------- property style


def random_digits(rng: random.Random, length: int) -> str:
    """Digits drawn from 2 to 9.

    Excluding 0 and 1 keeps a generated secret's four digit runs from
    colliding with the version string and the small attempt counters that
    legitimately appear in a response, so the leak assertions below cannot
    produce a false positive.
    """

    return "".join(rng.choice("23456789") for _ in range(length))


def spoken(digits: str) -> str:
    words = {
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
    }
    return " ".join(words[d] for d in digits)


def test_property_a_verbatim_read_back_always_verifies() -> None:
    rng = random.Random(20261104)
    for _ in range(300):
        number = random_digits(rng, rng.randint(NEAR_MISS_MIN_DIGITS, 17))
        request = make_request(transcript=spoken(number), expected=number)
        result = verify_request(request, attempts_used=1, attempts_remaining=2)
        assert result.verdict is Verdict.VERIFIED, number


def test_property_messy_wording_still_verifies() -> None:
    rng = random.Random(7)
    fillers = ("um", "er", "so", "the account number is", "hold on", "yeah", "")
    for _ in range(300):
        number = random_digits(rng, rng.randint(NEAR_MISS_MIN_DIGITS, 14))
        parts = []
        for digit in spoken(number).split():
            if rng.random() < 0.25:
                parts.append(rng.choice(fillers))
            parts.append(digit)
            if rng.random() < 0.15:
                parts.append(rng.choice(("-", ",", "...")))
        transcript = " ".join(parts)
        result = verify_request(
            make_request(transcript=transcript, expected=number),
            attempts_used=1,
            attempts_remaining=2,
        )
        assert result.verdict is Verdict.VERIFIED, transcript


def test_property_one_substituted_digit_is_always_needs_review() -> None:
    rng = random.Random(11)
    for _ in range(300):
        number = random_digits(rng, rng.randint(NEAR_MISS_MIN_DIGITS, 17))
        index = rng.randrange(len(number))
        replacement = rng.choice([d for d in "0123456789" if d != number[index]])
        corrupted = number[:index] + replacement + number[index + 1 :]
        match = classify_match(RedactedDigits(number), corrupted)
        assert match is MatchClass.SINGLE_DIGIT_SUBSTITUTION, (number, corrupted)
        assert verdict_for_match(match) is Verdict.NEEDS_REVIEW


def test_property_one_adjacent_transposition_is_always_needs_review() -> None:
    rng = random.Random(13)
    checked = 0
    while checked < 300:
        number = random_digits(rng, rng.randint(NEAR_MISS_MIN_DIGITS, 17))
        index = rng.randrange(len(number) - 1)
        if number[index] == number[index + 1]:
            continue
        corrupted = (
            number[:index] + number[index + 1] + number[index] + number[index + 2 :]
        )
        assert (
            classify_match(RedactedDigits(number), corrupted)
            is MatchClass.ADJACENT_TRANSPOSITION
        ), (number, corrupted)
        checked += 1


def test_property_one_dropped_or_added_digit_is_always_needs_review() -> None:
    rng = random.Random(17)
    for _ in range(300):
        number = random_digits(rng, rng.randint(NEAR_MISS_MIN_DIGITS + 1, 17))
        cut = rng.randrange(len(number))
        shorter = number[:cut] + number[cut + 1 :]
        assert (
            classify_match(RedactedDigits(number), shorter) is MatchClass.MISSING_DIGIT
        ), (number, shorter)

        at = rng.randrange(len(number) + 1)
        longer = number[:at] + rng.choice("0123456789") + number[at:]
        assert (
            classify_match(RedactedDigits(number), longer) is MatchClass.EXTRA_DIGIT
        ), (number, longer)


def test_property_two_separated_substitutions_are_always_mismatch() -> None:
    rng = random.Random(19)
    checked = 0
    while checked < 300:
        number = random_digits(rng, rng.randint(NEAR_MISS_MIN_DIGITS, 17))
        first = rng.randrange(len(number))
        second = rng.randrange(len(number))
        if abs(first - second) < 2:
            continue
        corrupted = list(number)
        for index in (first, second):
            corrupted[index] = rng.choice(
                [d for d in "0123456789" if d != number[index]]
            )
        spoken_digits = "".join(corrupted)
        match = classify_match(RedactedDigits(number), spoken_digits)
        assert match is MatchClass.UNRELATED, (number, spoken_digits)
        assert verdict_for_match(match) is Verdict.MISMATCH
        checked += 1


def test_property_no_response_ever_leaks_the_expected_value() -> None:
    rng = random.Random(23)
    corruptions = ("exact", "substitute", "transpose", "drop", "add", "random", "noise")
    for _ in range(400):
        number = random_digits(rng, rng.randint(NEAR_MISS_MIN_DIGITS, 17))
        kind = rng.choice(corruptions)
        if kind == "exact":
            transcript = spoken(number)
        elif kind == "substitute":
            index = rng.randrange(len(number))
            transcript = spoken(
                number[:index]
                + rng.choice([d for d in "23456789" if d != number[index]])
                + number[index + 1 :]
            )
        elif kind == "transpose":
            index = rng.randrange(len(number) - 1)
            transcript = spoken(
                number[:index] + number[index + 1] + number[index] + number[index + 2 :]
            )
        elif kind == "drop":
            cut = rng.randrange(len(number))
            transcript = spoken(number[:cut] + number[cut + 1 :])
        elif kind == "add":
            at = rng.randrange(len(number) + 1)
            transcript = spoken(number[:at] + rng.choice("23456789") + number[at:])
        elif kind == "random":
            transcript = spoken(random_digits(rng, rng.randint(1, 20)))
        else:
            transcript = rng.choice(
                ("um", "four hundred", "double", "hold on a moment", "")
            )

        request = make_request(
            transcript=transcript,
            expected=number,
            voice_decision=rng.choice(list(Decision)),
        )
        answer = run_verification(request, ledger=AttemptLedger())
        secret = RedactedDigits(number)
        assert not leaks(answer.model_dump_json(), secret), (number, transcript)
        assert not leaks(repr(answer), secret)
        assert not leaks(repr(request), secret)
        assert not leaks(answer.details.reason, secret)
        assert not leaks(answer.combined.reason, secret)


def test_property_a_verified_result_always_needed_both_halves() -> None:
    rng = random.Random(29)
    for _ in range(200):
        number = random_digits(rng, rng.randint(NEAR_MISS_MIN_DIGITS, 17))
        voice = rng.choice(list(Decision))
        answer = run_verification(
            make_request(
                transcript=spoken(number), expected=number, voice_decision=voice
            ),
            ledger=AttemptLedger(),
        )
        if answer.combined.verdict is Verdict.VERIFIED:
            assert voice is Decision.ALLOW
            assert answer.details.verdict is Verdict.VERIFIED


# ---------------------------------------------------------- schema snapshot

SNAPSHOT = Path(__file__).resolve().parents[2] / "schemas" / "verification.schema.json"


def render(bundle: dict) -> str:
    return json.dumps(bundle, indent=2, sort_keys=True) + "\n"


def test_verification_schema_snapshot_matches() -> None:
    current = render(verification_schema_bundle())

    if os.environ.get("CALLBACK_UPDATE_SNAPSHOT") == "1":
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(current, encoding="utf-8")
        pytest.skip(f"snapshot rewritten at {SNAPSHOT}")

    assert SNAPSHOT.exists(), (
        f"no verification snapshot at {SNAPSHOT}. Create it with "
        "CALLBACK_UPDATE_SNAPSHOT=1 uv run pytest "
        "tests/contract/test_verification.py"
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
        policy_changed = current_obj["policy"] != recorded_obj["policy"]
        pytest.fail(
            "verification schema drifted from the snapshot.\n"
            f"  version recorded: {recorded_obj['verification_version']}\n"
            f"  version current : {current_obj['verification_version']}\n"
            f"  models added    : {added}\n"
            f"  models removed  : {removed}\n"
            f"  models changed  : {changed}\n"
            f"  enums changed   : {enum_changed}\n"
            f"  policy changed  : {policy_changed}\n"
            "If this change is intended, rewrite the snapshot with "
            "CALLBACK_UPDATE_SNAPSHOT=1 and bump VERIFICATION_VERSION."
        )


def test_snapshot_records_the_current_verification_version() -> None:
    assert SNAPSHOT.exists(), "run the snapshot test first"
    recorded = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert recorded["verification_version"] == VERIFICATION_VERSION


def test_snapshot_pins_the_product_vocabulary_and_the_thresholds() -> None:
    bundle = verification_schema_bundle()
    assert bundle["enums"]["Verdict"] == ["verified", "mismatch", "needs_review"]
    assert bundle["policy"]["near_miss_min_digits"] == NEAR_MISS_MIN_DIGITS
    assert bundle["policy"]["max_near_miss_edits"] == 1
    assert bundle["policy"]["leak_substring_digits"] == LEAK_SUBSTRING_DIGITS
    assert bundle["policy"]["verdict_by_match"]["exact"] == "verified"
    assert bundle["policy"]["disclosure_by_verdict"] == {
        "verified": "confirmed",
        "mismatch": "not_confirmed",
        "needs_review": "not_confirmed",
    }
    # The expected value is write only wherever it appears.
    assert (
        bundle["models"]["VerificationRequest"]["properties"]["expected"]["writeOnly"]
        is True
    )
    for model in bundle["models"].values():
        assert "expected" not in model.get("properties", {}) or model["properties"][
            "expected"
        ].get("writeOnly")
