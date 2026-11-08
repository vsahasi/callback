import pytest

from callback_agent.spoken import parse_spoken_digits, parse_yes_no


@pytest.mark.parametrize(
    "text, digits",
    [
        ("four two four two four two", "424242"),
        ("424 242", "424242"),
        ("4-2-4, 2-4-2", "424242"),
        ("um the code is 4 2 4 2 4 2", "424242"),
        ("oh one two", "012"),
        ("zero one two", "012"),
        ("double four two", "442"),
        ("triple seven", "777"),
        ("forty two", "42"),
        ("forty", "40"),
        ("Four. Two. Four.", "424"),
        ("I didn't get it", ""),
    ],
)
def test_parse_spoken_digits(text, digits):
    assert parse_spoken_digits(text) == digits


@pytest.mark.parametrize(
    "text, answer",
    [
        ("yes", True),
        ("Yeah, that was me.", True),
        ("correct", True),
        ("Yes, I agree", True),
        ("no", False),
        ("Nope, that wasn't me", False),
        ("I don't agree", False),
        ("what?", None),
        ("", None),
    ],
)
def test_parse_yes_no(text, answer):
    assert parse_yes_no(text) is answer


# --- Further cases -----------------------------------------------------------------------------
# Each one below is a case that a plausible wrong parser gets wrong while still passing the two
# tests above.


@pytest.mark.parametrize(
    "text, digits",
    [
        # The two tests above never say three, five, six, eight or nine, and skip most tens words.
        pytest.param(
            "zero one two three four five six seven eight nine", "0123456789", id="digit_words"
        ),
        pytest.param(
            "twenty thirty forty fifty sixty seventy eighty ninety",
            "2030405060708090",
            id="tens_words_alone",
        ),
        pytest.param(
            "twenty one thirty two forty three fifty four sixty five seventy six eighty seven "
            "ninety eight",
            "2132435465768798",
            id="tens_words_with_a_unit",
        ),
    ],
)
def test_every_word_the_parser_knows_is_heard(text, digits):
    assert parse_spoken_digits(text) == digits


@pytest.mark.parametrize(
    "text, digits",
    [
        pytest.param("4 two 4", "424", id="numeral_word_numeral"),
        pytest.param("one 2 three 45", "12345", id="words_and_numerals_mixed"),
        pytest.param("FOUR TWO", "42", id="upper_case"),
        pytest.param("Forty-Two!", "42", id="hyphenated_pair"),
        pytest.param("Oh... one, two?", "012", id="oh_with_punctuation"),
        pytest.param("", "", id="empty"),
    ],
)
def test_digits_from_mixed_and_messy_transcripts(text, digits):
    assert parse_spoken_digits(text) == digits


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("hello", id="hello"),
        pytest.param("no", id="no"),
        pytest.param("john went to ohio", id="oh_inside_john_and_ohio"),
        pytest.param("someone's weight", id="one_inside_someone_eight_inside_weight"),
    ],
)
def test_a_word_that_merely_contains_a_digit_word_is_not_a_digit(text):
    assert parse_spoken_digits(text) == ""


@pytest.mark.parametrize(
    "text, digits",
    [
        pytest.param("four two double", "42", id="double_at_the_end_repeats_nothing"),
        pytest.param("triple", "", id="triple_alone"),
        pytest.param("double check 4 2", "42", id="double_only_reaches_the_next_word"),
        pytest.param("double 4 2", "442", id="double_a_numeral"),
        pytest.param("double oh seven", "007", id="double_oh"),
        pytest.param("triple zero", "000", id="triple_zero"),
    ],
)
def test_double_and_triple_repeat_only_the_number_right_after_them(text, digits):
    assert parse_spoken_digits(text) == digits


@pytest.mark.parametrize(
    "text, digits",
    [
        pytest.param("twenty twenty", "2020", id="tens_then_tens"),
        pytest.param("forty oh five", "4005", id="tens_then_oh"),
        pytest.param("forty um two", "402", id="tens_then_filler_then_unit"),
    ],
)
def test_a_tens_word_pairs_only_with_a_unit_word_right_after_it(text, digits):
    # Anything else after a tens word is read on its own, so nothing is swallowed.
    assert parse_spoken_digits(text) == digits


@pytest.mark.parametrize(
    "word",
    [
        "yes",
        "yeah",
        "yep",
        "correct",
        "right",
        "sure",
        "agree",
        "confirm",
        "that was me",
        "yup",
        "agreed",
        "confirmed",
        "okay",
        "ok",
        "absolutely",
        "definitely",
        "consent",
        "it was me",
        "I did",
        "I do",
        "of course",
    ],
)
def test_every_listed_agreement_is_a_yes(word):
    assert parse_yes_no(word) is True


@pytest.mark.parametrize(
    "word",
    [
        "no",
        "nope",
        "not",
        "don't",
        "didn't",
        "wasn't",
        "never",
        "cannot",
        "refuse",
        "refused",
        "decline",
        "declined",
        "unable",
        "unwilling",
        "reject",
        "rejected",
        "deny",
        "denied",
    ],
)
def test_every_listed_negation_is_a_no(word):
    assert parse_yes_no(word) is False


@pytest.mark.parametrize(
    "text, answer",
    [
        # A negation anywhere wins, before or after the agreement.
        pytest.param("I'm not sure", False, id="not_sure"),
        pytest.param("Yes, actually no", False, id="yes_then_no"),
        # Typed text can carry a curly apostrophe (U+2019). Left as it is, the negation is missed
        # and "I don't agree" comes out as a yes.
        pytest.param("I don\u2019t agree", False, id="curly_apostrophe_dont"),
        pytest.param("That wasn\u2019t me", False, id="curly_apostrophe_wasnt"),
        # Whole words only: a word that merely contains a listed word is not that word. A
        # substring match would turn the first two into a no and the rest into a yes.
        pytest.param("Yes, I know", None, id="no_inside_know_is_not_a_no"),
        pytest.param("Yes, nothing else", None, id="not_inside_nothing_is_not_a_no"),
        pytest.param("unsure", None, id="sure_inside_unsure_is_not_a_yes"),
        pytest.param("I disagree", None, id="agree_inside_disagree_is_not_a_yes"),
        pytest.param("that's incorrect", None, id="correct_inside_incorrect_is_not_a_yes"),
        pytest.param("that was meh", None, id="that_was_me_needs_the_whole_word_me"),
        pytest.param("YES!", True, id="upper_case"),
    ],
)
def test_parse_yes_no_edge_cases(text, answer):
    assert parse_yes_no(text) is answer


# --- Negation by contraction -------------------------------------------------------------------
# Any word ending in "n't", and "cannot", counts as a negation, so these are a no (False) rather
# than just an answer that is not a yes (None).


@pytest.mark.parametrize(
    "text, answer",
    [
        pytest.param("I can't agree", False, id="cant_agree"),
        pytest.param("cannot confirm", False, id="cannot_confirm"),
        pytest.param("that isn't correct", False, id="isnt_correct"),
        pytest.param("I won't", False, id="wont"),
        # A curly apostrophe (U+2019) is the same as a straight one.
        pytest.param("I can\u2019t agree", False, id="curly_cant_agree"),
        pytest.param("that isn\u2019t correct", False, id="curly_isnt_correct"),
        # Agreement without a negation still passes.
        pytest.param("Yes, I agree", True, id="agree"),
        pytest.param("that was me", True, id="that_was_me"),
        # Other contractions are not negations, and neither is a word that merely ends in "nt".
        # "want" and "sent" are not allowed words, so that answer is None rather than a no.
        pytest.param("that's right", True, id="other_contractions_do_not_negate"),
        pytest.param("Yes, I want it sent", None, id="nt_at_the_end_of_a_word_does_not_negate"),
    ],
)
def test_parse_yes_no_negation_by_contraction(text, answer):
    assert parse_yes_no(text) is answer


@pytest.mark.parametrize(
    "word",
    [
        "can't",
        "won't",
        "isn't",
        "aren't",
        "doesn't",
        "couldn't",
        "wouldn't",
        "shouldn't",
        "haven't",
        "hasn't",
        "weren't",
        "ain't",
    ],
)
def test_every_n_t_contraction_is_a_no(word):
    assert parse_yes_no(word) is False


# --- An agreement word is not an agreement -----------------------------------------------------
# A yes is a short answer made only of core words and fillers, with no question mark and no
# negation. An answer that merely contains an agreement word, such as a refusal, a doubt or a
# question, is not a yes.


@pytest.mark.parametrize(
    "text",
    [
        "I refuse to confirm",
        "I decline to agree",
        "I'm unable to confirm that",
        "I'm unwilling to agree",
        "I have the right to refuse",
    ],
)
def test_a_refusal_that_contains_an_agreement_word_is_a_no(text):
    assert parse_yes_no(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "Are you sure?",
        "Is that correct?",
        "Is that right?",
        "Do I agree?",
        "Agree to what?",
        "Do I have to agree?",
        "Can you confirm that?",
        "Right, who is this?",
    ],
)
def test_a_question_is_not_a_yes(text):
    assert parse_yes_no(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "I'm sure it's a scam",
        "I'm sure someone else did that",
        "Let me confirm with my manager first",
        "I have nothing to confirm",
    ],
)
def test_a_statement_that_contains_an_agreement_word_is_not_a_yes(text):
    assert parse_yes_no(text) is None


def test_a_negation_still_makes_a_question_a_no():
    assert parse_yes_no("Why not?") is False


@pytest.mark.parametrize("text", ["Yes?", "That's right?"])
def test_a_question_mark_alone_is_enough_to_make_it_none(text):
    assert parse_yes_no(text) is None


def test_fillers_alone_are_not_a_yes():
    assert parse_yes_no("Um, well") is None


def test_a_yes_may_be_padded_with_fillers():
    assert parse_yes_no("Uh, yeah, of course, please go ahead") is True


@pytest.mark.parametrize(
    "filler",
    [
        "i",
        "i'm",
        "im",
        "it",
        "that",
        "that's",
        "thats",
        "was",
        "is",
        "me",
        "do",
        "did",
        "of",
        "course",
        "please",
        "uh",
        "um",
        "mm",
        "mhm",
        "hmm",
        "oh",
        "well",
        "so",
        "go",
        "ahead",
        "true",
    ],
)
def test_every_listed_filler_is_allowed_next_to_a_yes(filler):
    assert parse_yes_no(f"yes {filler}") is True


def test_a_yes_is_at_most_twelve_words():
    assert parse_yes_no(" ".join(["yes"] * 12)) is True
    assert parse_yes_no(" ".join(["yes"] * 13)) is None


def test_a_long_answer_with_a_negation_is_still_a_no():
    assert parse_yes_no(" ".join(["yes"] * 13 + ["not"])) is False


# --- Questions without a question mark, and short yeses ----------------------------------------
# An answer whose first word starts a question is a question even when the transcript has no "?".
# A negation still makes it a no.


@pytest.mark.parametrize(
    "text, answer",
    [
        pytest.param("Is that correct", None, id="is_that_correct"),
        pytest.param("Is that right", None, id="is_that_right"),
        pytest.param("Do I agree", None, id="do_i_agree"),
        pytest.param("Are you sure", None, id="are_you_sure"),
        pytest.param("Can you confirm that", None, id="can_you_confirm_that"),
        pytest.param("Is that correct, no", False, id="a_negation_still_makes_it_a_no"),
        # Only the first word counts, and "i" does not start a question, so these stay a yes.
        pytest.param("I do", True, id="i_do"),
        pytest.param("I did", True, id="i_did"),
        pytest.param("that is correct", True, id="that_is_correct"),
        pytest.param("that's right", True, id="thats_right"),
    ],
)
def test_a_question_without_a_question_mark_is_not_a_yes(text, answer):
    assert parse_yes_no(text) is answer


@pytest.mark.parametrize(
    "word",
    [
        "is",
        "are",
        "do",
        "does",
        "can",
        "could",
        "would",
        "will",
        "should",
        "did",
        "have",
        "has",
        "was",
        "were",
    ],
)
def test_an_answer_that_starts_with_a_question_word_is_none(word):
    # Only is, do, did and was are also fillers. The others fail the allowlist anyway.
    assert parse_yes_no(f"{word} yes") is None


@pytest.mark.parametrize(
    "text, answer",
    [
        pytest.param("I consent", True, id="i_consent"),
        pytest.param("I'm sure", True, id="im_sure"),
        pytest.param("Im sure", True, id="im_sure_without_the_apostrophe"),
        pytest.param("I'm sure it's a scam", None, id="im_sure_it_is_a_scam"),
    ],
)
def test_a_short_yes_may_use_consent_and_im(text, answer):
    assert parse_yes_no(text) is answer


# --- A question word after a filler, and "I'm okay" --------------------------------------------
# A question word still gives a question away after a leading filler ("So is that right"), and
# right before that, it, I or you ("Yes, is that correct"). "I'm okay" can mean "no thanks".


@pytest.mark.parametrize(
    "text, answer",
    [
        pytest.param("So is that right", None, id="so_is_that_right"),
        pytest.param("Well, do I agree", None, id="well_do_i_agree"),
        pytest.param("Yes, is that correct", None, id="yes_is_that_correct"),
        pytest.param("Oh, is that right", None, id="oh_is_that_right"),
        pytest.param("Um, uh, do I agree", None, id="two_fillers_then_do_i_agree"),
        pytest.param("Yes, was that me", None, id="yes_was_that_me"),
        # Here "is" is not before that, it, I or you, so only the leading filler gives it away.
        pytest.param("Um, is yes", None, id="a_question_word_after_um"),
        # Only a leading filler is skipped, never "that" or "I", so these stay a yes.
        pytest.param("that is correct", True, id="that_is_correct"),
        pytest.param("that's right", True, id="thats_right"),
        pytest.param("I do", True, id="i_do"),
        pytest.param("Well, I do", True, id="well_i_do"),
        pytest.param("Oh, that is correct", True, id="oh_that_is_correct"),
    ],
)
def test_parse_yes_no_question_word_after_a_filler_or_a_yes(text, answer):
    assert parse_yes_no(text) is answer


@pytest.mark.parametrize("question_word", ["is", "do", "did", "was"])
@pytest.mark.parametrize("subject", ["that", "it", "i"])
def test_a_question_word_right_before_its_subject_is_none(question_word, subject):
    # "you" and the other question words are not allowed words, so the allowlist already says None.
    assert parse_yes_no(f"yes {question_word} {subject}") is None


@pytest.mark.parametrize(
    "text, answer",
    [
        pytest.param("I'm okay", None, id="im_okay"),
        pytest.param("im okay", None, id="im_okay_without_the_apostrophe"),
        pytest.param("I'm ok", None, id="im_ok"),
        # "am" is not an allowed word, so this one is None for that reason.
        pytest.param("I am okay", None, id="i_am_okay"),
        pytest.param("okay", True, id="okay"),
        # "okay" is only a problem right after "I'm".
        pytest.param("Okay, I'm sure", True, id="okay_then_im_sure"),
        pytest.param("I'm sure, okay", True, id="im_sure_then_okay"),
    ],
)
def test_im_okay_is_not_a_yes(text, answer):
    assert parse_yes_no(text) is answer


# --- "That's okay" is not a yes (R80) ------------------------------------------------------------
# It can mean "no thanks, that's okay", and it answers none of the questions asked here ("Did you
# just ask me to ...?", "Do you agree?"). Plain "okay" is still a yes, and so is every answer that
# was one before.

WAVED_OFF = [
    "that's okay",
    "that is okay",
    "it's okay",
    "it is okay",
    "that's ok",
    "that is ok",
    "it's ok",
    "it is ok",
    "thats okay",
    "thats ok",
    "its okay",
    "its ok",
]


@pytest.mark.parametrize("text", WAVED_OFF)
def test_that_is_okay_is_not_a_yes(text):
    assert parse_yes_no(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "That's okay.",
        "THAT IS OK!",
        "That\u2019s okay",  # a curly apostrophe (U+2019) reads as a straight one
        "Oh, that's okay",
        "Um, it is okay",
        "Yes, that's okay",  # the same words anywhere in the answer, as for "I'm okay"
        "okay, that's okay",
    ],
)
def test_it_is_not_a_yes_whatever_surrounds_it_or_however_it_is_cased_and_punctuated(text):
    assert parse_yes_no(text) is None


@pytest.mark.parametrize(
    "text, answer",
    [
        pytest.param("okay", True, id="okay"),
        pytest.param("OK", True, id="ok"),
        pytest.param("Okay.", True, id="okay_with_a_full_stop"),
        pytest.param("okay, it was me", True, id="okay_then_it_was_me"),
        pytest.param("it was okay", True, id="it_was_okay"),  # "was", not "is"
        pytest.param("okay, that is correct", True, id="okay_then_that_is_correct"),
        pytest.param("that's right", True, id="thats_right"),
        pytest.param("yes it is", True, id="yes_it_is"),
        pytest.param("that is correct, okay", True, id="that_is_correct_then_okay"),
        # a negation is still a no, so "no, that's okay" is the no it sounds like
        pytest.param("No, that's okay", False, id="no_thats_okay"),
        pytest.param("that's not okay", False, id="thats_not_okay"),
    ],
)
def test_okay_on_its_own_and_every_other_yes_stay_as_they_were(text, answer):
    assert parse_yes_no(text) is answer


# --- "oh" is a zero only next to a digit (ledger :147) -----------------------------------------
# A natural "Oh, it's 4 2 4 2 4 2" must not read as seven digits and cost one of the three tries.


@pytest.mark.parametrize(
    "text, digits",
    [
        pytest.param("Oh okay it's 4 2 4 2 4 2", "424242", id="oh_to_start_with"),
        pytest.param("Oh, it's four two four two four two", "424242", id="oh_comma_then_words"),
        pytest.param("oh well, 4 2 4 2 4 2", "424242", id="oh_well"),
        pytest.param("four oh two", "402", id="oh_between_digit_words"),
        pytest.param("4 oh 2", "402", id="oh_between_numerals"),
        pytest.param("4 o 2", "402", id="o_between_numerals"),
        pytest.param("four o two", "402", id="o_between_digit_words"),
        pytest.param("oh four two", "042", id="oh_before_a_digit"),
        pytest.param("four two oh", "420", id="oh_after_a_digit"),
        pytest.param("oh 4 2", "042", id="oh_before_a_numeral"),
        pytest.param("4 2 O", "420", id="a_capital_o"),
        pytest.param("one, oh, two", "102", id="oh_with_commas_between"),
        pytest.param("one oh zero", "100", id="oh_then_zero"),
        pytest.param("zero oh", "00", id="oh_after_zero"),
    ],
)
def test_oh_and_o_are_zero_next_to_a_digit(text, digits):
    assert parse_spoken_digits(text) == digits


@pytest.mark.parametrize(
    "text",
    ["oh", "o", "Oh.", "oh okay", "oh well, never mind", "oh no", "o k", "oh, hello there"],
)
def test_oh_with_no_digit_next_to_it_is_dropped(text):
    assert parse_spoken_digits(text) == ""


@pytest.mark.parametrize(
    "text, digits",
    [
        pytest.param("oh oh seven", "007", id="a_run_before_a_digit"),
        pytest.param("seven oh oh", "700", id="a_run_after_a_digit"),
        pytest.param("four oh oh two", "4002", id="a_run_between_digits"),
        pytest.param("oh oh okay it's 4 2 4 2 4 2", "424242", id="a_run_with_no_digit_next_to_it"),
        pytest.param("oh oh", "", id="a_run_alone"),
        pytest.param("oh well oh 7", "07", id="only_the_run_next_to_a_digit_counts"),
        pytest.param("forty oh five", "4005", id="oh_next_to_a_tens_word_and_a_digit"),
        pytest.param("oh forty two", "042", id="oh_before_a_tens_word"),
    ],
)
def test_a_run_of_ohs_counts_together_when_a_digit_is_next_to_the_run(text, digits):
    assert parse_spoken_digits(text) == digits


def test_a_dropped_oh_cancels_double_like_any_other_word():
    # "double" repeats only the number right after it, and a dropped word is not that number.
    assert parse_spoken_digits("double oh okay 7") == "7"
    assert parse_spoken_digits("double oh 7") == "007"


@pytest.mark.parametrize(
    "text, digits",
    [
        pytest.param("ten", "", id="ten"),
        pytest.param("fourteen", "", id="a_teen"),
        pytest.param("eleven twelve", "", id="two_teens"),
        pytest.param("nineteen", "", id="nineteen_is_not_nine"),
        pytest.param("hundred", "", id="hundred"),
        pytest.param("one hundred", "1", id="only_the_one_of_one_hundred"),
        pytest.param("four ten two", "42", id="ten_between_digits_is_dropped"),
    ],
)
def test_ten_the_teens_and_hundred_are_not_parsed(text, digits):
    # The docstring says so: they are dropped like any other word, and what is around them stays.
    assert parse_spoken_digits(text) == digits
