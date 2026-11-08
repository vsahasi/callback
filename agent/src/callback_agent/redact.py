"""Redact numbers in a turn before it is forwarded: no bank number reaches the detector.

People read account and routing numbers in groups, with filler between, and Flux can split a number
across turns. So a rule by run length leaks, and the rule here does not look at length:
- Outside a challenge answer, every numeral becomes [NUM], whatever its length, and so does every
  run of two or more spoken digit words ("zero two one", "oh four two", "double four five"). A
  single digit word ("one") stays, so ordinary speech survives.
- In the caller's answer to a voice challenge or an enroll prompt (`exempt_short_runs`), the digits
  are the challenge itself and are forwarded as spoken. Only a run of 7 or more digits, which is
  never a challenge, is redacted there.
- The agent's own turns get no exemption.

A run is a stretch of digit tokens (numerals, digit words, "oh" next to a digit, tens words, "double
X" and "triple X", the vocabulary of spoken.py) separated only by whitespace and `, . - / #`. In
`words`, an entry that holds a numeral is a number whatever else is attached to it (`$12,345`,
`#021000021`, `000123456789?`), and an entry made of digit words only joins the run next to it. One
redacted run is one `[NUM]` word, with the first word's start and the last word's end, so the word
count and timings stay aligned. This is a plain function of the text: no model is involved.
"""

from callback_shared.contract import Turn, Word

from callback_agent.spoken import DigitToken, digit_tokens, word_spans

REDACTED = "[NUM]"
_SEPARATORS = " \t\r\n-,./#"
_BANK_SIZED = 7  # digits: in a challenge answer, a run this long is never the challenge
# Words that can sit in a run without being a digit token on their own: "oh" and "o" next to a
# digit, and "double" and "triple" before one. digit_tokens decides whether they count.
_DIGIT_LIKE = {"oh", "o", "double", "triple"}
_REPEATS = {"double", "triple"}
_BREAK = "\0"  # stands for a word that is no digit token, in a list of words


class _Run:
    """Adjacent digit tokens: where they start and end (token indexes) and what they hold."""

    def __init__(self, index: int) -> None:
        self.first = self.last = index
        self.digits = 0
        self.tokens = 0
        self.numeral = False
        self.repeat = False


def _runs(tokens: list[DigitToken], words: list[str], joined) -> list[tuple[_Run, bool]]:
    """Group adjacent tokens into runs, and say for each whether the non-exempt rule redacts it.

    Two tokens are in one run when no other word lies between them (the second starts where the
    first ends) and `joined` says only separators do.
    """
    runs: list[_Run] = []
    for index, token in enumerate(tokens):
        previous = tokens[runs[-1].last] if runs else None
        if previous is not None and previous.end == token.start and joined(previous, token):
            run = runs[-1]
            run.last = index
        else:
            run = _Run(index)
            runs.append(run)
        run.digits += token.digits
        run.tokens += 1
        span = words[token.start : token.end]
        run.numeral = run.numeral or any(w.isdigit() for w in span)
        run.repeat = run.repeat or span[0] in _REPEATS
    return [(run, run.numeral or run.tokens >= 2 or run.repeat) for run in runs]


def _redacts(run: _Run, always: bool, exempt_short_runs: bool) -> bool:
    return run.digits >= _BANK_SIZED if exempt_short_runs else always


def _redact_text(text: str, exempt_short_runs: bool) -> str:
    spans = word_spans(text)
    words = [w for w, _, _ in spans]
    tokens = digit_tokens(words)

    def joined(a: DigitToken, b: DigitToken) -> bool:
        gap = text[spans[a.end - 1][2] : spans[b.start][1]]
        return not gap.strip(_SEPARATORS)

    pieces: list[str] = []
    cursor = 0
    for run, always in _runs(tokens, words, joined):
        if not _redacts(run, always, exempt_short_runs):
            continue
        start = spans[tokens[run.first].start][1]
        end = spans[tokens[run.last].end - 1][2]
        pieces.append(text[cursor:start])
        pieces.append(REDACTED)
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _redact_words(words: list[Word], exempt_short_runs: bool) -> list[Word]:
    # One flat list of words over all the entries. An entry that holds a numeral adds its numerals,
    # whatever else is on it. An entry of digit words only adds those words. Any other entry adds a
    # break, and an entry of only punctuation adds nothing. `entry_of` maps a flat word back to
    # its entry.
    flat: list[str] = []
    entry_of: list[int] = []
    for index, word in enumerate(words):
        names = [w for w, _, _ in word_spans(word.w)]
        added: list[str]
        if any(n.isdigit() for n in names):
            added = [n for n in names if n.isdigit()]
        elif names and all(n in _DIGIT_LIKE or digit_tokens([n]) for n in names):
            added = names
        elif names or any(c.isalpha() for c in word.w):
            added = [_BREAK]
        else:
            continue
        flat.extend(added)
        entry_of.extend([index] * len(added))
    tokens = digit_tokens(flat)
    out: list[Word] = []
    cursor = 0
    for run, always in _runs(tokens, flat, lambda a, b: True):
        if not _redacts(run, always, exempt_short_runs):
            continue
        start = entry_of[tokens[run.first].start]
        end = entry_of[tokens[run.last].end - 1]
        out.extend(words[cursor:start])
        out.append(Word(w=REDACTED, start_ms=words[start].start_ms, end_ms=words[end].end_ms))
        cursor = end + 1
    out.extend(words[cursor:])
    return out


def redact_turn(turn: Turn, *, exempt_short_runs: bool) -> Turn:
    """`turn` with every number that must not leave the agent replaced by [NUM].

    `exempt_short_runs` is for the caller's answer to a secret ask, and for nothing else: it lets
    the challenge's own digits through, and still redacts a run of 7 or more.
    """
    return Turn(
        role=turn.role,
        text=_redact_text(turn.text, exempt_short_runs),
        start_ms=turn.start_ms,
        end_ms=turn.end_ms,
        words=_redact_words(turn.words, exempt_short_runs),
        interrupted=turn.interrupted,
    )
