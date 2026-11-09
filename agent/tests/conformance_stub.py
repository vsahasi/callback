"""A stand-in detector that replays the answers of section 5 of the final contract ruling.

It is not a model of the detector. Each scenario has its answers written down, as the ruling lists
them: the authorize decisions in order, and how each challenge ends. It exists so the conformance
cases (conformance_support.py) can be run, and their logic proved, with no detector up. It also
checks what the agent sends: the bearer token, the contract version header on every request, and
the scenario header on session create only.

The one rule it keeps is the one a case needs: after a deny, a tier 1 or 2 authorize answers with a
deny that names session_locked, and a hold is always an allow.
"""

import itertools
import json
from collections import defaultdict
from dataclasses import dataclass, field

import httpx

TOKEN = "stub-token"
# The version this fake claims to serve. A literal, not an import from
# callback_shared, because a fake that reads the real constant could never
# catch the agent sending the wrong one. test_conformance_stub.py asserts the
# two agree, so a contract bump fails there and points at this line.
CONTRACT_VERSION = "1.2.0"
USER = {
    "user_id": "u_eric",
    "display_name": "Eric Catalano",
    "role": "CFO",
    "phone_last_four": "0100",
    "phone_e164": "+15555550100",
}
OTP_CODE = "424242"
PHRASE = "4 7 2, 9 1 3, amber, pillow, rocket"
DIGITS = "4 7 2, 9 1 3"
ENROLL = "Please repeat after me: 4 7 2, 9 1 3."

ALL_FACTORS = ["voice_challenge", "sms_otp", "callback_on_file"]
_TIER = {
    "hold_vendor": 0,
    "hold_bill": 0,
    "list_vendors": 1,
    "list_vendor_accounts": 1,
    "list_bills": 1,
}


def _decision(kind: str, factors: list[str], reasons: list[str], ttl_s: int) -> dict:
    return {"decision": kind, "required_factors": factors, "reasons": reasons, "ttl_s": ttl_s}


def _deny(code: str, *lines: str) -> dict:
    return _decision("deny", [], [code, *lines], 0)


# The voiceprint alone would have passed: the deny is for the synthetic speech.
_CLONE_DENY = {
    **_deny("synthetic_speech", "spoof_high 0.92", "voiceprint_pass 0.78≥0.62"),
    "signals": {"voiceprint": {"score": 0.78, "threshold": 0.62, "passed": True}},
}

# The authorize answers by (scenario, tier), in order, as section 5 gives them.
SCRIPT: dict[tuple[str, int], list[dict]] = {
    ("real_cfo", 2): [
        _decision("step_up", ALL_FACTORS, ["voiceprint_pass 0.83≥0.62", "spoof_low 0.04"], 120),
        _decision("step_up", ["voice_challenge"], ["voiceprint_pass 0.83≥0.62"], 120),
        _decision(
            "allow", [], ["voiceprint_pass 0.83≥0.62", "spoof_low 0.04", "text_low 0.10"], 60
        ),
    ],
    ("real_cfo", 1): [_decision("allow", [], ["voiceprint_pass 0.83≥0.62"], 300)],
    ("cloned_voice", 2): [_CLONE_DENY],
    ("cloned_voice", 1): [_CLONE_DENY],
    ("replay", 2): [
        _decision("step_up", ALL_FACTORS, ["voiceprint_grey 0.50", "spoof_mid 0.45"], 120),
        _deny("challenge_failed_replay", "phrase_match false", "spoof_mid 0.45"),
    ],
    ("replay", 1): [
        _decision("step_up", ["voice_challenge"], ["spoof_mid 0.45"], 120),
        _deny("challenge_failed_replay", "phrase_match false", "spoof_mid 0.45"),
    ],
    ("ai_agent", 2): [_deny("automated_caller", "spoof_high 0.85", "agent_likelihood 0.90")],
    ("ai_agent", 1): [_deny("automated_caller", "spoof_high 0.85", "agent_likelihood 0.90")],
    ("human_impostor", 2): [
        _decision("step_up", ALL_FACTORS, ["voiceprint_fail 0.29<0.45"], 120),
        _deny(
            "speaker_mismatch",
            "voiceprint_fail 0.29<0.45",
            "text_flags: urgency, verification_refusal",
        ),
    ],
    ("human_impostor", 1): [
        _decision("step_up", ["voice_challenge"], ["voiceprint_fail 0.29<0.45"], 120),
        _deny(
            "speaker_mismatch",
            "voiceprint_fail 0.29<0.45",
            "text_flags: urgency, verification_refusal",
        ),
    ],
}

# How a challenge ends, by scenario.
_PASSED = {"status": "passed", "phrase_match": True, "speaker_score": 0.83}
OUTCOME = {
    "real_cfo": _PASSED,
    "cloned_voice": {**_PASSED, "speaker_score": 0.78},
    "ai_agent": {**_PASSED, "speaker_score": 0.70},
    "replay": {
        "status": "failed",
        "phrase_match": False,
        "speaker_score": 0.80,
        "reason": "phrase_mismatch",
    },
    "human_impostor": {
        "status": "failed",
        "phrase_match": True,
        "speaker_score": 0.29,
        "reason": "speaker_mismatch",
    },
}


@dataclass
class Session:
    scenario: str
    channel: str
    locked: str | None = None
    authorizes: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    polls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    challenge_kinds: dict[str, str] = field(default_factory=dict)
    enrollment_prompts: int = 0
    user_turns: int = 0  # user turns accepted so far
    issued_at_turn: dict[str, int] = field(default_factory=dict)  # user_turns when each was issued
    current: str | None = None  # the pending challenge, if any
    refused: int = 0  # challenges that ended unanswered (superseded with no user turn after them)


class StubDetector:
    """Handle requests as `httpx.MockTransport(stub.handle)`. `log` has each one as (method, path
    template), and `problems` lists anything the agent sent that breaks the contract."""

    def __init__(self, *, token: str = TOKEN) -> None:
        self.token = token
        self.log: list[tuple[str, str]] = []
        self.problems: list[str] = []
        self.sessions: dict[str, Session] = {}
        self.turn_texts: list[str] = []  # the text of each user turn the detector accepted
        self.fail_turns = False  # answer every /turns with a 500, so the forwarder drops them
        self.model_refusals = False  # apply the refused-challenge rule instead of the script
        self._ids = itertools.count(1)
        self._challenges = itertools.count(1)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handle))

    def sequence(self, *, turns: bool = False) -> list[str]:
        """The log as 'METHOD template', with consecutive polls of one challenge folded to one.

        Turns are left out unless asked for: the forwarder posts them in the background, so where
        they fall between the other calls is not fixed.
        """
        out: list[str] = []
        for method, template in self.log:
            if template.endswith("/turns") and not turns:
                continue
            entry = f"{method} {template}"
            if not (out and out[-1] == entry and template.endswith("{cid}")):
                out.append(entry)
        return out

    # --- the routes ------------------------------------------------------------------------

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.headers.get("authorization") != f"Bearer {self.token}":
            return _error(401, "unauthorized")
        if request.headers.get("x-contract-version") != CONTRACT_VERSION:
            return _error(400, "contract_version_mismatch")
        parts = request.url.path.strip("/").split("/")
        body = json.loads(request.content) if request.content else None
        method = request.method
        has_scenario = "x-callback-scenario" in request.headers
        if has_scenario and parts[1:] != ["sessions"]:
            self.problems.append(f"scenario header on {request.url.path}")
        template = _template(parts)
        self.log.append((method, template))
        match (method, template):
            case ("GET", "/v1/users"):
                name = request.url.params.get("name", "")
                return _json([USER_SUMMARY] if name == "Eric Catalano" else [])
            case ("GET", "/v1/users/{id}"):
                return _json(USER) if parts[2] == "u_eric" else _error(404, "user_not_found")
            case ("POST", "/v1/sessions"):
                return self._create(request, body)
            case ("POST", "/v1/enrollments"):
                return _json({"enrollment_id": "enr_" + "a" * 32, "prompts_required": 7})
            case ("POST", "/v1/enrollments/{id}/complete"):
                return _json({"status": "enrolled", "prompts_passed": 7})
        session = self.sessions.get(parts[2]) if len(parts) > 2 else None
        if session is None:
            return _error(404, "session_not_found")
        return self._session_route(session, method, template, parts, body)

    def _create(self, request: httpx.Request, body: dict) -> httpx.Response:
        scenario = request.headers.get("x-callback-scenario", "real_cfo")
        if body["claimed_user_id"] != "u_eric":
            return _error(404, "user_not_found")
        session_id = f"ses_{next(self._ids):032x}"
        self.sessions[session_id] = Session(scenario, body["channel"])
        return _json({"session_id": session_id})

    def _session_route(self, s, method, template, parts, body) -> httpx.Response:
        match (method, template):
            case ("POST", "/v1/sessions/{id}/authorize"):
                return self._authorize(s, parts[2], body)
            case ("POST", "/v1/sessions/{id}/challenges"):
                cid = f"chl_{next(self._challenges):016x}"
                kind = body["kind"]
                s.challenge_kinds[cid] = kind
                if s.current is not None and s.issued_at_turn[s.current] == s.user_turns:
                    s.refused += 1  # superseded, and no user turn came after it (section 2.7)
                s.issued_at_turn[cid] = s.user_turns
                s.current = cid
                say = {"phrase": PHRASE, "digits": DIGITS, "enroll": ENROLL}.get(kind, DIGITS)
                return _json({"challenge_id": cid, "say": say, "expires_at": 1_700_000_045_000})
            case ("GET", "/v1/sessions/{id}/challenges/{cid}"):
                return self._poll(s, parts[4])
            case ("POST", "/v1/sessions/{id}/otp/send"):
                return _json({"sent_to_last_four": "0100"})
            case ("POST", "/v1/sessions/{id}/otp/check"):
                return _json({"verified": body["code"] == OTP_CODE})
            case (
                "POST",
                "/v1/sessions/{id}/caller"
                | "/v1/sessions/{id}/turns"
                | "/v1/sessions/{id}/events"
                | "/v1/sessions/{id}/factors"
                | "/v1/sessions/{id}/end",
            ):
                if template.endswith("/turns"):
                    if self.fail_turns:
                        return _error(500, "internal_error")
                    s.user_turns += body["role"] == "user"
                    self._check_turn(body)
                return httpx.Response(204)
        return _error(404, "session_not_found")

    def _authorize(self, s: Session, session_id: str, body: dict) -> httpx.Response:
        action = body["action"]
        tier = _TIER.get(action, 2)
        if tier == 0:
            return _decide(session_id, action, _decision("allow", [], ["tier0_hold"], 0))
        if s.locked is not None:
            decision = _deny("session_locked", s.locked)
        elif self.model_refusals:
            decision = self._by_refusals(s, tier)
        else:
            script = SCRIPT[(s.scenario, tier)]
            decision = script[min(s.authorizes[tier], len(script) - 1)]
            s.authorizes[tier] += 1
            if decision["decision"] == "deny":
                s.locked = decision["reasons"][0]
        return _decide(session_id, action, decision)

    def _by_refusals(self, s: Session, tier: int) -> dict:
        """The detector with no answers to its challenges: two refused ones are a deny."""
        if s.refused >= 2:
            s.locked = "challenges_refused"
            return _deny("challenges_refused")
        return _decision("step_up", ALL_FACTORS if tier == 2 else ["voice_challenge"], [], 120)

    def _poll(self, s: Session, cid: str) -> httpx.Response:
        # A challenge settles only after a user turn was received after it was issued (2.7).
        if s.user_turns == s.issued_at_turn[cid]:
            return _json({"status": "pending"})
        s.polls[cid] += 1
        if s.polls[cid] == 1:
            return _json({"status": "pending"})
        if s.challenge_kinds.get(cid) == "enroll":
            return _json({"status": "passed", "phrase_match": True, "speaker_score": 0.83})
        return _json(OUTCOME[s.scenario])

    def _check_turn(self, turn: dict) -> None:
        # A challenge answer keeps its spoken digits. Anything else with 4 or more digit words
        # must have been redacted to [NUM] before it got here.
        self.turn_texts.append(turn["text"])
        if turn["text"] == "four two four two four two":
            self.problems.append("an OTP answer was forwarded unredacted")


USER_SUMMARY = {k: USER[k] for k in ("user_id", "display_name", "role", "phone_last_four")}


def _template(parts: list[str]) -> str:
    out = []
    for i, part in enumerate(parts):
        if i == 2 and parts[1] in ("users", "sessions", "enrollments"):
            out.append("{id}")
        elif i == 4 and parts[3] == "challenges":
            out.append("{cid}")
        else:
            out.append(part)
    return "/" + "/".join(out)


def _json(payload) -> httpx.Response:
    return httpx.Response(200, json=payload, headers={"X-Contract-Version": CONTRACT_VERSION})


def _error(status: int, code: str) -> httpx.Response:
    body = {"contract_version": CONTRACT_VERSION, "error": {"code": code, "message": code}}
    return httpx.Response(status, json=body, headers={"X-Contract-Version": CONTRACT_VERSION})


def _decide(session_id: str, action: str, decision: dict) -> httpx.Response:
    return _json({"session_id": session_id, "action": action, **decision, "agent_say": ""})
