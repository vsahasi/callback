"""The opt-in local transcript: what it writes, what it never writes, and when it is off."""

import json
import logging

import pytest
from callback_shared.contract import Role, Turn, Word
from livekit.agents import ConversationItemAddedEvent
from livekit.agents.llm import ChatMessage, FunctionCall, FunctionCallOutput
from livekit.agents.voice.events import FunctionToolsExecutedEvent
from test_session_wiring import CODE, FakeForwarder, FakeSession, new_state

from callback_agent.session_wiring import wire_turn_forwarding
from callback_agent.transcript import TRANSCRIPT_DIR_ENV, Transcript, open_transcript


def records(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


# --- on or off --------------------------------------------------------------------------------


@pytest.mark.parametrize("env", [{}, {TRANSCRIPT_DIR_ENV: ""}, {TRANSCRIPT_DIR_ENV: "   "}])
def test_there_is_no_transcript_unless_a_directory_is_named(env):
    assert open_transcript(env, "call-_+15555550100_abc") is None


def test_each_call_gets_its_own_file_in_the_named_directory(tmp_path):
    transcript = open_transcript({TRANSCRIPT_DIR_ENV: str(tmp_path / "t")}, "call-_+1628_abc")
    assert transcript is not None
    assert transcript.path.parent == tmp_path / "t"
    assert transcript.path.name.endswith("call-_1628_abc.jsonl")  # the + is not kept
    assert transcript.path.parent.is_dir()


# --- turns ------------------------------------------------------------------------------------


def test_a_turn_is_written_with_every_number_redacted(tmp_path):
    transcript = Transcript(tmp_path / "call.jsonl")
    turn = Turn(
        role=Role.USER,
        text="routing 021000021 and my code is four two four two four two",
        start_ms=1_000,
        end_ms=4_000,
        words=[Word(w="routing", start_ms=1_000, end_ms=1_300)],
    )
    transcript.turn(turn)
    [record] = records(transcript.path)
    assert record == {
        "kind": "turn",
        "role": "user",
        "start_ms": 1_000,
        "end_ms": 4_000,
        "interrupted": False,
        "text": "routing [NUM] and my code is [NUM]",
        "latency_ms": {},
    }


def test_a_turn_carries_the_latency_livekit_measured_in_milliseconds(tmp_path):
    transcript = Transcript(tmp_path / "call.jsonl")
    turn = Turn(role=Role.AGENT, text="One moment.", start_ms=1_000, end_ms=2_000)
    metrics = {
        "started_speaking_at": 1.0,
        "llm_node_ttft": 0.7124,
        "tts_node_ttfb": 0.2,
        "end_of_turn_delay": 0.5,
        "provider_request_ids": ["req_1"],
    }
    transcript.turn(turn, metrics)
    [record] = records(transcript.path)
    assert record["latency_ms"] == {
        "llm_node_ttft": 712,
        "tts_node_ttfb": 200,
        "end_of_turn_delay": 500,
    }


# --- tools ------------------------------------------------------------------------------------


def test_tool_calls_are_written_by_name_and_duration_without_their_arguments(tmp_path):
    transcript = Transcript(tmp_path / "call.jsonl")
    call = FunctionCall(
        call_id="c1",
        name="change_vendor_bank_account",
        arguments='{"routing_number": "021000021", "account_number": "000123456789"}',
        created_at=100.0,
    )
    output = FunctionCallOutput(call_id="c1", name=call.name, output="ending 6789", is_error=False)
    event = FunctionToolsExecutedEvent(
        function_calls=[call], function_call_outputs=[output], created_at=101.25
    )
    transcript.tools(event)
    text = transcript.path.read_text()
    assert records(transcript.path) == [
        {
            "kind": "tools",
            "at_ms": 101_250,
            "calls": [{"name": "change_vendor_bank_account", "ms": 1_250, "error": False}],
        }
    ]
    assert "021000021" not in text and "6789" not in text


# --- failure ----------------------------------------------------------------------------------


def test_a_failed_write_is_logged_and_never_raised(tmp_path, caplog):
    transcript = Transcript(tmp_path / "missing-dir" / "call.jsonl")
    with caplog.at_level(logging.WARNING, logger="callback_agent.transcript"):
        transcript.turn(Turn(role=Role.USER, text="hello", start_ms=1, end_ms=2))
    assert "could not write the transcript" in caplog.text


# --- wired into the session -------------------------------------------------------------------


def test_the_session_writes_a_redacted_transcript_even_of_an_exempt_answer(tmp_path):
    # The caller's answer to a secret ask reaches the detector with its digits (it is the code
    # being checked). The transcript never keeps them.
    session, state = FakeSession(), new_state(answer_exempt=True)
    forwarder = FakeForwarder()
    transcript = Transcript(tmp_path / "call.jsonl")
    wire_turn_forwarding(session, state, forwarder, transcript=transcript)
    state.pending_segments.append(CODE)
    message = ChatMessage(role="user", content=["my code is 482913"])
    session.fire("conversation_item_added", ConversationItemAddedEvent(item=message))
    assert "482913" in forwarder.sent[0].text
    [record] = records(transcript.path)
    assert record["role"] == "user" and record["text"] == "my code is [NUM]"


def test_the_session_writes_its_tool_calls(tmp_path):
    session, state = FakeSession(), new_state()
    transcript = Transcript(tmp_path / "call.jsonl")
    wire_turn_forwarding(session, state, FakeForwarder(), transcript=transcript)
    call = FunctionCall(call_id="c1", name="identify_caller", arguments="{}", created_at=10.0)
    output = FunctionCallOutput(call_id="c1", name=call.name, output="ok", is_error=False)
    session.fire(
        "function_tools_executed",
        FunctionToolsExecutedEvent(
            function_calls=[call], function_call_outputs=[output], created_at=12.0
        ),
    )
    assert records(transcript.path)[0]["calls"] == [
        {"name": "identify_caller", "ms": 2_000, "error": False}
    ]


def test_without_a_transcript_the_session_listens_for_no_tool_calls():
    session, state = FakeSession(), new_state()
    wire_turn_forwarding(session, state, FakeForwarder())
    assert session.handlers["function_tools_executed"] == []
