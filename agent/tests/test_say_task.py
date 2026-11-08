"""LiveKitVoice.say, the gate's Speaker, in a roomless text session with the copied FakeLLM.

The gate speaks from inside a tool, so here a tool awaits the line too. The session is driven the
way the repo's test_nested_agent_task.py drives it.
"""

import asyncio
import logging

from fake_llm import FakeLLM, FakeLLMResponse
from livekit.agents import Agent, AgentSession, function_tool
from livekit.agents.llm import ChatMessage, FunctionToolCall
from stuck_output import StuckTranscription

from callback_agent.capture import LiveKitVoice

LINE = "hello"


def fake_llm() -> FakeLLM:
    speak = FunctionToolCall(name="speak", arguments="{}", call_id="call_1")
    return FakeLLM(
        fake_responses=[
            FakeLLMResponse(input="go", content="", ttft=0, duration=0, tool_calls=[speak]),
            # The answer to the tool's output, so it can only be said once the tool has returned.
            FakeLLMResponse(input="said", content="ok", ttft=0, duration=0),
        ]
    )


class SpeakingAgent(Agent):
    """Its one tool says LINE through LiveKitVoice, the way the gate does."""

    def __init__(self, **voice_options) -> None:
        super().__init__(instructions="test")
        self._voice_options = voice_options
        self.returned = asyncio.Event()

    @function_tool
    async def speak(self) -> str:
        """Say the line."""
        await LiveKitVoice(self.session, **self._voice_options).say(LINE)
        self.returned.set()
        return "said"


async def say_from_a_tool(agent, *, output=None):
    """Send "go" to `agent` and return what the agent said, and the handle of each say().

    The run ends once the task's on_enter has ended, which can be before the tool resumes. So this
    also waits for the tool to return, and for "ok", which answers the tool's output.
    """
    said, handles, ok = [], [], asyncio.Event()

    def on_item(ev):
        if isinstance(ev.item, ChatMessage) and ev.item.role == "assistant":
            said.append(ev.item.text_content)
            if ev.item.text_content == "ok":
                ok.set()

    def on_speech(ev):
        if ev.source == "say":
            handles.append(ev.speech_handle)

    async with AgentSession(llm=fake_llm()) as session:
        if output is not None:
            session.output.transcription = output
        session.on("conversation_item_added", on_item)
        session.on("speech_created", on_speech)
        await session.start(agent)
        await asyncio.wait_for(session.run(user_input="go"), timeout=5)
        await asyncio.wait_for(agent.returned.wait(), timeout=5)
        await asyncio.wait_for(ok.wait(), timeout=5)
    return said, handles


async def test_say_from_a_tool_returns():
    said, handles = await say_from_a_tool(SpeakingAgent())
    assert said == [LINE, "ok"]  # the line once, then the reply to the tool's output
    assert [h.allow_interruptions for h in handles] == [False]
    assert not handles[0].interrupted


async def test_a_line_that_never_finishes_playing_is_cut_off(caplog):
    with caplog.at_level(logging.WARNING):
        said, handles = await say_from_a_tool(
            SpeakingAgent(playout_timeout_s=0.1), output=StuckTranscription(LINE)
        )
    assert said[-1] == "ok"  # the tool returned, so the gate's lock would be free again
    assert handles[0].interrupted  # the stuck line was cut off rather than left playing
    assert "still playing" in caplog.text
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors == []  # nothing completed the task twice, and on_enter never raised
