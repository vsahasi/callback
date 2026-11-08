import asyncio
import logging

import pytest
from fake_llm import FakeLLM, FakeLLMResponse
from fake_toolbox import FakeToolbox
from livekit import rtc
from livekit.agents import Agent, AgentSession, ModelSettings
from livekit.agents.llm import ChatMessage

from callback_agent import prompts
from callback_agent.agents import CallbackAgent


def new_agent(**options) -> CallbackAgent:
    """A CallbackAgent over a fake toolbox: these tests are about the greeting and the audio tap."""
    return CallbackAgent(toolbox=FakeToolbox(), **options)


async def first_messages(count, *, responses=()):
    """Start a CallbackAgent in a roomless text session and return its first `count` chat messages.

    The first conversation_item_added of a session is an AgentHandoff marker, not a message, so it
    is skipped.
    """
    messages = []
    enough = asyncio.Event()

    def on_item(ev):
        if isinstance(ev.item, ChatMessage):
            messages.append(ev.item)
            if len(messages) >= count:
                enough.set()

    async with AgentSession(llm=FakeLLM(fake_responses=list(responses))) as session:
        session.on("conversation_item_added", on_item)
        await session.start(new_agent())
        await asyncio.wait_for(enough.wait(), timeout=5)
    return messages


async def test_disclosure_is_said_first():
    messages = await first_messages(1)
    assert messages[0].role == "assistant" and messages[0].text_content == prompts.DISCLOSURE


async def test_the_agent_asks_who_it_is_speaking_with_after_the_disclosure():
    ask = FakeLLMResponse(
        input=prompts.ASK_WHO, content="Who am I speaking with?", ttft=0, duration=0
    )
    messages = await first_messages(2, responses=[ask])
    assert [(m.role, m.text_content) for m in messages] == [
        ("assistant", prompts.DISCLOSURE),
        ("assistant", "Who am I speaking with?"),
    ]


async def test_the_disclosure_cannot_be_interrupted():
    said = asyncio.Event()
    handles = []

    def on_speech(ev):
        if ev.source == "say":
            handles.append(ev.speech_handle)
            said.set()

    async with AgentSession(llm=FakeLLM(fake_responses=[])) as session:
        session.on("speech_created", on_speech)
        await session.start(new_agent())
        await asyncio.wait_for(said.wait(), timeout=5)
    assert handles[0].allow_interruptions is False


@pytest.mark.parametrize(
    "phrase",
    [
        "automated assistant",  # we are an AI
        "recorded",  # California all-party consent
        "automated fraud checks",  # analysis by a machine
        "service providers",  # the audio goes to third parties
        "hang up",  # a way out
    ],
)
def test_disclosure_covers_the_consent_points(phrase):
    assert phrase in prompts.DISCLOSURE


# The stt_node tap. Agent.default.stt_node needs a live STT, so these tests swap it for a stand-in
# that reads every frame it is given and answers one event per frame.


def make_frames(count=3):
    return [rtc.AudioFrame.create(24000, 1, 480) for _ in range(count)]


@pytest.fixture
def stt_saw(monkeypatch):
    saw = []

    async def stand_in(agent, audio, model_settings):
        async for frame in audio:
            saw.append(frame)
            yield f"event {len(saw)}"

    monkeypatch.setattr(Agent.default, "stt_node", stand_in)
    return saw


async def run_stt_node(agent, frames):
    async def audio():
        for frame in frames:
            yield frame

    return [event async for event in agent.stt_node(audio(), ModelSettings())]


async def test_stt_node_taps_every_frame_and_passes_audio_and_events_through(stt_saw):
    frames, tapped = make_frames(), []
    events = await run_stt_node(new_agent(frame_tap=tapped.append), frames)
    assert tapped == frames
    assert stt_saw == frames
    assert events == ["event 1", "event 2", "event 3"]


async def test_stt_node_works_without_a_tap(stt_saw):
    frames = make_frames()
    assert await run_stt_node(new_agent(), frames) == ["event 1", "event 2", "event 3"]
    assert stt_saw == frames


async def test_a_failing_tap_is_dropped_and_never_breaks_the_stt_stream(stt_saw, caplog):
    frames, calls = make_frames(), []

    def bad_tap(frame):
        calls.append(frame)
        raise RuntimeError("boom")

    with caplog.at_level(logging.ERROR, logger="callback_agent.agents"):
        events = await run_stt_node(new_agent(frame_tap=bad_tap), frames)
    assert stt_saw == frames and len(events) == 3  # the call stays up
    assert calls == frames[:1]  # one failure and the tap is dropped
    assert "frame tap" in caplog.text
