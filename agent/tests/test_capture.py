"""LiveKitVoice.ask, the step-ups' TurnCapture, in a roomless text session with the copied FakeLLM.

A tool asks, the way the step-ups will from inside the gate, and a second session.run() gives the
answer: the multi-run pattern of the repo's test_nested_agent_task.py. That works because
CaptureTask captures in llm_node, which session.run() reaches, and on_user_turn_completed it never
calls.
"""

import asyncio
import json
import logging

import pytest
from fake_llm import FakeLLM, FakeLLMResponse
from livekit.agents import Agent, AgentSession, ModelSettings, function_tool
from livekit.agents.llm import ChatContext, ChatMessage, FunctionToolCall
from stuck_output import StuckTranscription

from callback_agent.capture import CaptureTask, LiveKitVoice

PROMPT = "Please repeat: 4 7 2"
ANSWER = "4 7 2"


def last_input(chat_ctx: ChatContext) -> str | None:
    """What a request to the LLM ends with: a message's text, or a tool's output."""
    last = chat_ctx.items[-1]
    return last.output if last.type == "function_call_output" else last.text_content


class RecordingLLM(FakeLLM):
    """The copied FakeLLM, noting what each request to it ended with."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.inputs: list[str | None] = []

    def chat(self, *, chat_ctx, **kwargs):
        self.inputs.append(last_input(chat_ctx))
        return super().chat(chat_ctx=chat_ctx, **kwargs)


def fake_llm() -> RecordingLLM:
    ask = FunctionToolCall(name="ask_caller", arguments="{}", call_id="call_1")
    return RecordingLLM(
        fake_responses=[
            FakeLLMResponse(input="go", content="", ttft=0, duration=0, tool_calls=[ask]),
            # Said only if the answer reached the LLM, which it never may.
            FakeLLMResponse(input=ANSWER, content="the LLM answered", ttft=0, duration=0),
        ]
    )


class ProbeAgent(Agent):
    """Its one tool asks PROMPT through LiveKitVoice and records the answer."""

    def __init__(self, *, timeout_s: float, secret: bool = False, **voice_options) -> None:
        super().__init__(instructions="test")
        self._timeout_s = timeout_s
        self._secret = secret
        self._voice_options = voice_options
        self.recorded: list[str | None] = []
        self.answered = asyncio.Event()

    @function_tool
    async def ask_caller(self) -> str:
        """Ask the caller to repeat the digits."""
        voice = LiveKitVoice(self.session, **self._voice_options)
        answer = await voice.ask(PROMPT, timeout_s=self._timeout_s, secret=self._secret)
        self.recorded.append(answer)
        self.answered.set()
        return "asked"


def messages(agent: Agent) -> list[ChatMessage]:
    return [item for item in agent.chat_ctx.items if isinstance(item, ChatMessage)]


async def ask_and_answer(agent: ProbeAgent, llm: RecordingLLM) -> list[tuple[str, str | None]]:
    """Run "go", which asks, then answer ANSWER. Returns each message the session committed.

    A committed message is a conversation_item_added event, which is what the session wiring
    forwards to the detector and logs.
    """
    committed = []

    def on_item(ev):
        if isinstance(ev.item, ChatMessage):
            committed.append((ev.item.role, ev.item.text_content))

    async with AgentSession(llm=llm) as session:
        session.on("conversation_item_added", on_item)
        await session.start(agent)
        await asyncio.wait_for(session.run(user_input="go"), timeout=5)
        assert isinstance(session.current_agent, CaptureTask)
        await asyncio.wait_for(session.run(user_input=ANSWER), timeout=5)
        assert isinstance(session.current_agent, ProbeAgent)
    return committed


async def test_ask_returns_the_next_user_turn():
    agent, llm = ProbeAgent(timeout_s=5), fake_llm()
    committed = await ask_and_answer(agent, llm)
    assert agent.recorded == [ANSWER]
    assert ("user", ANSWER) in committed  # so the detector still gets the turn to score
    assert [text for role, text in committed if role == "assistant"] == [PROMPT]  # no reply
    assert llm.inputs == ["go", "asked"]  # the answer never reached the LLM


def everything_in(agent: Agent) -> str:
    """Every item of the agent's chat context, calls and outputs included, as one text: all that
    the next request to its LLM can carry."""
    return json.dumps(agent.chat_ctx.to_dict())


@pytest.mark.parametrize("secret", [True, False], ids=["secret_prompt", "plain_prompt"])
async def test_the_answer_never_enters_the_parent_agents_chat_context(secret):
    # LiveKit merges a task's chat context into its parent's when the task completes, so the answer
    # stays out only because of the order things happen in. The test above looks at the last
    # request to the LLM; this looks at everything the parent holds, which the next one would carry.
    agent, llm = ProbeAgent(timeout_s=5, secret=secret), fake_llm()
    committed = await ask_and_answer(agent, llm)
    assert agent.recorded == [ANSWER]
    # The turn was committed all the same, for the detector: so the search below has something to
    # find, and finds nothing only because the parent's context does not hold it.
    assert ("user", ANSWER) in committed
    assert not any(m.role == "user" and m.text_content == ANSWER for m in messages(agent))
    # A plain prompt is in the context, and it quotes the answer: take it out before looking.
    assert ANSWER not in everything_in(agent).replace(PROMPT, "")
    if secret:
        assert PROMPT not in everything_in(agent)  # and a secret prompt is not there to take out


async def test_a_turn_before_the_prompt_has_played_is_not_the_answer():
    # session.run() during the prompt reaches llm_node, so the node itself must not count the turn.
    # On a call LiveKit drops such a turn before it gets this far.
    task = CaptureTask(PROMPT, timeout_s=5)
    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content=ANSWER)
    assert await task.llm_node(chat_ctx, [], ModelSettings()) is None
    assert not task.done()


async def test_a_second_turn_after_the_answer_changes_nothing():
    # Two turns can reach llm_node before LiveKit closes the task: the first is the answer, and the
    # second must neither replace it nor complete the task again.
    agent = ProbeAgent(timeout_s=5)
    async with AgentSession(llm=fake_llm()) as session:
        await session.start(agent)
        await asyncio.wait_for(session.run(user_input="go"), timeout=5)
        task = session.current_agent
        for text in [ANSWER, "something else"]:
            chat_ctx = ChatContext()
            chat_ctx.add_message(role="user", content=text)
            assert await task.llm_node(chat_ctx, [], ModelSettings()) is None
        await asyncio.wait_for(agent.answered.wait(), timeout=5)
    assert agent.recorded == [ANSWER]


async def test_ask_times_out_as_none():
    agent = ProbeAgent(timeout_s=0.1)
    async with AgentSession(llm=fake_llm()) as session:
        await session.start(agent)
        await asyncio.wait_for(session.run(user_input="go"), timeout=5)
        await asyncio.wait_for(agent.answered.wait(), timeout=5)
        assert agent.recorded == [None]
        assert isinstance(session.current_agent, ProbeAgent)


async def test_secret_prompt_stays_out_of_the_chat_context():
    agent = ProbeAgent(timeout_s=5, secret=True)
    committed = await ask_and_answer(agent, fake_llm())
    assert agent.recorded == [ANSWER]
    parent_messages = messages(agent)
    assert not any(PROMPT in (m.text_content or "") for m in parent_messages)
    # Nor was it ever committed, so the turn log never prints it.
    assert not any(PROMPT in (text or "") for _, text in committed)


async def test_a_prompt_that_is_not_secret_joins_the_chat_context():
    agent = ProbeAgent(timeout_s=5, secret=False)
    await ask_and_answer(agent, fake_llm())
    assert any(PROMPT in (m.text_content or "") for m in messages(agent))


async def test_ask_returns_none_when_the_session_closes_under_it():
    agent = ProbeAgent(timeout_s=60)  # only the close can end the wait
    async with AgentSession(llm=fake_llm()) as session:
        await session.start(agent)
        await asyncio.wait_for(session.run(user_input="go"), timeout=5)
        task = session.current_agent
        assert isinstance(task, CaptureTask)
        await asyncio.wait_for(session.aclose(), timeout=5)
        await asyncio.wait_for(agent.answered.wait(), timeout=5)
    assert agent.recorded == [None]
    # LiveKit, not the task, completed it here, and its answer timer still stopped. White-box:
    # a timer left running would only fire later, into a task that is already done.
    assert task._timers and all(timer.cancelled() for timer in task._timers)


async def test_a_prompt_that_never_finishes_playing_gets_no_answer(caplog):
    agent = ProbeAgent(timeout_s=5, playout_timeout_s=0.1)
    with caplog.at_level(logging.WARNING):
        async with AgentSession(llm=fake_llm()) as session:
            session.output.transcription = StuckTranscription(PROMPT)
            await session.start(agent)
            await asyncio.wait_for(session.run(user_input="go"), timeout=5)
            await asyncio.wait_for(agent.answered.wait(), timeout=5)
    assert agent.recorded == [None]  # the gate's lock is free again
    assert "still playing" in caplog.text
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
