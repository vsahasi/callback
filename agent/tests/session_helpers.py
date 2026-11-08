"""Helpers for tests that run CallbackAgent in a roomless text session."""

import asyncio
import contextlib

from livekit.agents import AgentSession
from livekit.agents.llm import ChatMessage


@contextlib.asynccontextmanager
async def started(agent, llm, **session_options):
    """A started roomless session whose greeting has been said, so that run() starts clean.

    The agent opens with the recording disclosure and then asks who is calling; both are in the chat
    before the session is handed over. A session that fails to close is given up on after 3 s, so a
    deadlock fails the test that caused it and does not hang the run.
    """
    spoken = asyncio.Event()
    messages: list[ChatMessage] = []

    def on_item(ev) -> None:
        if isinstance(ev.item, ChatMessage):
            messages.append(ev.item)
            if len(messages) >= 2:  # the disclosure, then the question
                spoken.set()

    session = AgentSession(llm=llm, **session_options)
    session.on("conversation_item_added", on_item)
    try:
        await session.start(agent)
        await asyncio.wait_for(spoken.wait(), timeout=30)
        yield session
    finally:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(session.aclose(), timeout=3)
