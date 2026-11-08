"""How the gate and the step-ups talk to the caller: AgentTasks on the call's LiveKit session.

LiveKitVoice is the gate's Speaker (say) and the step-ups' TurnCapture (say and ask). It meets both
Protocols by shape, so it imports neither. Every line goes through a task, because LiveKit lets a
tool body, on_enter or on_exit await a task, and the research could not rule out a deadlock when a
tool body awaits a plain session.say(). The gate always runs inside a tool.

A task must be awaited directly: asyncio.wait_for and asyncio.timeout break LiveKit's await rules.
So the tasks keep time themselves, on call_later timers, and complete themselves:
- A line still playing after `playout_timeout_s` (30 s) is cut off with a forced interrupt, and the
  task completes as if the line had ended (ruling R38): a hung TTS must not hold the gate's lock
  forever. The interrupt is what frees the caller. LiveKit closes a task's activity only once the
  task's speech and on_enter have ended, and a forced interrupt ends even a stuck speech, at the
  latest when LiveKit cancels it after its INTERRUPTION_TIMEOUT (5 s).
- CaptureTask's answer timer starts once its prompt has played.

The tasks complete exactly once. LiveKit completes them too (cancel() when the session closes), so
each completion of ours checks done() first, and complete() itself stops the task's timers, whoever
calls it. on_enter never raises: LiveKit resumes the parent agent only when the task completes, so a
line that cannot be said completes the task with the error rather than strand the call in it.

Nothing here logs a line or an answer: a prompt can hold the challenge phrase, an answer a code.
"""

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from livekit.agents import AgentSession, AgentTask, ModelSettings, llm
from livekit.agents.llm import ToolError
from livekit.agents.voice import SpeechHandle

from callback_agent.agents import TimedSttMixin

logger = logging.getLogger(__name__)

PLAYOUT_TIMEOUT_S = 30.0  # ruling R38: a line still playing after this long is cut off


class _OneLine:
    """Mixed into SayTask and CaptureTask: one line said, interruptions off, one completion.

    It goes in front of TimedSttMixin and AgentTask, whose session, done() and complete() it uses.
    """

    def __init__(self, *, playout_timeout_s: float, **kwargs: Any) -> None:
        self._playout_timeout_s = playout_timeout_s
        self._timers: list[asyncio.TimerHandle] = []
        super().__init__(**kwargs)

    def complete(self, result: Any) -> None:
        # Every completion passes here, LiveKit's own included, so no timer outlives the task.
        for timer in self._timers:
            timer.cancel()
        super().complete(result)

    def _finish(self, result: Any) -> None:
        """Complete the task, unless something has completed it already."""
        if not self.done():
            self.complete(result)

    def _call_later(
        self, delay: float, callback: Callable[..., None], *args: Any
    ) -> asyncio.TimerHandle:
        timer = asyncio.get_running_loop().call_later(delay, callback, *args)
        self._timers.append(timer)
        return timer

    async def _say_line(self, text: str, *, add_to_chat_ctx: bool = True) -> bool:
        """Say `text` with interruptions off. True if it played out and the task is still running.

        On False the task has completed: LiveKit cancelled it (the session is closing), the line
        could not be said, or it was still playing after `playout_timeout_s` and was cut off.
        """
        if self.done():
            return False
        try:
            handle = self.session.say(
                text, allow_interruptions=False, add_to_chat_ctx=add_to_chat_ctx
            )
        except Exception as exc:
            self._finish(exc)  # the awaiting tool gets the error
            return False
        cut_off = self._call_later(self._playout_timeout_s, self._cut_off, handle)
        await handle  # once it has played, or once a cut-off has ended it
        cut_off.cancel()
        return not self.done()

    def _cut_off(self, handle: SpeechHandle) -> None:
        logger.warning(
            "a line was still playing after %.1f s, so it is cut off", self._playout_timeout_s
        )
        handle.interrupt(force=True)
        self._finish(None)


class SayTask(_OneLine, TimedSttMixin, AgentTask[None]):
    """Says one line with interruptions off, then completes. Every line of the gate is one."""

    def __init__(self, text: str, *, playout_timeout_s: float = PLAYOUT_TIMEOUT_S) -> None:
        super().__init__(
            instructions="Saying a line to the caller. Never speak.",
            playout_timeout_s=playout_timeout_s,
        )
        self._text = text

    async def on_enter(self) -> None:
        if await self._say_line(self._text):
            self._finish(None)


class CaptureTask(_OneLine, TimedSttMixin, AgentTask[str | None]):
    """Says a prompt, then completes with the text of the caller's next finished turn.

    The LLM never answers that turn, but LiveKit still commits it as a conversation item, which is
    what the session wiring forwards to the detector. None if no turn ends within `timeout_s` of the
    prompt playing, or if the prompt was cut off. A `secret` prompt never enters the chat context,
    so the main LLM can never repeat it.
    """

    def __init__(
        self,
        prompt: str,
        *,
        timeout_s: float,
        secret: bool = False,
        playout_timeout_s: float = PLAYOUT_TIMEOUT_S,
    ) -> None:
        super().__init__(
            instructions="Verification step. Never speak.",
            turn_handling={"preemptive_generation": {"enabled": False}},
            playout_timeout_s=playout_timeout_s,
        )
        self._prompt = prompt
        self._timeout_s = timeout_s
        self._secret = secret
        self._prompt_played = False

    async def on_enter(self) -> None:
        if not await self._say_line(self._prompt, add_to_chat_ctx=not self._secret):
            return
        self._prompt_played = True
        self._call_later(self._timeout_s, self._no_answer)

    def _no_answer(self) -> None:
        logger.info("no answer within %.1f s of the prompt", self._timeout_s)
        self._finish(None)

    async def llm_node(
        self, chat_ctx: llm.ChatContext, tools: list[llm.Tool], model_settings: ModelSettings
    ) -> None:
        """Take the caller's turn as the answer and say nothing. It never calls an LLM.

        This is the one place a finished turn reaches on both of its paths, the end of a spoken
        turn and session.run(), which never calls on_user_turn_completed. A turn that ends while
        the prompt plays is dropped by LiveKit before it gets here, and one that gets here anyway
        (session.run() during the prompt) is ignored. A return that is not an AsyncIterable means
        no reply.
        """
        last = chat_ctx.items[-1] if chat_ctx.items else None
        if self._prompt_played and isinstance(last, llm.ChatMessage) and last.role == "user":
            self._finish(last.text_content or "")
        return None


class LiveKitVoice:
    """The caller's side of one LiveKit session, for the gate and the step-ups.

    say and ask each await an AgentTask, so await them from a tool body, an on_enter or an on_exit,
    which is where the gate runs, and never through asyncio.wait_for or asyncio.timeout: the tasks
    keep time themselves.
    """

    def __init__(
        self, session: AgentSession, *, playout_timeout_s: float = PLAYOUT_TIMEOUT_S
    ) -> None:
        self._session = session  # LiveKit runs each task in the session of the tool awaiting it
        self._playout_timeout_s = playout_timeout_s

    async def say(self, text: str) -> None:
        """Say `text` with interruptions off, and return once it has played or been cut off.

        If the session closes under it, it returns quietly.
        """
        try:
            await SayTask(text, playout_timeout_s=self._playout_timeout_s)
        except ToolError:
            logger.info("the session closed before a line was said")

    async def ask(self, prompt: str, *, timeout_s: float, secret: bool = False) -> str | None:
        """Say `prompt` with interruptions off, then return the caller's next finished turn.

        None on a timeout, a cut-off prompt, or a session that closes under it (the caller left).
        """
        try:
            return await CaptureTask(
                prompt,
                timeout_s=timeout_s,
                secret=secret,
                playout_timeout_s=self._playout_timeout_s,
            )
        except ToolError:
            logger.info("the session closed while waiting for an answer")
            return None
