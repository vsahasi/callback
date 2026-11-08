# LiveKit Agents 1.8.10 (Python): AgentTask flows, text-only testing, and the callback leg

Pinned to git tag `livekit-agents@1.8.10` (commit `ff962a0`, 2026-10-28); every source link below points at that tag. The docs.livekit.io pages were fetched on 2026-11-04 (rendered 2026-11-05 UTC) and are not version-pinned. Where they disagree with the 1.8.10 source, that is flagged. Two files came from `main` rather than a pinned version: the `livekit/protocol` enum and the `livekit/python-sdks` room service.

## Q1. AgentTask and workflows: typed tasks, await sites, no-LLM capture turns, SpeechHandle, timeouts

### Takeaway
`AgentTask[ResultT]` (`from livekit.agents import AgentTask`) is the 1.8.10 primitive. You subclass it, finish it with `self.complete(result)`, and `await MyTask(...)` only from a `@function_tool` body, `on_enter`, or `on_exit`; awaiting it anywhere else raises `RuntimeError`. While it runs, the task is `session.current_agent`, with its own tools and an empty chat context unless you pass one. The parent activity is paused, and the same activity instance resumes later. When the task completes, its chat items (minus system/developer messages and, by default, function calls) are merged into the parent, and then the parent resumes.

A no-LLM challenge turn is feasible: `session.say(..., allow_interruptions=False, add_to_chat_ctx=False)`, then `on_user_turn_completed` plus `raise StopResponse()`, optionally with `llm=None` on the task. Three source behaviours shape it:
- STT receives silence while uninterruptible speech plays.
- A user turn that *ends* during that speech is dropped before the hook runs.
- The hook runs only on the audio end-of-turn path, never from `session.run()`.

### Cited Findings

#### Version and import paths
- The tag `livekit-agents@1.8.10` is commit `ff962a0` (2026-10-28) and sets `__version__ = "1.8.10"`. PyPI shows 1.8.10 uploaded 2026-10-28T17:17Z and `2.0.0b1` uploaded 2026-10-12 (pre-release) — [version.py](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/version.py); [PyPI JSON](https://pypi.org/pypi/livekit-agents/json)
- Top-level exports include `AgentTask`, `StopResponse`, `RunContext`, `CloseReason`, `RunResult`, `mock_tools`, and `get_job_context`. `AgentTask` and `SpeechHandle` are also exported from `livekit.agents.voice` — [__init__.py L39-L278](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/__init__.py#L39-L278); [voice/__init__.py](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/__init__.py#L2-L73)
- Workflows live in `livekit.agents.beta.workflows`: `TaskGroup`, `TaskCompletedEvent`, `TaskGroupResult`, `GetEmailTask`/`GetEmailResult`, `WarmTransferTask`/`WarmTransferResult`, `TwilioConnectorWarmTransferTask`, and the name/address/DOB/phone/credit-card/DTMF tasks — [beta/workflows/__init__.py](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/beta/workflows/__init__.py); [docs index of prebuilt tasks](https://docs.livekit.io/agents/llms.txt)

#### Defining and awaiting a task (source)
- The constructor at [agent.py L878-L927](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent.py#L878-L927) is:
  ```python
  class AgentTask(Agent, Generic[TaskResult_T]):
      def __init__(self, *, instructions: str | Instructions,          # required, no default
                   chat_ctx=NOT_GIVEN, tools: list[llm.Tool | llm.Toolset] | None = None,
                   stt=NOT_GIVEN, vad=NOT_GIVEN, turn_handling: NotGivenOr[TurnHandlingOptions] = NOT_GIVEN,
                   llm=NOT_GIVEN, tts=NOT_GIVEN, preserve_function_call_history: bool = False,
                   # deprecated: turn_detection, allow_interruptions, min/max_endpointing_delay
                   mcp_servers=NOT_GIVEN) -> None
  ```
- `complete(result: TaskResult_T | Exception)` raises `RuntimeError("<Cls> is already done")` if called twice. An `Exception` result is raised at the await site via `set_exception`. If `complete()` is called inside a speech-handle context, it also stores the result on that handle as `_maybe_run_final_output`, which `RunResult.final_output` reads — [agent.py L939-L958](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent.py#L939-L958)
- `done()` is at [agent.py L929-L930](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent.py#L929-L930). `cancel()` force-interrupts the task's activity and completes it with `ToolError(f"AgentTask {self.id} is cancelled")` — [agent.py L932-L937](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent.py#L932-L937)
- Await rules:
  - A task is not re-entrant: a second await raises `RuntimeError("... is not re-entrant, await only once")`.
  - The awaiting asyncio task must be flagged as an inline task. If it is not, the await raises `RuntimeError("<Cls> should only be awaited inside tool_functions or the on_enter/on_exit methods of an Agent")`.
  - Source: [agent.py L960-L976](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent.py#L960-L976)
- The inline flag is set on three kinds of asyncio task:
  - The `on_enter` task — [agent_activity.py L1093-L1097](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent_activity.py#L1093-L1097)
  - The `on_exit` task — [agent_activity.py L1405](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent_activity.py#L1405)
  - Each tool execution, together with its function call and speech handle — [generation.py L1098-L1099](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/generation.py#L1098-L1099)
- The docs agree on three call sites (`on_enter`, `on_exit`, a tool body) and say awaiting elsewhere raises `RuntimeError`. If one LLM step calls several tools that each await a task, the tasks run one at a time — [tasks.md "Running a task"](https://docs.livekit.io/agents/logic/tasks.md). The source mechanism is a per-activity lock in `_inline_task_slot` — [agent_activity.py L1452-L1510](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent_activity.py#L1452-L1510)
- Docs example, copied with light trimming from [tasks.md](https://docs.livekit.io/agents/logic/tasks.md):
  ```python
  from livekit.agents import Agent, AgentTask, function_tool

  class CollectConsent(AgentTask[bool]):
      def __init__(self, chat_ctx=None):
          super().__init__(instructions="Ask for recording consent and get a clear yes or no answer.", chat_ctx=chat_ctx)
      async def on_enter(self) -> None:
          await self.session.generate_reply(instructions="Ask for permission to record the call...")
      @function_tool
      async def consent_given(self) -> None:
          """Use this when the user gives consent to record."""
          self.complete(True)
      @function_tool
      async def consent_denied(self) -> None:
          """Use this when the user denies consent to record."""
          self.complete(False)

  class CustomerServiceAgent(Agent):
      async def on_enter(self) -> None:
          if await CollectConsent(chat_ctx=self.chat_ctx):        # awaited from on_enter
              ...
      @function_tool()
      async def collect_contact_info(self):
          """Collect the user's contact information."""
          result = await GetContactInfoTask(chat_ctx=self.chat_ctx.copy(exclude_instructions=True))  # from a tool
          return f"Recorded contact info for {result.name}."
  ```

#### Chat context, tools, and active agent during and after the task (source + docs)
- "By default, a task starts with an empty chat context." To include history, pass `chat_ctx=`; `copy(exclude_instructions=True)` drops the parent's system prompt — [tasks.md "Passing conversation history to a task"](https://docs.livekit.io/agents/logic/tasks.md)
- How the await switches agents:
  - It calls `session._update_activity(self, previous_activity="pause", blocked_tasks=..., wait_on_enter=False)`. The parent activity is paused and the task gets its own activity.
  - The switch does not wait for the task's `on_enter` to finish.
  - A running `RunResult` watches the task's `on_enter` as a guard, then un-watches the parent speech so a test run can complete while the task waits for user input.
  - Source: [agent.py L1029-L1094](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent.py#L1029-L1094)
- **STT pipeline on a task switch.** The parent hands its STT pipeline to the task's activity **only if both agents use the default `stt_node` and the same STT instance**. The source comment says "only reuse with the default stt_node, a custom override may access the old self.session/activity". Otherwise the new activity starts a fresh STT stream through its *own* `stt_node` — [agent_activity.py L1101-L1115](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent_activity.py#L1101-L1115); [L1355-L1371](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent_activity.py#L1355-L1371). A reused pipeline is rebound to the currently active agent's node — [audio_recognition.py L268-L272](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/audio_recognition.py#L268-L272)
- The repo test asserts `isinstance(sess.current_agent, SimpleTask)` while the task waits, and `EnterHandoffAgent` after it completes — [tests/test_nested_agent_task.py L114-L160](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/tests/test_nested_agent_task.py#L114-L160)
- The tool's speech while the task runs:
  - The parent (tool-call) speech gets `_hold_interruptions()` for the task's whole lifetime.
  - If that speech is already interrupted when the task starts, the await raises `RuntimeError("the speech that awaited the inline task is interrupted")`.
  - If the activity is closing, the await raises `ToolError` ("...is closing"); if an agent transition is in progress, it raises `ToolError` ("An agent transition is in progress...").
  - Source: [agent_activity.py L1452-L1510](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent_activity.py#L1452-L1510)
- After completion (the `finally` block):
  - If `session.current_agent != self` (for example after `update_agent`), it logs a warning and closes the old activity without handing back.
  - Otherwise it runs `merged = old_agent.chat_ctx.copy().merge(self.chat_ctx, exclude_function_call=not self._preserve_function_call_history, exclude_instructions=True)`, writes that into the parent's `_chat_ctx`, and resumes the parent via `_update_activity(old_agent, new_activity="resume", wait_on_enter=False)`.
  - Source: [agent.py L1094-L1118](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent.py#L1094-L1118)
- `ChatContext.merge` will "Add messages from `other_chat_ctx` into this one, avoiding duplicates, and keep items sorted by created_at" — [chat_context.py L738-L760](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/llm/chat_context.py#L738-L760)
- If the asyncio task doing the await finishes before the `AgentTask` completes, the framework logs an error and completes the task with `RuntimeError("The asyncio.Task finished before ... was completed.")` — [agent.py L987-L1014](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent.py#L987-L1014)
- Tool visibility: a task "can define its own tools" and "takes control of the session until it returns a result" — [tasks.md](https://docs.livekit.io/agents/logic/tasks.md). `ToolFlag` values are `NONE`, `IGNORE_ON_ENTER`, and `CANCELLABLE` — [tool_context.py L171-L174](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/llm/tool_context.py#L171-L174). `WarmTransferTask` uses `@function_tool(flags=ToolFlag.IGNORE_ON_ENTER)` so the LLM cannot call its tools during `on_enter` — [warm_transfer.py L237-L260](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/beta/workflows/warm_transfer.py#L237-L260)

#### No-LLM capture turn: `say` + `on_user_turn_completed` + `StopResponse` (source)
- `StopResponse` is `class StopResponse(Exception)` in `livekit/agents/llm/tool_context.py`, docstring: "raised by the user to indicate that the agent should not generate a response". Import it as `from livekit.agents import StopResponse` or `from livekit.agents.llm import StopResponse` — [tool_context.py L160-L169](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/llm/tool_context.py#L160-L169); [__init__.py L56](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/__init__.py#L56)
- The hook signature is `async def on_user_turn_completed(self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage) -> None`, documented as "Called when the user has finished speaking, and the LLM is about to respond" — [agent.py L344-L352](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent.py#L344-L352). `turn_ctx` is a temporary copy: "changes will not be kept inside the Agent.chat_ctx" — [agent_activity.py L2876-L2879](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent_activity.py#L2876-L2879)
- In 1.8.10, `on_user_turn_completed` is called only from `_user_turn_completed_impl`, the audio end-of-turn path; the module has no other call site. Raising `StopResponse` there hits `except StopResponse: ... return  # ignore this turn`, which returns before `_generate_reply`: no LLM call, and no code path appends the user message to `agent.chat_ctx` — [agent_activity.py L2790-L2905](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent_activity.py#L2790-L2905)
- **Early-exit gates before the hook** — [agent_activity.py L2850-L2874](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent_activity.py#L2850-L2874):
  - If the current speech has `allow_interruptions=False`, the code logs "skipping reply to user input, current speech generation cannot be interrupted" and returns. The hook is not called and the transcript is not added to the chat context.
  - If scheduling is paused or new turns are blocked, it logs "skipping on_user_turn_completed, speech scheduling is paused".
- **Audio is replaced while uninterruptible speech plays.**
  - The option is `InterruptionOptions.discard_audio_if_uninterruptible`: "Drop buffered audio while the agent speaks and cannot be interrupted. Defaults to `True`" — [turn.py L150-L198](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/turn.py#L150-L198)
  - The implementation "substitute[s] silence on the paths that would otherwise see contaminated/echoed audio (STT, realtime model)… VAD, AMD and the interruption detector keep receiving the real frame." The same switch fires during AEC warm-up — [agent_activity.py L1693-L1712](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent_activity.py#L1693-L1712)
  - The old `AgentSession(discard_audio_if_uninterruptible=...)` kwarg is deprecated in favour of `turn_handling=TurnHandlingOptions(...)` — [agent_session.py L400](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent_session.py#L400)
- `llm=None` on the task:
  - `AgentActivity.llm` returns `self._agent.llm if is_given(self._agent.llm) else self._session.llm`, so an explicit `None` disables the LLM for that agent — [agent_activity.py L5334-L5335](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent_activity.py#L5334-L5335)
  - After the hook runs, `elif self.llm is None: return  # skip response if no llm is set` — [agent_activity.py L2915](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent_activity.py#L2915)
  - But `generate_reply()`, and therefore `session.run()`, raises `RuntimeError("trying to generate reply without an LLM model")` — [agent_activity.py L1825-L1826](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent_activity.py#L1825-L1826)
- Preemptive generation is on by default (`enabled: True`, `preemptive_tts: False`, `max_speech_duration: 10.0`, `max_retries: 3`) — [turn.py L201-L228](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/turn.py#L201-L228). Per-agent `turn_handling["preemptive_generation"]` overrides the session's — [agent_activity.py L689-L693](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent_activity.py#L689-L693). "A parked preemptive generation is never scheduled" — [agent_activity.py L1437](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent_activity.py#L1437)
- `llm_node` return handling: a `str` is spoken as-is; anything that is not an `AsyncIterable` (for example `None`) makes the generation return `False`, so no reply text is produced — [generation.py L249-L264](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/generation.py#L249-L264)
- `StopResponse` raised from a *tool* yields a `FunctionCallOutput` with `output == ""`, `is_error False`, and `reply_required False` ("StopResponse means 'say nothing'") — [tests/test_tool_output_per_call.py L96-L101](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/tests/test_tool_output_per_call.py#L96-L101)
- No prebuilt task or tool in `beta/` (or `voice/ivr`, `voice/amd`) overrides `on_user_turn_completed` or raises `StopResponse`, so the hook-based capture pattern is not demonstrated in official code. This comes from my grep of [beta/](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/beta)

#### `say()`, `SpeechHandle`, and "agent speech ended" signals (source)
- `AgentSession.say(text: str | AsyncIterable[str], *, audio=NOT_GIVEN, allow_interruptions=NOT_GIVEN, add_to_chat_ctx: bool = True) -> SpeechHandle`. It raises `RuntimeError` if the session isn't running or is closing — [agent_session.py L1574-L1606](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent_session.py#L1574-L1606)
- `generate_reply(*, user_input, instructions, tool_choice, tools, allow_interruptions, chat_ctx, input_modality="text")`. Docstring: "`await handle` waits for the reply to finish and never raises; check `handle.exception()`" — [agent_session.py L1608-L1676](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent_session.py#L1608-L1676)
- In the activity's `say()`:
  - It raises "trying to generate speech from text without a TTS model" **only if** there is an enabled audio output.
  - `allow_interruptions` defaults to the activity setting.
  - It emits `speech_created` with `SpeechCreatedEvent(user_initiated=True, source="say")`.
  - Source: [agent_activity.py L1730-L1765](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent_activity.py#L1730-L1765)
- `SpeechHandle` API — [speech_handle.py L33-L272](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/speech_handle.py#L33-L272):
  - Priority constants: `SPEECH_PRIORITY_LOW/NORMAL/HIGH = 0/5/10`.
  - Properties: `id`, `scheduled`, `interrupted` (`_interrupt_fut.done()`), and `allow_interruptions`. The `allow_interruptions` setter raises `RuntimeError` if you try to disable interruptions on an already-interrupted handle.
  - `chat_items`, `done()`, and `exception()`; `exception()` raises `asyncio.InvalidStateError` if the handle is not done.
  - `interrupt(*, force=False, source="programmatic")` raises if interruptions are disallowed and `force` is not set.
  - `wait_for_playout()` "waits for the entire assistant turn to complete playback". It raises `RuntimeError` when called from inside the tool that owns the handle (use `RunContext.wait_for_playout()` instead).
  - `__await__` calls `wait_for_playout()` and then returns the handle; `add_done_callback(cb)` and `remove_done_callback(cb)` are also available.
- `SpeechHandle` has **no playout-finished timestamp attribute** (see its full API list above). Where playout end is recorded:
  - In the `say` path, after `audio_output.wait_for_playout()` the code sets `stopped_speaking_at = time.time()`.
  - It attaches `metrics={"started_speaking_at", "stopped_speaking_at", "playback_latency", ...}` to the assistant `ChatMessage` **only when `forwarded_text and add_to_chat_ctx`**, then emits `conversation_item_added`.
  - It then moves the agent state from `speaking` to `listening` (or `thinking`).
  - Source: [agent_activity.py L3350-L3440](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent_activity.py#L3350-L3440)
- `MetricsReport` fields include `started_speaking_at`, `stopped_speaking_at`, `transcription_delay`, `end_of_turn_delay`, `on_user_turn_completed_delay`, and `llm_node_ttft` — [chat_context.py L286-L330](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/llm/chat_context.py#L286-L330). `AgentStateChangedEvent(old_state, new_state, created_at)` — [events.py L338-L342](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/events.py#L338-L342)
- The lower-level hook is the audio output itself. `AudioOutput` is an `rtc.EventEmitter` emitting `playback_started` (`PlaybackStartedEvent(created_at)`), `playback_finished` (`PlaybackFinishedEvent(playback_position, interrupted, synchronized_transcript)`, which has no timestamp), and `playback_progressed`. Wrapper outputs forward these events up the chain — [io.py L111-L262](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/io.py#L111-L262)

#### Timeouts and cancellation (source + docs)
- `AgentTask` has no timeout parameter (constructor above). LiveKit's own tasks build their own timeouts:
  - `GetDtmfTask(dtmf_input_timeout: float = 4.0)` uses `@debounced(delay=dtmf_input_timeout)` and completes with `ToolError` when the input is incomplete — [dtmf_inputs.py L43, L116-L131](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/beta/workflows/dtmf_inputs.py#L116-L131)
  - `WarmTransferTask(ringing_timeout=...)` maps to `CreateSIPParticipantRequest.ringing_timeout.FromNanoseconds(...)` — [warm_transfer.py L54, L396-L397](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/beta/workflows/warm_transfer.py#L382-L400)
- The away timer:
  - `AgentSession(user_away_timeout: float | None = 15.0)` moves the user state to `"away"` — [agent_session.py L435](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent_session.py#L435); `UserStateChangedEvent(old_state, new_state, created_at)` — [events.py L331-L335](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/events.py#L331-L335)
  - Repo tests: while a task awaited from a tool waits on the user, the away timer runs (#8018) — [test_agent_task_user_away.py L131-L133](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/tests/test_agent_task_user_away.py#L131-L133)
- Session close unwinds nested tasks:
  - `_aclose_impl` loops `while isinstance(activity.agent, AgentTask)`. With `drain=True` it first drains, then calls `agent_task.cancel()` and `await agent_task._wait_for_inactive()`. The awaiting code therefore gets `ToolError("AgentTask ... is cancelled")` — [agent_session.py L1389-L1408](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent_session.py#L1389-L1408)
  - Test docstrings: "A drained shutdown() while an AgentTask is speaking lets the sentence finish"; "drain=False, like a participant disconnect or aclose(), still cuts the task off" — [test_agent_task_shutdown_drain.py L131-L175](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/tests/test_agent_task_shutdown_drain.py#L131-L175)
- A participant disconnect closes the session only through RoomIO (details in Q3), using `_close_soon(reason=CloseReason.PARTICIPANT_DISCONNECTED)` with the default `drain=False` — [agent_session.py L1263-L1274](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent_session.py#L1263-L1274)

#### TaskGroup (docs + source)
- `TaskGroup(AgentTask[TaskGroupResult])` takes `(*, summarize_chat_ctx=True, return_exceptions=False, chat_ctx=NOT_GIVEN, on_task_completed=None)` — [task_group.py](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/beta/workflows/task_group.py)
- The docs mark it experimental. Usage is `task_group.add(lambda: IntroTask(), id="intro_task", description="...")`, then `results = await task_group` and `results.task_results` — [tasks.md "Task group"](https://docs.livekit.io/agents/logic/tasks.md)
- Calling `session.shutdown()` from `on_task_completed` raises `RuntimeError`; raise a custom exception there instead and catch it where you await the group — [tasks.md "Early exit from a task group"](https://docs.livekit.io/agents/logic/tasks.md)

#### Docs vs source discrepancies
- The docs advise "Avoid awaiting playout inside onEnter() when triggered from a tool… call `generate_reply()` without awaiting it" — [tasks.md "Best practices for testing task groups"](https://docs.livekit.io/agents/logic/tasks.md). The 1.8.10 Python source starts the task with `wait_on_enter=False` — [agent.py L1029-L1040](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent.py#L1029-L1040). The repo test `test_nested_agent_task_no_deadlock` awaits `generate_reply` in `OuterTask.on_enter`, with `OuterTask` started from a tool, and asserts no deadlock — [test_nested_agent_task.py L30-L70](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/tests/test_nested_agent_task.py#L30-L70). The caution may be Node-oriented (it uses `onEnter()` naming).

### Inferences
- **Challenge task design.** The sketch below is my own assembly from the APIs above; it has not been run.
  ```python
  import asyncio, time
  from dataclasses import dataclass
  from livekit.agents import AgentTask, StopResponse, llm

  @dataclass
  class ChallengeResult:
      transcript: str | None           # None on timeout
      agent_speech_end: float | None   # wall clock when local playout finished
      timed_out: bool = False

  class VoiceChallengeTask(AgentTask[ChallengeResult]):
      def __init__(self, phrase: str, *, answer_timeout: float = 30.0) -> None:
          super().__init__(
              instructions="Verification step.",                  # required kwarg in 1.8.10
              llm=None,                                            # variant A: no LLM at all in this task
              turn_handling={"preemptive_generation": {"enabled": False}},  # matters if you keep an LLM
          )
          self._phrase, self._timeout = phrase, answer_timeout
          self._speech_end: float | None = None
          self._timer: asyncio.TimerHandle | None = None

      async def on_enter(self) -> None:
          handle = self.session.say(self._phrase, allow_interruptions=False, add_to_chat_ctx=False)
          await handle                                   # == handle.wait_for_playout()
          self._speech_end = time.time()
          self._timer = asyncio.get_running_loop().call_later(self._timeout, self._on_timeout)

      def _on_timeout(self) -> None:
          if not self.done():
              self.complete(ChallengeResult(None, self._speech_end, timed_out=True))

      async def on_user_turn_completed(self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage) -> None:
          if self._speech_end is not None and not self.done():
              if self._timer:
                  self._timer.cancel()
              self.complete(ChallengeResult(new_message.text_content, self._speech_end))
          raise StopResponse()   # no LLM reply, and the user message is not added to chat_ctx
  ```
  - From a tool: `res = await VoiceChallengeTask(phrase)`. When the task finishes, the parent resumes and the tool continues with `res`.
  - Because `add_to_chat_ctx=False` and `StopResponse` keep both the phrase and the answer out of the task's chat context, nothing about the challenge is merged into the parent LLM's context. That is desirable if the main LLM must not be able to repeat the phrase.
- **Custom `stt_node` and tasks.** The pass-1 Flux word-timestamp `stt_node` override disables pipeline reuse. So every task start and end reopens the STT (Flux) stream, and the task runs *its own* `stt_node`, which is the default unless it overrides one. Put the custom `stt_node` in a mixin shared by the main agent and every challenge/callback task, or timestamps silently stop during the challenge. The reopen happens in `on_enter`, before the caller is asked to speak, so the reconnect latency should fall inside the phrase playout.
- **Early answers lose words.** With the default `discard_audio_if_uninterruptible=True`, a caller who starts repeating before the phrase finishes has those words replaced by silence for STT.
  - If their whole turn ends before playout ends, the turn is dropped entirely (the gate at L2850-L2857), and the next utterance is captured instead.
  - Setting `turn_handling={"interruption": {"discard_audio_if_uninterruptible": False}}` on the task keeps overlapping audio. On speakerphone or browser calls without AEC, that also lets agent echo into STT, which is why the default discards it.
  - Either way, have the gate tell the detection service whether the utterance began before `agent_speech_end`; the pass-1 Flux word timestamps can provide this.
- **The `agent_speech_end` timestamp.**
  - Most precise in-framework option: listen on `session.output.audio.on("playback_finished", ...)` and stamp `time.time()`.
  - Next best: `time.time()` right after `await handle`, or the `agent_state_changed` event to `listening`.
  - With `add_to_chat_ctx=True`, you can instead read `metrics["stopped_speaking_at"]` from the `conversation_item_added` message.
  - All of these measure when audio left the agent's local output, not when the caller's handset played it. SIP/WebRTC network and jitter-buffer delay are unmeasured.
- **Timeouts: complete from inside the task.** Do it with a timer, as in the sketch, rather than wrapping the await.
  - On Python ≤3.11, `asyncio.wait_for(MyTask(), 30)` wraps the awaitable in a *new* asyncio.Task that lacks the inline flag. That should raise the "should only be awaited inside tool_functions..." error (agent.py L972-L976).
  - On 3.12+, `wait_for` and `asyncio.timeout()` cancel the awaiting tool task instead. That resumes the parent but leaves the task's future unresolved, which then trips the "asyncio.Task finished before ... was completed" error path (agent.py L987-L1014).
- **Text-testable variant.** To make the same task drivable by `session.run()` (see Q2), keep an LLM on it, i.e. drop `llm=None`, and capture in `llm_node`, which runs on both the audio path and `session.run()`:
  ```python
  async def llm_node(self, chat_ctx, tools, model_settings):
      last = chat_ctx.items[-1]
      if last.type == "message" and last.role == "user" and not self.done():
          self.complete(ChallengeResult(last.text_content, self._speech_end))
      return None   # generation.py L263: non-AsyncIterable -> no reply
  ```
  The cost: on this path the user message *is* added to the task's chat context, and it is later merged into the parent's (merge excludes only function calls and system messages). Strip it afterwards if the parent LLM must not see it.
- **SMS-code step.** The same pattern works (say the prompt, capture the next utterance). Alternatively, use an LLM-driven task with a `record_code(code: str)` tool, as in the docs' `CollectConsent` pattern, if you want the LLM to normalise "four two seven…" into digits. `GetDtmfTask` exists if keypad entry is acceptable.

### Gaps
- None of these code paths were executed: the `llm=None` task, completing from `on_user_turn_completed`, and an async `llm_node` returning `None`. One possible race was not checked: the parent's `_update_activity(..., "resume")` closing the task activity while `_user_turn_completed_task` is still unwinding from `StopResponse`.
- I could not find official docs describing `discard_audio_if_uninterruptible` together with `StopResponse` for a "repeat after me" use case.
- `ChatMessage.text_content` is used throughout the repo (for example in `tests/fake_llm.py`), but I did not open its definition.

## Q2. Testing: text-only `AgentSession`, `RunResult.expect`, `mock_tools`, fake LLMs, and multi-input task flows

### Takeaway
Text-only pytest needs no room or audio: `async with AgentSession(llm=...) as session: await session.start(MyAgent()); result = await session.run(user_input="...")`. `RunResult.expect` gives `next_event`, `is_*`, `contains_*`, `skip_*`, `no_more_events`, and an LLM `judge`, and `mock_tools(AgentCls, {...})` swaps tool *execution* while keeping the real schemas.

The docs assume a live LLM. The repo ships a deterministic `FakeLLM` in `tests/fake_llm.py`; it is not in the wheel, so copy it. The repo's own tests drive a tool that awaits an `AgentTask` across several sequential `session.run()` calls.

Caveat for Callback: `session.run()` goes through `generate_reply()`, so `on_user_turn_completed`, and any `StopResponse` capture in it, never runs in text tests. Either test that hook directly, capture in `llm_node`, or copy the repo's fake-audio harness.

### Cited Findings
- **Docs setup.** Install `pytest` and `pytest-asyncio` (`uv add pytest pytest-asyncio`) and mark tests `@pytest.mark.asyncio` — [unit-tests.md "Installation", "Test setup"](https://docs.livekit.io/testing/unit-tests.md). The docs test, copied:
  ```python
  @pytest.mark.asyncio
  async def test_assistant_greeting() -> None:
      async with (
          inference.LLM(model="google/gemma-4-31b-it") as llm,
          AgentSession(llm=llm) as session,
      ):
          await session.start(Assistant())
          result = await session.run(user_input="Hello")
          await result.expect.next_event().is_message(role="assistant").judge(
              llm, intent="Makes a friendly introduction and offers assistance.")
          result.expect.no_more_events()
  ```
- The repo's pytest config sets `asyncio_mode = "auto"` and `asyncio_default_fixture_loop_scope = "function"` — [pyproject.toml L178-L180](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/pyproject.toml#L178-L180)
- **`session.run`.**
  - Signature: `run(*, user_input: str, input_modality: Literal["text","audio"]="text", output_type: type[Run_T] | None = None, output_options=NOT_GIVEN) -> RunResult[Run_T]`.
  - It raises `RuntimeError("nested runs are not supported")` if a run is in progress.
  - Its body is just `self.generate_reply(user_input=user_input, input_modality=input_modality)`.
  - Source: [agent_session.py L864-L884](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent_session.py#L864-L884)
- **What a `RunResult` holds** — [run_result.py L97-L276](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/run_result.py#L97-L276):
  - `events` (`ChatMessageEvent`, `FunctionCallEvent`, `FunctionCallOutputEvent`, `AgentHandoffEvent`), `expect`, `final_output` (raises if the run is not done or has no output), `done()`, and it is awaitable.
  - Only items from *watched* speech handles are recorded.
  - A speech error propagates as an exception when you await the result.
  - With `output_type`, the final output is the last handle's `_maybe_run_final_output`, which `AgentTask.complete()` sets. A wrong type raises `UnexpectedModelBehavior`.
- **Assertion API** — [run_result.py L316-L1108](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/run_result.py#L316-L1108):
  - `RunAssert.__getitem__(int | slice)`.
  - `next_event(*, type=None | "message" | "function_call" | "function_call_output" | "agent_handoff")`.
  - `skip_next_event_if(...)`, `skip_next(count=1)`, `no_more_events()`.
  - `contains_function_call(*, name, arguments)`, `contains_message(*, role)`, `contains_function_call_output(...)`, `contains_agent_handoff(...)`.
  - `EventAssert.is_function_call(*, name, arguments)` does a *partial* key match on `json.loads(item.arguments)`.
  - `is_function_call_output(*, output, is_error)` does an exact string match.
  - `is_message(*, role)` and `is_agent_handoff(*, new_agent_type)`.
  - `ChatMessageAssert.judge(llm_v: llm.LLM, *, intent: str)` is async and calls the LLM with a `check_intent(success, reason)` tool.
- The docs add a pitfall: `next_event(type="function_call")` returns a type-specific assert with no `is_*` methods, so chain `next_event().is_function_call(name=...)` instead — [unit-tests.md "Return types for next_event"](https://docs.livekit.io/testing/unit-tests.md)
- **`mock_tools`** — [run_result.py L1110-L1160](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/run_result.py#L1110-L1160):
  - Signature: `mock_tools(agent: type[Agent], mocks: dict[str, Callable], *, session: AgentSession | None = None)`.
  - Docstring: "Mocks intercept tool *execution* only; the LLM keeps seeing the real tool schemas. A mock may declare any subset of the real tool's parameters."
  - Without `session` it is a context manager; with `session` it applies for the session's lifetime. Context-manager mocks win over session mocks.
  - Sync or async callables both work.
- The docs add that returning an `Exception` from a mock makes the tool raise, for example `with mock_tools(Assistant, {"lookup_weather": lambda: RuntimeError("down")}):`. Import with `from livekit.agents import mock_tools` — [unit-tests.md "Mocking tools"](https://docs.livekit.io/testing/unit-tests.md)
- **API keys.** "The testing helpers work live against your LLM provider"; CI needs provider keys, or `LIVEKIT_API_KEY`/`SECRET` for LiveKit Inference — [unit-tests.md "Integrating with CI"](https://docs.livekit.io/testing/unit-tests.md). `judge()` always calls a real LLM (code above).
- **Scripted fake LLM (repo only).** `tests/fake_llm.py` defines `FakeLLMResponse(input: str, content: str, ttft: float, duration: float, tool_calls: list[FunctionToolCall] = [])` and `FakeLLM(fake_responses=[...])` — [tests/fake_llm.py](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/tests/fake_llm.py)
  - The lookup key is the last chat item's text when that item is a `user` or `system` message (which covers `generate_reply(instructions=...)`), or `output` when it is a `function_call_output`.
  - An unknown key gives an empty response. Any other last item raises `ValueError("No input text found")`.
  - It imports only public names: `livekit.agents.llm.{LLM, ChatChunk, ChatContext, ChoiceDelta, FunctionToolCall, LLMStream, Tool, ToolChoice}` and `livekit.agents.types.{DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr}`.
- **Multi-input `AgentTask` flow (repo test).** The pattern to copy, abridged from [tests/test_nested_agent_task.py L114-L160](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/tests/test_nested_agent_task.py#L114-L160):
  ```python
  llm = FakeLLM(fake_responses=[
      FakeLLMResponse(input="enter_greeting", content="", ttft=1.0, duration=1.0,
          tool_calls=[FunctionToolCall(name="start_task", arguments="{}", call_id="call_1")]),
      FakeLLMResponse(input="hi", content="hello!", ttft=1.0, duration=2.0),
      FakeLLMResponse(input="task_greeting", content="hello from task", ttft=0, duration=0),
      FakeLLMResponse(input="bye", content="", ttft=0, duration=0,
          tool_calls=[FunctionToolCall(name="finish", arguments="{}", call_id="call_2")]),
      FakeLLMResponse(input="task completed", content="all done", ttft=0, duration=0),
  ])
  async with AgentSession(llm=llm) as sess:
      await sess.start(EnterHandoffAgent())
      await asyncio.wait_for(sess.run(user_input="hi"), timeout=5.0)
      assert isinstance(sess.current_agent, SimpleTask)
      await asyncio.wait_for(sess.run(user_input="bye"), timeout=5.0)
      assert isinstance(sess.current_agent, EnterHandoffAgent)
  ```
  `test_nested_agent_task_no_deadlock` states the contract: "session.run() must return when an AgentTask hands off to a nested task that collects additional user input before completing" — [L61-L70](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/tests/test_nested_agent_task.py#L61-L70)
- **Docs tips for task tests** — [tasks.md "Best practices for testing task groups"](https://docs.livekit.io/agents/logic/tasks.md):
  - Output from `on_enter` before the first `run()` is not in the `RunResult`.
  - Prefer `contains_function_call` over `next_event`.
  - Initialise `userdata`.
  - With `TaskGroup`, `await asyncio.sleep(0.5)` before the first `run()` to avoid "trying to generate reply without an LLM model".
- The survey example test follows these tips: a real `inference.LLM`, a `_drive_until_called` helper, and `summarize_chat_ctx=False` in tests — [examples/survey/test_survey_agent.py L1-L40](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/examples/survey/test_survey_agent.py#L1-L40). Other example tests: [examples/drive_thru/test_agent.py](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/examples/drive_thru/test_agent.py), [examples/frontdesk/test_agent.py](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/examples/frontdesk/test_agent.py)
- **Job context in tests.**
  - 1.8.10 ships `livekit.agents.testing.fake_job_context(*, room=None, job_id="fake-job", job_metadata="", agent_identity="fake-agent", inference_executor=None)`, a context manager that installs a real `JobContext` marked `fake_job=True` so `get_job_context()` works — [testing.py L27-L93](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/testing.py#L27-L93)
  - In a fake job, `delete_room()`, `add_sip_participant()` and `transfer_sip_participant()` are logged no-ops — [job.py L758-L822](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/job.py#L758-L822)
  - `get_job_context(*, required: bool = True)` has a `required=False` overload returning `JobContext | None` — [job.py L94-L101](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/job.py#L94-L101)
  - **Docs contradict this:** "`get_job_context()` is unavailable in test environments and raises a `RuntimeError`… mock the call using `unittest.mock`" — [unit-tests.md "Considerations"](https://docs.livekit.io/testing/unit-tests.md)
- **Room access in tests.** `session.room_io` raises `RuntimeError("Cannot access room_io: the AgentSession was not started with a room.")` in roomless tests — [agent_session.py L1255-L1260](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent_session.py#L1255-L1260)
- **Fake-audio harness (repo only)** — [tests/fake_session.py L32-L200](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/tests/fake_session.py#L32-L200):
  - `create_session(actions: FakeActions, *, speed_factor=1.0, turn_handling=None, extra_kwargs=None, can_pause_audio=False, with_stt=True)` and `run_session(session, agent, *, drain_delay=5, record=NOT_GIVEN)`.
  - `FakeActions.add_user_speech(start_time, end_time, transcript, *, stt_delay=0.2, final=True)`, `add_llm(content, tool_calls=None, *, input=..., ttft=0.1, duration=0.3)`, and `add_tts(...)`.
  - It uses `FakeVAD`/`FakeSTT`/`FakeTTS`/fake audio I/O with VAD endpointing (`turn_detection=None`), so the real end-of-turn path, and therefore `on_user_turn_completed`, executes. `tests/test_agent_session.py` overrides the hook — [L96](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/tests/test_agent_session.py#L96)
- **Verbose output.** `LIVEKIT_EVALS_VERBOSE=1 uv run pytest -s -o log_cli=true` prints each `RunResult` — [unit-tests.md "Verbose output"](https://docs.livekit.io/testing/unit-tests.md)

### Inferences
- **Fully offline tests.** Copy `tests/fake_llm.py` into your test dir (Apache-2.0). Use `ttft=0`/`duration=0`; the repo tests use larger values only because they run under a repo-specific virtual-time fixture (`pytest.mark.virtual_time`, registered in the repo `conftest.py`). Don't copy those markers.
- **`say()` in roomless tests.** Since `say()` only requires TTS when an enabled audio output exists, `session.say(...)` inside a task's `on_enter` should work in a roomless text test and complete immediately.
- **Sketch: tool, then gate, then challenge task, in text tests.** This needs the `llm_node`-capture variant from Q1, because `run()` never calls `on_user_turn_completed`:
  ```python
  # tests/test_gate.py   (pytest-asyncio; asyncio_mode="auto" or @pytest.mark.asyncio)
  from livekit.agents import AgentSession
  from livekit.agents.llm import FunctionToolCall
  from .fake_llm import FakeLLM, FakeLLMResponse   # copied from livekit/agents@1.8.10 tests/

  async def test_send_money_requires_voice_challenge():
      llm = FakeLLM(fake_responses=[
          FakeLLMResponse(input="send 500 dollars to Bob", content="", ttft=0, duration=0,
              tool_calls=[FunctionToolCall(name="send_money", arguments='{"amount": 500}', call_id="c1")]),
          FakeLLMResponse(input="transfer sent", content="Done, it's sent.", ttft=0, duration=0),
      ])
      gate = FakeGate(step_ups=["voice"])            # your detection-service stub
      async with AgentSession(llm=llm) as session:
          await session.start(BankAgent(gate=gate))
          await session.run(user_input="send 500 dollars to Bob")
          assert isinstance(session.current_agent, VoiceChallengeTask)
          r2 = await session.run(user_input="4 7 2, 9 1 8, maple, copper, river")
          assert gate.scored_transcripts == ["4 7 2, 9 1 8, maple, copper, river"]
          assert isinstance(session.current_agent, BankAgent)
          r2.expect.contains_message(role="assistant")
  ```
- **Testing the hook-based (audio) capture without audio.** Call it directly: `with pytest.raises(StopResponse): await task.on_user_turn_completed(ChatContext(), ChatMessage(role="user", content=["4 7 2"]))`, then assert `task.done()`. For an end-to-end audio-path test, copy `fake_session.py` and its fakes.
- **Callback-leg code** (`ctx.api...`, `session.room_io`) is not exercisable in roomless tests. Gate it behind an injectable "telephony" interface and assert on calls to a fake.

### Gaps
- I did not run any of these tests. Event ordering for the *first* `run()` was not verified (for example whether the `send_money` `FunctionCallEvent` lands in `r1` or `r2`), so the sketch asserts on `current_agent` and the gate, as the repo test does.
- Not verified: whether `fake_job_context()` with an unconnected `rtc.Room()` is enough for `session.start(room=...)`. Its docstring shows a connected room.
- `contains_function_call_output` exists, but I did not read its exact keyword arguments.

## Q3. Callback leg mechanics in one live session

### Takeaway
One `AgentSession` can survive the caller leaving and re-attach to a newly dialed SIP participant. 1.8.10 has `session.room_io.set_participant(identity)` (and `unset_participant()`), and the official AMD example calls it *before* dialing.

Start the session with `RoomOptions(close_on_disconnect=False)`: a caller hangup is `CLIENT_INITIATED`, which otherwise auto-closes the session. Dial with `ctx.api.sip.create_sip_participant(CreateSIPParticipantRequest(..., wait_until_answered=True), timeout=...)`, catching `api.SipCallError` and `asyncio.TimeoutError`, then run the voice challenge.

Re-linking swaps the `rtc.AudioStream` feeding the session's single audio channel; RoomIO does not restart `stt_node`. To end only the caller's leg, use `ctx.api.room.remove_participant(...)`. To end everything, use `ctx.delete_room()`, `EndCallTool`, or `session.shutdown()` followed by `ctx.shutdown()`.

### Cited Findings
- **Auto-close options and reasons.**
  - `RoomOptions.close_on_disconnect: bool = True`: "Close the AgentSession if the linked participant disconnects with reasons in CLIENT_INITIATED, ROOM_DELETED, or USER_REJECTED".
  - `participant_identity` "Can be overridden by the `participant` argument of RoomIO constructor or `set_participant`".
  - `delete_room_on_close: bool = False`.
  - `DEFAULT_CLOSE_ON_DISCONNECT_REASONS = [CLIENT_INITIATED, ROOM_DELETED, USER_REJECTED]`.
  - Source: [room_io/types.py L17-L21, L110-L131](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/room_io/types.py#L110-L131)
- **What fires when the linked participant leaves** (`RoomIO._on_participant_disconnected`, registered on the room's `participant_disconnected` event at L189) — [room_io.py L457-L489](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/room_io/room_io.py#L457-L489):
  - It ignores participants other than the linked one.
  - It records an OTel session-span event `participant_disconnected` with the disconnect reason. This is a trace event, not a public session event.
  - It resets the link (so `linked_participant` becomes `None`) and cancels the user-away timer.
  - Only if `close_on_disconnect` is set and the reason is in the defaults does it call `_close_soon(reason=CloseReason.PARTICIPANT_DISCONNECTED)`, logging "(disable via `RoomInputOptions.close_on_disconnect=False`)".
- **Close reasons and events.** `CloseReason` values are `ERROR`, `JOB_SHUTDOWN`, `PARTICIPANT_DISCONNECTED`, `USER_INITIATED`, and `TASK_COMPLETED`. `CloseEvent(type="close", error, reason, created_at)` — [events.py L589-L603](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/events.py#L589-L603). `session.shutdown(*, drain=True)` closes with `USER_INITIATED`; `_close_soon` defaults to `drain=False` — [agent_session.py L1263-L1277](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent_session.py#L1263-L1277). RoomIO deletes the room on session close when `delete_room_on_close` is set — [room_io.py L549](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/room_io/room_io.py#L549)
- **SIP disconnect reasons per docs** — [sip-participant.md "Disconnect reasons"](https://docs.livekit.io/reference/telephony/sip-participant.md):
  - `USER_REJECTED`: outbound, pre-connection; auto-closes the session.
  - `USER_UNAVAILABLE` (408/480): does **not** auto-close; "Your code must call `ctx.shutdown()` explicitly".
  - `SIP_TRUNK_FAILURE` (5xx): does not auto-close.
  - `CLIENT_INITIATED`: either side hung up after connect, and "Inbound BYE messages are mapped to this reason"; auto-closes.
  - `ROOM_DELETED`: auto-closes.
- **`PARTICIPANT_REMOVED`.** Protocol enum value `PARTICIPANT_REMOVED = 4`, described as "RoomService.RemoveParticipant was called". It is not in RoomIO's default close list — [livekit_models.proto (main)](https://github.com/livekit/protocol/blob/main/protobufs/livekit_models.proto); [room_io/types.py L17-L21](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/room_io/types.py#L17-L21)
- **Re-linking API** — [room_io.py L300-L360](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/room_io/room_io.py#L300-L360):
  - `RoomIO.set_participant(participant_identity: str | None)` "Switch[es] audio and video streams to specified participant". `None` delegates to `unset_participant()`.
  - If the identity is already in the room, it links immediately and calls `session._on_room_io_participant_linked`.
  - Otherwise it stores the identity, and `_on_participant_connected` links it on join, subject to the `participant_kinds` filter.
  - It propagates the identity to the audio input, video input and user transcription output.
  - Other members: `linked_participant`, `unset_participant()`, and `wait_for_ready()`; see also [L401-L455](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/room_io/room_io.py#L401-L455)
- **Audio input behaviour on a switch** — [room_io/_input.py L135-L210](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/room_io/_input.py#L135-L210):
  - "set_participant can be called before the participant is connected".
  - It closes the current stream and subscribes to the new participant's track.
  - Each track's `_forward_task` pushes frames into the same `_data_ch`.
  - The STT node is built once over that input (`self._stt_node(self._audio_input(), ModelSettings())`) — [audio_recognition.py L237](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/audio_recognition.py#L237)
- **Effects of linking a participant** (`_on_room_io_participant_linked`) — [agent_session.py](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent_session.py):
  - It resets the away timer.
  - For an outbound SIP participant (kind SIP without the dispatch-rule attribute), it disables AEC warm-up. Otherwise the default `_DEFAULT_AEC_WARMUP_DURATION = 3.0` s applies ([L118](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent_session.py#L118)), during which STT gets silence ([agent_activity.py L1693-L1712](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent_activity.py#L1693-L1712)).
- **Test coverage of unlink and re-link.** `test_roomio_away_timer_waits_for_linked_participant` starts with `RoomOptions(..., close_on_disconnect=False)`, emits `participant_disconnected`, and asserts `linked_participant is None` with the user state staying `listening` while unlinked. It then re-links via `set_participant`/`participant_connected` and asserts the timer resumes — [tests/test_room_io.py L544-L620](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/tests/test_room_io.py#L544-L620)
- **Official dial-then-continue example** — [examples/telephony/amd.py L60-L103](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/examples/telephony/amd.py#L60-L103), copied with trimming:
  ```python
  session.room_io.set_participant(participant_identity)        # focus before dialing (L73)
  try:
      await ctx.api.sip.create_sip_participant(
          api.CreateSIPParticipantRequest(room_name=ctx.room.name, sip_trunk_id=outbound_trunk_id,
              sip_call_to=phone_number, participant_identity=participant_identity,
              wait_until_answered=True),
          timeout=45)   # "The API timeout must outlast the ring window"
  except (api.SipCallError, asyncio.TimeoutError) as e:
      ctx.shutdown("call not answered"); return
  # "The call may end just before wait_until_answered returns."
  participant = ctx.room.remote_participants.get(participant_identity)
  if participant is None:
      ctx.shutdown("participant missing"); return
  ```
  The same example registers `ctx.add_shutdown_callback(hangup)`, which deletes the room.
- **Outbound outcomes** — [outbound-calls.md "Handling call outcomes"](https://docs.livekit.io/telephony/making-calls/outbound-calls.md):
  - Answered (200): `wait_until_answered` returns and `sip.callStatus = active`.
  - 486/603: `SipCallError`, reason `USER_REJECTED`.
  - 408/480: `SipCallError`, reason `USER_UNAVAILABLE`.
  - 5xx: `SipCallError`, reason `SIP_TRUNK_FAILURE`.
  - Voicemail answers with 200 and is "not a failure" (use AMD).
  - Error fields: `except api.SipCallError as e: e.sip_status_code, e.sip_status`.
- **`sip.callStatus` values:** `active`, `automation` (connected, still dialing DTMF), `dialing`, `hangup`, `ringing` — [sip-participant.md attributes](https://docs.livekit.io/reference/telephony/sip-participant.md)
- **Mid-call disconnects.** Docs pattern: `@ctx.room.on("participant_disconnected")` and branch on `participant.disconnect_reason` (`rtc.DisconnectReason.CLIENT_INITIATED`, ...) — [outbound-calls.md "Handling mid-call disconnections"](https://docs.livekit.io/telephony/making-calls/outbound-calls.md)
- **`ctx.add_sip_participant` lacks answer detection.** `JobContext.add_sip_participant(*, call_to, trunk_id, participant_identity, participant_name="SIP-participant")` builds a request **without** `wait_until_answered` — [job.py L783-L822](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/job.py#L783-L822). Use `ctx.api.sip.create_sip_participant` directly, as `amd.py` and `WarmTransferTask` do.
- **Removing one participant** (docs snippet; page URL not resolved, cited via the combined docs file) — [docs llms-full.txt](https://docs.livekit.io/llms-full.txt); signature `remove_participant(self, remove: RoomParticipantIdentity) -> RemoveParticipantResponse` — [python-sdks room_service.py L190-L202 (main)](https://github.com/livekit/python-sdks/blob/main/livekit-api/livekit/api/room_service.py#L190-L202):
  ```python
  from livekit.api import RoomParticipantIdentity
  await lkapi.room.remove_participant(RoomParticipantIdentity(room=room_name, identity=identity))
  ```
  The same docs passage notes that "On LiveKit Cloud, the cutoff stays in effect for 10 minutes after the `RemoveParticipant` call" (token revocation).
- **Ending the whole call** — [outbound-calls.md "Hang up"](https://docs.livekit.io/telephony/making-calls/outbound-calls.md):
  - Use `EndCallTool`, or the custom pattern `await ctx.wait_for_playout()` then `await get_job_context().delete_room()`.
  - `JobContext.delete_room(room_name=None) -> asyncio.Future[DeleteRoomResponse]` swallows `NOT_FOUND` — [job.py L758-L781](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/job.py#L758-L781)
  - `JobContext.shutdown(reason="user requested")` — [job.py L876-L877](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/job.py#L876-L877)
- **`EndCallTool`** — [beta/tools/end_call.py](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/beta/tools/end_call.py):
  - Import with `from livekit.agents.beta.tools import EndCallTool`.
  - Signature: `EndCallTool(*, extra_description="", delete_room=True, end_instructions="say goodbye to the user", ignore_on_enter=False, on_tool_called=None, on_tool_completed=None)`. It is a `Toolset` exposing an `end_call` tool.
  - When that speech finishes it calls `session.shutdown()`.
  - On session `close`, it adds a shutdown callback that runs `job_ctx.delete_room()` (if `delete_room`) and calls `job_ctx.shutdown(reason=ev.reason.value)`.
- **Warm transfer (prebuilt)** — [beta/workflows/warm_transfer.py L205-L451](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/beta/workflows/warm_transfer.py#L205-L451):
  - `WarmTransferTask` dials into a **separate room** `<room>-human-agent`, running a **second** `AgentSession` with `RoomOptions(close_on_disconnect=True, delete_room_on_close=True, participant_identity=...)`.
  - The request is `CreateSIPParticipantRequest(..., wait_until_answered=True)` with an optional `ringing_timeout`.
  - While dialing, it mutes the caller session through `session.input.set_audio_enabled(False)` and `session.output.set_audio_enabled(False)`/`set_transcription_enabled(False)`.
  - It merges the calls with `room.move_participant(MoveParticipantRequest(room, identity, destination_room))`.
  - Example apps: [examples/warm-transfer/](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/examples/warm-transfer/warm_transfer.py)
- **Dependency pins.** livekit-agents 1.8.10 pins `livekit==1.1.21`, `livekit-api>=1.2.0,<2`, `livekit-protocol>=1.1.29,<2` — [livekit-agents/pyproject.toml L30-L33](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/pyproject.toml#L30-L33)
- **Deepgram keep-alive.** The Deepgram v1 STT stream sends `{"type": "KeepAlive"}` from a keepalive task — [deepgram/stt.py L481, L622-L640](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-plugins/livekit-plugins-deepgram/livekit/plugins/deepgram/stt.py#L622-L640). The Flux `stt_v2.py` has no KeepAlive; it has a `CloseStream` message and a reconnect loop with `liveness_timeout` — [deepgram/stt_v2.py L323, L467-L530](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-plugins/livekit-plugins-deepgram/livekit/plugins/deepgram/stt_v2.py#L467-L530)
- **2.0.0b1 only (do not adopt).**
  - `AgentTask` becomes `AgentTask(Agent, Generic[TaskResult_T, TaskState_T])`, documented as "An agent with a job that ends with a value: `AgentTask[Output, State]`… `self.state` its own typed state". `instructions` gains a default `""`.
  - `set_participant`, `mock_tools`, `fake_job_context`, `StopResponse`, `EndCallTool`, and `RunResult` exist in both versions.
  - Source: [2.0.0b1 agent.py](https://github.com/livekit/agents/blob/livekit-agents%402.0.0b1/livekit-agents/livekit/agents/voice/agent.py)

### Inferences
- **Recommended single-session callback flow.** Untested; assembled from the cited APIs.
  ```python
  # entrypoint
  await session.start(agent=BankAgent(...), room=ctx.room,
                      room_options=room_io.RoomOptions(close_on_disconnect=False))   # we own hangups
  # because close_on_disconnect=False, end the call yourself on unplanned hangups:
  @ctx.room.on("participant_disconnected")
  def _on_left(p: rtc.RemoteParticipant):
      if p.identity == current_linked_identity and not callback_in_progress:
          asyncio.create_task(_end_call())   # session.shutdown(); ctx.delete_room() / ctx.shutdown()

  class CallbackTask(AgentTask[CallbackResult]):          # awaited from the gated tool
      async def on_enter(self) -> None:
          ctx = get_job_context()
          await self.session.say("For your security I'll hang up and call you back on the number we have on file.",
                                 allow_interruptions=False, add_to_chat_ctx=False)
          await ctx.api.room.remove_participant(api.RoomParticipantIdentity(room=ctx.room.name, identity=self._caller))
          callee = f"callback-{uuid4().hex[:8]}"           # fresh identity, not the removed caller's
          self.session.room_io.set_participant(callee)    # link before dialing (amd.py pattern)
          try:
              await ctx.api.sip.create_sip_participant(api.CreateSIPParticipantRequest(
                  room_name=ctx.room.name, sip_trunk_id=self._trunk_id, sip_call_to=self._number_on_file,
                  participant_identity=callee, wait_until_answered=True), timeout=45)
          except api.SipCallError as e:                   # 486/603 busy/decline, 408/480 no answer, 5xx trunk
              self.complete(CallbackResult(answered=False, sip_status=e.sip_status_code)); return
          except asyncio.TimeoutError:
              self.complete(CallbackResult(answered=False, sip_status=None)); return
          if ctx.room.remote_participants.get(callee) is None:
              self.complete(CallbackResult(answered=False, sip_status=None)); return
          res = await VoiceChallengeTask(self._phrase)    # nested task awaited from on_enter: allowed
          self.complete(CallbackResult(answered=True, challenge=res))
  ```
- **Two ways the session stays up after the caller leaves.**
  - If the agent removes the caller via `remove_participant`, the caller's reason should be `PARTICIPANT_REMOVED`, which RoomIO never auto-closes on.
  - If the *caller* hangs up first, the reason is `CLIENT_INITIATED`, which does close the session unless `close_on_disconnect=False`. That makes `close_on_disconnect=False` plus your own `participant_disconnected` handler the robust choice. It also covers `USER_UNAVAILABLE`/`SIP_TRUNK_FAILURE`, which never auto-close even by default.
- **Away timer and nested-task completion.** With `close_on_disconnect=False`, an `AgentTask` that is waiting when its participant leaves just keeps waiting; nothing completes it automatically. Complete or cancel it from your `participant_disconnected` handler, or rely on the task's own timer. The away timer is cancelled while unlinked, so `user_state` won't flip to `away` during the ring gap.
- **STT continuity across the switch.** RoomIO does not restart `stt_node`, so a Deepgram Flux stream may simply see no audio for the 10–45 s gap (caller gone, callee ringing). The plugin's reconnect/liveness logic should recover if Deepgram closes an idle socket, but stale Flux turn state across the gap is possible. Consider calling `session.clear_user_turn()` ([agent_session.py L1744](https://github.com/livekit/agents/blob/livekit-agents%401.8.10/livekit-agents/livekit/agents/voice/agent_session.py#L1744)) after linking the callee; I did not read its docstring.
- **Clipped first words after answer.** The callee may not be subscribed to the agent's track the instant `wait_until_answered` returns. A short pause before the challenge `say`, or waiting on `ctx.wait_for_participant(identity=callee)` as the outbound docs suggest, reduces the chance that the first words are clipped.
- **Fresh callee identity.** Removing a participant revokes its token for 10 minutes on Cloud. Use a fresh identity for the callee rather than reusing the removed caller's.
- **The second-session alternative.** `WarmTransferTask`'s design (a second room plus a second `AgentSession`, then `move_participant`) is the officially exercised alternative if single-session re-linking misbehaves. It costs a second STT/LLM/TTS pipeline and more moving parts.

### Gaps
- No docs page I read states outright that `remove_participant` on a **SIP** participant sends a BYE and ends the PSTN leg. That is my expectation, but it is unverified here. I also could not resolve the canonical docs page URL for the `remove_participant` snippet; the combined file has no per-page URLs.
- I did not verify Deepgram Flux's server-side idle timeout, or whether `stt_v2` reconnects cleanly after a long no-audio gap.
- I did not verify whether awaiting a fresh `session.say(...)` handle *inside a tool body on the same activity* can deadlock behind the tool's own speech. The sketch avoids that by speaking from the `CallbackTask`'s own activity.
- Telnyx- and Twilio-specific SIP status nuances (for example, which carriers return 480 vs 408 on no-answer) were not researched.
- `DEFAULT_PARTICIPANT_KINDS` contents were not re-read. `amd.py` linking a not-yet-joined SIP participant via `set_participant` implies SIP is accepted by default.
