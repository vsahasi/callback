# LiveKit Agents Python framework (v1.8.x, state as of 2026-11-04) for the Callback voice agent

Scope: everything Veer needs to start coding on 2026-11-05. Facts marked "source" were verified by reading the
`livekit-agents==1.8.10` sdist downloaded from PyPI on 2026-11-04 (GitHub links point at the matching tag
`livekit-agents@1.8.10`). Facts marked "local test" were verified by actually installing and importing the packages
on this Apple Silicon Mac with uv 0.10.12 on 2026-11-04 (local install log, not committed).

Side effect of the local test to be aware of: `uv python install 3.13 3.14` was run, so uv-managed
`cpython-3.13.12-macos-aarch64-none` and `cpython-3.14.3-macos-aarch64-none` now exist under
`~/.local/share/uv/python/` with `python3.13`/`python3.14` symlinks in `~/.local/bin/`.

## Q1. Current versions, Python support (3.14?), recommended install

### Takeaway
Use `livekit-agents==1.8.10` (released 2026-10-28) with the matching 1.8.10 plugins; it declares Python `>=3.10,<3.15` and
installs and imports cleanly on **uv-managed arm64 CPython 3.14.3 and 3.13.12**. Veer's existing Homebrew Python 3.14 is an
**x86_64 (Rosetta) build** and the install failed there, so pin a uv-managed arm64 interpreter (`uv python pin 3.13` or `3.14`)
rather than using `/usr/local/bin/python3.14`.

### Cited Findings
- `livekit-agents` latest stable is **1.8.10, uploaded 2026-10-28**; `requires_python = "<3.15,>=3.10"`; classifiers list Python 3.10, 3.11, 3.12, 3.13 and 3.14. Recent releases: 1.8.9 (2026-10-24), 1.8.8 (10-15), 1.8.7 (10-14), **2.0.0b1 pre-release (2026-10-12)**, 1.8.6 (10-09), 1.8.5 (10-06), 1.8.4 (10-01) — [PyPI JSON](https://pypi.org/pypi/livekit-agents/json), [PyPI page](https://pypi.org/project/livekit-agents/)
- Plugin versions are lock-stepped: `livekit-plugins-deepgram` 1.8.10 (2026-10-28, requires `livekit-agents[codecs]>=1.8.10`, `numpy>=1.26`), `livekit-plugins-openai` 1.8.10 (2026-10-28, requires `openai>=3.28,<4`, `httpx2>=2.12,<3`), `livekit-plugins-cartesia` 1.8.10 (2026-10-28), `livekit-plugins-silero` 1.8.10 (2026-10-29, requires `onnxruntime>=1.18`), `livekit-plugins-turn-detector` 1.8.10. All plugins declare `requires_python >=3.10.0` (no upper bound, classifier only lists 3.10) — [deepgram](https://pypi.org/pypi/livekit-plugins-deepgram/json), [openai](https://pypi.org/pypi/livekit-plugins-openai/json), [cartesia](https://pypi.org/pypi/livekit-plugins-cartesia/json), [silero](https://pypi.org/pypi/livekit-plugins-silero/json)
- `livekit-agents` 1.8.10 pins the realtime SDK to **`livekit==1.1.21`** (even though `livekit` 1.2.0 was released 2026-10-27), plus `livekit-api>=1.2.0,<2`, `livekit-protocol>=1.1.29`, `livekit-local-inference>=0.2.7`, `livekit-blingfire~=1.1`, `openai>=3.19,<4`, `av>=14`, `numpy>=1.26`, `onnxruntime` (via silero), `sounddevice>=0.5`, `typer`, `click`, opentelemetry 1.45 — [PyPI JSON](https://pypi.org/pypi/livekit-agents/json), [livekit on PyPI](https://pypi.org/project/livekit/)
- The `livekit` (rtc) wheels are abi-agnostic per-platform wheels incl. `py3-none-macosx_11_0_arm64` — [livekit PyPI JSON](https://pypi.org/pypi/livekit/json)
- Docs install commands use the `~=1.8` specifier with extras, e.g. `uv add "livekit-agents[cartesia]~=1.8"` and `uv add "livekit-agents[openai]~=1.8"`; the Deepgram page installs `livekit-agents[deepgram]` pinned `~=1.8` — [Cartesia plugin page](https://docs.livekit.io/agents/models/tts/cartesia.md), [OpenAI plugin page](https://docs.livekit.io/agents/models/llm/openai.md), [Deepgram plugin page](https://docs.livekit.io/agents/models/stt/plugins/deepgram.md)
- Local test: `uv pip install "livekit-agents[deepgram,openai,cartesia,silero]~=1.8"` resolved and installed **78 packages** on uv-managed arm64 CPython **3.14.3** and **3.13.12**; all of `livekit.agents`, `livekit.plugins.{deepgram,openai,cartesia,silero}`, `livekit.rtc`, `onnxruntime 1.31.1`, `av 19.0.2`, `numpy 2.5.4` imported; resolved `openai==3.46.0`, `livekit==1.1.21`, `livekit-api==1.2.2`, `livekit-local-inference==0.2.7`, `livekit-blingfire==1.2.0`, `httpx2==2.16.0`, `livekit-protocol==1.1.31` — local test log, not committed
- Local test: `uv python list` showed the pre-existing Python 3.14.3 is `cpython-3.14.3-macos-x86_64-none` (`/usr/local/bin/python3.14 -> ../Cellar/python@3.14/...`), i.e. Homebrew Intel; the same install command in a venv on that interpreter failed to resolve (uv printed only a "pre-releases weren't enabled" hint, no compatible set) — local test log (first run), not committed
- `download-files`: the per-script form `python agent.py download-files` still works but prints a deprecation telling you to use `uv run -m livekit.agents download-files` instead; `python -m livekit.agents` exposes `{download-files,start,console}` and `download-files` "Discover[s] installed livekit-plugins-* packages and run[s] their download_files step" (local test ran it: cartesia, openai, silero reported "finished downloading files" in <2 s) — [cli/_legacy.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/cli/_legacy.py) (lines 1945-1969), local test log
- `silero.VAD.load()` returned a VAD in 0.8 s on 3.13 arm64 with no prior download step (model is bundled) — local test log
- The quickstart's stated requirement is "Python >= 3.10" plus uv; it scaffolds with `lk agent init my-agent --template agent-starter-python` after `brew install livekit-cli` and `lk cloud auth` — [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md)

### Inferences
- Recommended setup (combines the verified pieces):
  ```bash
  brew install livekit-cli && lk cloud auth
  mkdir callback-agent && cd callback-agent
  uv python install 3.13            # arm64 managed build (already present on this Mac)
  uv init --python 3.13             # or: uv python pin 3.13
  uv add "livekit-agents[deepgram,openai,cartesia,silero]~=1.8" python-dotenv
  uv run -m livekit.agents download-files
  ```
  3.14 also works today, but 3.13 is the safer choice for any extra wheel you may add later (e.g. `livekit-plugins-turn-detector` pulls `transformers`). Avoid `~=1.8` being satisfied by 2.0.0b1: it is not, because `~=1.8` means `>=1.8,<2` and uv excludes pre-releases by default.
- Because `uv venv --python 3.14` picked the x86_64 Homebrew interpreter before any managed Python existed, Veer should either rely on the now-installed managed builds or pass `--python cpython-3.13-macos-aarch64-none` explicitly the first time.

### Gaps
- I did not capture the full resolver error for the x86_64 venv, so I cannot say which dependency lacked a macOS x86_64 wheel (onnxruntime/av are the usual suspects). Not needed if you use the arm64 interpreter.
- No official LiveKit statement on Python 3.14 support beyond the PyPI metadata; the only 3.14 mention in release notes is a bithuman-plugin fix in 1.8.3 — [releases](https://github.com/livekit/agents/releases).

## Q2. Minimal AgentSession worker today (entrypoint, dispatch, dev vs start, opening line)

### Takeaway
The current (1.8.x) entrypoint style is `server = AgentServer()` + `@server.rtc_session(agent_name="callback")` + `agents.cli.run_app(server)`;
`WorkerOptions(entrypoint_fnc=..., agent_name=...)` is still accepted by `cli.run_app` but the docs no longer show it. Setting `agent_name`
switches to explicit dispatch (no auto-join of every room). `session.start()` connects to the room for you; speak the recording
disclosure from `Agent.on_enter` with `session.say(..., allow_interruptions=False)` and follow with `generate_reply(...)`.

### Cited Findings
- Quickstart `agent.py` (verbatim structure; it uses LiveKit Inference models, swap for plugins as below):
  ```python
  from dotenv import load_dotenv
  from livekit import agents
  from livekit.agents import (
      AgentServer,
      AgentSession,
      Agent,
      inference,
      room_io,
      TurnHandlingOptions,
  )

  load_dotenv(".env.local")


  class Assistant(Agent):
      def __init__(self) -> None:
          super().__init__(instructions="You are a helpful voice AI assistant. ...")


  server = AgentServer()


  @server.rtc_session(agent_name="my-agent")
  async def my_agent(ctx: agents.JobContext):
      session = AgentSession(
          stt=inference.STT(model="assemblyai/universal-3-6-pro", language="en"),
          llm=inference.LLM(model="google/gemma-4-31b-it"),
          tts=inference.TTS(model="fishaudio/s2.1-pro", voice="..."),
          turn_handling=TurnHandlingOptions(turn_detection=inference.TurnDetector()),
      )
      await session.start(
          room=ctx.room,
          agent=Assistant(),
          room_options=room_io.RoomOptions(
              audio_input=room_io.AudioInputOptions(noise_cancellation=...)
          ),
      )
      await session.generate_reply(
          instructions="Greet the user and offer your assistance."
      )


  if __name__ == "__main__":
      agents.cli.run_app(server)
  ```
  — [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md)
- Top-level exports in 1.8.10 include `AgentServer, WorkerOptions, AgentSession, Agent, AgentTask, JobContext, RunContext, function_tool, ToolError, TurnHandlingOptions, room_io, RoomInputOptions, RoomOutputOptions, inference, cli` and the event classes `UserInputTranscribedEvent, ConversationItemAddedEvent, SpeechCreatedEvent, AgentStateChangedEvent, UserStateChangedEvent, FunctionToolsExecutedEvent, MetricsCollectedEvent, AgentFalseInterruptionEvent, ToolExecutionUpdatedEvent` (source `livekit/agents/__init__.py`) — [source tree](https://github.com/livekit/agents/tree/livekit-agents@1.8.10/livekit-agents/livekit/agents)
- `AgentServer.rtc_session(func=None, *, agent_name: str = "", type=ServerType.ROOM, on_request=None, on_session_end=None, on_simulation_end=None)`; agent name resolution order is `LIVEKIT_AGENT_NAME_OVERRIDE` env var, then the explicit `agent_name` arg, then `LIVEKIT_AGENT_NAME` env var, else `""` — [worker.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/worker.py) (lines 485-531)
- `WorkerOptions.agent_name` docstring: "Set agent_name to enable explicit dispatch. When explicit dispatch is enabled, jobs will not be dispatched to rooms automatically. Instead, you can either specify the agent(s) to be dispatched in the end-user's token, or use the AgentDispatch.createDispatch API."; `WorkerOptions` also has `entrypoint_fnc`, `load_threshold` (dev default `inf`, prod `0.7`), `num_idle_processes` — [worker.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/worker.py) (lines 152-224)
- `cli.run_app(server: AgentServer | WorkerOptions)` docstring: "Run the agent via the (deprecated) rich Python CLI. This is the default entry used by `python myagent.py <command>`. The rich CLI lives in `._legacy` and is being phased out in favor of the LiveKit CLI (`lk agent ...`) and the thin interface in `livekit.agents.__main__`." — [cli/cli.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/cli/cli.py) (line 471)
- Local test: `python agent.py --help` lists commands `console`, `start`, `dev`, `connect`, `download-files` — local test log
- Dispatch docs: "When `agent_name` is not set, an agent is automatically dispatched to each new room" (not recommended because no metadata); "Explicit dispatch is the recommended approach for most applications"; "With `agent_name` set, the agent is only assigned to rooms when explicitly dispatched"; CLI: `lk dispatch create --agent-name test-agent --room my-room --metadata '{"user_id": "12345"}'` (room auto-created); token-based: `AccessToken(...).with_room_config(RoomConfiguration(agents=[RoomAgentDispatch(agent_name="test-agent", metadata='{"user_id": "12345"}')]))` and "Agent dispatch from the token only occurs when the room is first created"; SIP: "LiveKit recommends explicit agent dispatch for SIP inbound calls" — [Agent dispatch](https://docs.livekit.io/agents/server/agent-dispatch.md)
- `from livekit.api import AccessToken, RoomConfiguration, RoomAgentDispatch` imports fine from `livekit-api` 1.2.2 — local test log
- Startup modes: `start` (production: log level `info`, drains jobs on SIGINT/SIGTERM up to `drain_timeout`, load balancing, pre-warmed idle processes), `dev` (log `debug`, Python auto-reloads on code changes, `--no-reload` to disable, no drain), `console` (runs locally without connecting to LiveKit, mic/text toggle `Ctrl+T`, `--record`, `--list-devices`, `--input-device/--output-device`), `connect --room my-room [--participant-identity ...]` (joins a specific room, created if missing). Creds: `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`; `lk app env -w` writes them to `.env.local`; Python runs via `lk agent dev` / `lk agent start` / `lk agent console` or `uv run src/agent.py connect --room ...` — [Startup modes](https://docs.livekit.io/agents/server/startup-modes.md)
- `lk agent dev` "runs the agent in development mode. It prints an Agent Console link"; `lk agent create` registers/builds/deploys the first time, `lk agent deploy` afterwards — [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md)
- Job lifecycle: "If you use `AgentSession`, it connects to LiveKit automatically when started"; use `ctx.connect()` only to control timing/options (`await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)`); `participant = await ctx.wait_for_participant()`; telephony tip: wait for the SIP participant before starting the session; `metadata = json.loads(ctx.job.metadata)`; `ctx.add_shutdown_callback(my_shutdown_hook)` (10 s default, `shutdown_process_timeout`); end with `session.shutdown(drain=True)` or `await session.aclose()`; `await ctx.delete_room()` or `delete_room_on_close=True` — [Job lifecycle](https://docs.livekit.io/agents/server/job.md)
- Source confirms `session.start()` schedules `job_ctx.connect()` itself (`tasks.append(asyncio.create_task(job_ctx.connect(), name="_job_ctx_connect"))`) — [agent_session.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/voice/agent_session.py) (line 1138)
- `AgentSession.__init__` (1.8.10) keyword-only params: `stt, vad, llm, tts, turn_handling: TurnHandlingOptions, stt_context_options, tools, tool_handling: ToolHandlingOptions, max_tool_steps=3, use_tts_aligned_transcript, tts_text_transforms, min_consecutive_speech_delay=0.0, expressive=False, userdata, video_sampler, aec_warmup_duration, ivr_detection=False, user_away_timeout=15.0, transcription_timeout=None, session_close_transcript_timeout=2.0, conn_options, loop` plus deprecated `preemptive_generation, min_endpointing_delay, max_endpointing_delay, false_interruption_timeout, resume_false_interruption, allow_interruptions, discard_audio_if_uninterruptible, min_interruption_duration, min_interruption_words, turn_detection, agent_false_interruption_timeout` (all "Use turn_handling=TurnHandlingOptions(...) instead", `target_version="v2.0"`) — [agent_session.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/voice/agent_session.py) (lines 392-445)
- `session.say(text: str | AsyncIterable[str], *, audio=NOT_GIVEN, allow_interruptions=NOT_GIVEN, add_to_chat_ctx=True) -> SpeechHandle`; `session.generate_reply(*, user_input=NOT_GIVEN, instructions=NOT_GIVEN, tool_choice=NOT_GIVEN, tools: list[str]=NOT_GIVEN, allow_interruptions=NOT_GIVEN, chat_ctx=NOT_GIVEN, input_modality="text") -> SpeechHandle`; docstring: "`await handle` waits for the reply to finish and never raises; check `handle.exception()` for the failure instead." Other public API: `session.interrupt(*, force=False)`, `session.update_agent(agent)`, `session.shutdown(*, drain=True)`, `await session.aclose()`, `await session.wait_for_idle()`, `session.commit_user_turn(...)`, `session.clear_user_turn()`, properties `userdata, history, current_agent, current_speech, user_state, agent_state, usage, room_io, tools` — [agent_session.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/voice/agent_session.py) (lines 775-1807)
- `Agent.__init__(*, instructions, id=None, chat_ctx=NOT_GIVEN, tools=None, stt, vad, turn_handling, tool_handling, llm, tts, expressive, min_consecutive_speech_delay, use_tts_aligned_transcript, [deprecated: turn_detection, min_endpointing_delay, max_endpointing_delay, allow_interruptions, mcp_servers])`; hooks `async def on_enter(self)`, `async def on_exit(self)`, `async def on_user_turn_completed(self, turn_ctx, new_message)`; methods `update_instructions`, `update_tools`, `update_chat_ctx`; nodes `stt_node`, `llm_node`, `transcription_node`, `tts_node`, `realtime_audio_output_node` — [agent.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/voice/agent.py) (lines 48-72, 199-491)
- Docs `on_enter` example: `await self.session.generate_reply(instructions="Greet the user and ask how you can help them.")` — [Agents & handoffs](https://docs.livekit.io/agents/logic/agents-handoffs.md)
- Plugin constructors (source): `openai.LLM(*, model: str | ChatModels = "gpt-4.1", api_key, base_url, client, temperature, top_p, parallel_tool_calls, tool_choice, store, max_completion_tokens, timeout, max_retries, reasoning_effort, verbosity, prompt_cache_key, prompt_cache_retention, prompt_cache_breakpoints="auto", ...)` — [openai/llm.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-plugins/livekit-plugins-openai/livekit/plugins/openai/llm.py) (lines 95-125); `cartesia.TTS(*, api_key=None, model="sonic-3", language="en", encoding="pcm_s16le", voice=TTSDefaultVoiceId, speed=None, emotion=None, volume=None, sample_rate=24000, word_timestamps=True, add_phoneme_timestamps=False, ..., text_pacing=False, base_url="https://api.cartesia.ai")` with `TTSModels = Literal["sonic","sonic-2","sonic-lite","sonic-preview","sonic-turbo","sonic-3"]` — [cartesia/tts.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-plugins/livekit-plugins-cartesia/livekit/plugins/cartesia/tts.py) (lines 89-110); `silero.VAD.load(*, min_speech_duration=0.05, min_silence_duration=0.55, prefix_padding_duration=0.5, max_buffered_speech=60.0, activation_threshold=0.5, sample_rate=16000, force_cpu=True, ...)` ("blocking ... recommended to call this method inside your prewarm mechanism") — [silero/vad.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-plugins/livekit-plugins-silero/livekit/plugins/silero/vad.py) (lines 60-84)
- Docs: `openai.LLM(model="gpt-4.1")` is the shown example and `gpt-4.1` is the plugin default; `temperature` default 0.8; `tool_choice` in `'auto' | 'required' | 'none'`; env `OPENAI_API_KEY`. (LiveKit Inference's `openai/gpt-4.1` is marked deprecated, retiring 2027-01-14 — only relevant if you go through Inference instead of the plugin.) — [OpenAI plugin page](https://docs.livekit.io/agents/models/llm/openai.md)
- Docs: `cartesia.TTS(model="sonic-3", voice="f786b574-daa5-4673-aa0c-cbe3e8534c02")`, env `CARTESIA_API_KEY`; sonic-2/sonic-turbo/sonic are retired, docs point to `sonic-3.5` as replacement — [Cartesia plugin page](https://docs.livekit.io/agents/models/tts/cartesia.md)
- Server options: `setup_fnc` is assigned as a property (`server.setup_fnc = prewarm`, `proc.userdata["vad"] = silero.VAD.load()`, later `ctx.proc.userdata["vad"]`); `num_idle_processes` prod default `math.ceil(cpu_count)`, dev default 0; `load_threshold` default 0.7; `drain_timeout` default 1 h; health port 8081 in prod; "AgentSession provisions a bundled Silero VAD automatically, so the default pipeline doesn't need prewarming" — [Server options](https://docs.livekit.io/agents/server/options.md)
- Source: when `vad` is not given, `AgentSession` sets `vad = inference.VAD(model="silero")` (backed by `livekit.local_inference`), tracked as `_using_default_vad` — [agent_session.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/voice/agent_session.py) (lines 649-652), [inference/vad.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/inference/vad.py)

### Inferences
- A minimal Callback worker adapted from the above (all names verified in source):
  ```python
  import json, logging
  from dotenv import load_dotenv
  from livekit import agents, rtc
  from livekit.agents import (
      AgentServer,
      AgentSession,
      Agent,
      JobContext,
      JobProcess,
      TurnHandlingOptions,
      room_io,
  )
  from livekit.plugins import deepgram, openai, cartesia, silero

  load_dotenv(".env.local")
  server = AgentServer()


  def prewarm(proc: JobProcess):
      proc.userdata["vad"] = silero.VAD.load()


  server.setup_fnc = prewarm


  class Receptionist(Agent):
      def __init__(self, chat_ctx=None):
          super().__init__(
              instructions="You are Callback, a bank-style phone assistant ...",
              chat_ctx=chat_ctx,
          )

      async def on_enter(self) -> None:
          # (e) recording disclosure on every call, not interruptible
          await self.session.say(
              "This call is recorded and analyzed for security.",
              allow_interruptions=False,
          )
          self.session.generate_reply(
              instructions="Greet the caller and ask how you can help."
          )


  @server.rtc_session(agent_name="callback")
  async def entrypoint(ctx: JobContext):
      participant = (
          await ctx.wait_for_participant()
      )  # SIP or browser caller; hidden detector is never returned
      session = AgentSession[CallState](
          stt=deepgram.STTv2(model="flux-general-en", eager_eot_threshold=0.4),
          llm=openai.LLM(model="gpt-4.1"),
          tts=cartesia.TTS(model="sonic-3", voice="<voice-id>"),
          vad=ctx.proc.userdata["vad"],
          turn_handling=TurnHandlingOptions(
              turn_detection="stt"
          ),  # Flux owns end-of-turn
          userdata=CallState(call_id=ctx.room.name, caller=participant.identity),
      )
      await session.start(
          room=ctx.room,
          agent=Receptionist(),
          room_options=room_io.RoomOptions(participant_identity=participant.identity),
      )


  if __name__ == "__main__":
      agents.cli.run_app(server)
  ```
  Run locally with `uv run agent.py dev` (legacy CLI, still works) or `lk agent dev`; deploy with `lk agent create` / `lk agent deploy`.
- Waiting for the participant before `session.start()` matters for browser callers: `on_enter` fires as soon as the session starts, so without the wait the disclosure can play before anyone is listening (the docs' telephony tip recommends the same ordering).

### Gaps
- I did not run `dev`/`start` against LiveKit Cloud (no credentials in this session), so hot-reload and registration behavior are from docs only.
- The docs no longer document `WorkerOptions`; it works in 1.8.10 (`cli.run_app` accepts it) but expect removal in 2.0 (2.0.0b1 already exists).

## Q3. Deepgram Flux via the LiveKit plugin: class, model, end-of-turn, word timestamps, epoch-ms mapping

### Takeaway
Use `deepgram.STTv2(model="flux-general-en")` (WebSocket `wss://api.deepgram.com/v2/listen`) with
`turn_handling=TurnHandlingOptions(turn_detection="stt")`; Flux's `EndOfTurn` becomes `FINAL_TRANSCRIPT` + `END_OF_SPEECH`. The plugin
**does keep word-level timestamps**: every `SpeechData` in `FINAL_TRANSCRIPT`/`INTERIM`/`PREFLIGHT` events carries `words: list[TimedString]`
with `start_time`/`end_time`/`confidence` in seconds relative to the STT stream start (reconnect-safe via `start_time_offset`). They are only
visible inside a custom `Agent.stt_node`; `user_input_transcribed` and `conversation_item_added` only carry plain text. Epoch ms =
`(wall clock when the first audio frame entered the stream) + word.start_time`, the same rule the framework uses internally
(`input_started_at = time.time() - frame.duration`).

### Cited Findings
- Docs: "Use `STTv2` for the Flux model"; it connects to Deepgram's `/listen/v2` WebSocket API; parameters `model` (default `flux-general-en`, `flux-general-multi` for 10 languages), `eager_eot_threshold` (valid 0.3–0.9, must be <= `eot_threshold`), `eot_threshold` (valid 0.5–0.9, default 0.7), `eot_timeout_ms`, `keyterm` (str or list; singular), `language_hint` (multi only), `mip_opt_out`, `tags`; Python-only `update_options(...)` applies `eager_eot_threshold/eot_threshold/eot_timeout_ms/keyterm/language_hint` live, while `model/sample_rate/mip_opt_out/tags` reconnect; for turn detection "Set `turn_detection="stt"` in the turn handling options"; "Flux handles end-of-turn itself through a custom phrase endpointing model that uses acoustic and semantic cues. The session's bundled VAD still handles interruptions." Non-Flux `deepgram.STT` uses `/listen/v1` and defaults to `nova-3`. LiveKit Inference alternative: `inference.STT(model="deepgram/flux-general", language="en")` / `"deepgram/flux-general-en"` — [Deepgram plugin page](https://docs.livekit.io/agents/models/stt/plugins/deepgram.md)
- Turn-detection docs: `turn_detection="stt"` is "for when your STT has its own turn detection" and names AssemblyAI and Deepgram Flux; example `AgentSession(turn_handling=TurnHandlingOptions(turn_detection="stt"), stt=...)`; the session's VAD still handles interruptions — [Turn detection](https://docs.livekit.io/agents/logic/turns.md)
- Source constructor: `STTv2(*, model="flux-general-en", sample_rate=16000, audio_chunk_duration_ms=50, eager_eot_threshold=NOT_GIVEN, eot_threshold=NOT_GIVEN, eot_timeout_ms=NOT_GIVEN, keyterm=NOT_GIVEN, tags=NOT_GIVEN, language_hint=NOT_GIVEN, api_key=NOT_GIVEN, http_session=None, base_url="wss://api.deepgram.com/v2/listen", mip_opt_out=False, numerals=False, profanity_filter=False, redact=NOT_GIVEN, keyterms=NOT_GIVEN[deprecated])`; docstring: `audio_chunk_duration_ms` "Defaults to 50. Deepgram recommends 80 ms for Flux"; `eager_eot_threshold` "Disabled by default. Set to 0.3-0.9 to enable preempt[ive generation]"; `eot_threshold` "ranges 0.5-0.9. Defaults to 0.7"; `eot_timeout_ms` "Defaults to 3000"; capabilities `STTCapabilities(streaming=True, interim_results=True, aligned_transcript="word", offline_recognize=False, keyterms=True)`; API key from `DEEPGRAM_API_KEY`; there is no `language` constructor parameter (`STTOptions.language` defaults to `"en"`); docstring's `base_url` default text (`https://api.deepgram.com/v1/listen`) contradicts the code default (`wss://api.deepgram.com/v2/listen`) — [stt_v2.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-plugins/livekit-plugins-deepgram/livekit/plugins/deepgram/stt_v2.py) (lines 71-133)
- Source event mapping (Flux `TurnInfo.event` -> `SpeechEventType`): `StartOfTurn` -> `START_OF_SPEECH` then `INTERIM_TRANSCRIPT` (only if not already speaking); `Update` -> `INTERIM_TRANSCRIPT`; `EagerEndOfTurn` -> `PREFLIGHT_TRANSCRIPT`; `TurnResumed` -> `INTERIM_TRANSCRIPT`; `EndOfTurn` -> `FINAL_TRANSCRIPT` then `END_OF_SPEECH` (the END_OF_SPEECH copy gets `end_time` = offset + last word's `end`); `Error` raises `APIStatusError`; `ConfigureSuccess/Failure` only logged; `RECOGNITION_USAGE` emitted periodically; only `eager_eot_threshold/eot_threshold/eot_timeout_ms` are sent to Deepgram when given — [stt_v2.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-plugins/livekit-plugins-deepgram/livekit/plugins/deepgram/stt_v2.py)
- Source, words are kept (verbatim):
  ```python
  def _parse_transcription(
      language: str, data: dict[str, Any], start_time_offset: float
  ) -> list[stt.SpeechData]:
      transcript = data.get("transcript")
      words = data.get("words")
      if not words:
          return []
      confidence = sum(word["confidence"] for word in words) / len(words) if words else 0
      ...
      sd = stt.SpeechData(
          language=primary_language,
          start_time=data.get("audio_window_start", 0) + start_time_offset,
          end_time=data.get("audio_window_end", 0) + start_time_offset,
          confidence=confidence,
          text=transcript or "",
          source_languages=[LanguageCode(lang) for lang in detected_languages] or None,
          words=[
              TimedString(
                  text=word.get("word", ""),
                  start_time=word.get("start", 0) + start_time_offset,
                  end_time=word.get("end", 0) + start_time_offset,
                  confidence=word["confidence"],
                  start_time_offset=start_time_offset,
              )
              for word in words
          ],
      )
      return [sd]
  ```
  (note: a TurnInfo with an empty `words` list produces **no** event) — [stt_v2.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-plugins/livekit-plugins-deepgram/livekit/plugins/deepgram/stt_v2.py) (lines 783-816)
- `TimedString(str)` fields: `start_time`, `end_time`, `confidence`, `start_time_offset` ("Offset in seconds relative to the start of the audio input stream or session. Used by STT plugins to align words against the session timeline"), `speaker_id` — [types.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/types.py) (lines 183-221)
- `SpeechEventType` members: `START_OF_SPEECH, INTERIM_TRANSCRIPT, PREFLIGHT_TRANSCRIPT, FINAL_TRANSCRIPT, RECOGNITION_USAGE, END_OF_SPEECH` (no END_OF_TURN member). `SpeechData(language, text, start_time=0.0, end_time=0.0, confidence=0.0, speaker_id=None, is_primary_speaker=None, words: list[TimedString] | None = None, source_languages, source_texts, target_languages, target_texts, metadata)`. `SpeechEvent(type, request_id="", alternatives=[], recognition_usage=None, speech_start_time=None, created_at=time.time(), speech_end_time=None)` — [stt.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/stt/stt.py)
- Reconnect handling in the base `SpeechStream`: `_start_time_offset` starts at 0.0 and on every (re)connection attempt `self._start_time_offset += time.time() - last_start_time; self._start_time = time.time()` "so that returned transcripts can have linear timestamps"; `start_time` property docstring: "Wall-clock anchor for the stream. Seeded to `time.time()` when the stream is initialized (and re-seeded on each retry) ... so that server-provided stream-relative timestamps ... can be converted to wall-clock accurately." — [stt.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/stt/stt.py) (lines 407-484)
- The framework's own wall-clock conversion: on the first audio frame pushed to the STT pipeline it sets `self._stt_pipeline.input_started_at = time.time() - frame.duration`, and for STTs with `aligned_transcript` it computes `speech_end_time = self._input_started_at + ev.alternatives[0].end_time` (accepted only if `<= ev.created_at`); it also uses `min(ev.alternatives[0].end_time + self._input_started_at, now)` as the user's last speaking time — [audio_recognition.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/voice/audio_recognition.py) (lines 806-807, 1198-1207, 1248-1258)
- The STT pipeline accumulates the user turn as plain text (`transcript = ev.alternatives[0].text` ... `self._audio_transcript += f" {transcript}"`) and the user `ChatMessage` is created from that string, so word timings are not propagated to `conversation_item_added`; `UserInputTranscribedEvent` has only `transcript, is_final, item_id, speaker_id, language, created_at` — [audio_recognition.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/voice/audio_recognition.py) (lines 1271, 1299, 1592), [events.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/voice/events.py) (lines 345-353)
- Pipeline-node override pattern: `async def stt_node(self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings) -> Optional[AsyncIterable[stt.SpeechEvent]]`, wrapping `async for event in Agent.default.stt_node(self, filtered_audio(), model_settings): yield event`; imports `from livekit import rtc`, `from livekit.agents import ModelSettings, stt, Agent` — [Pipeline nodes](https://docs.livekit.io/agents/build/nodes.md)
- Deepgram side: Flux `TurnInfo` messages carry `event, turn_index, audio_window_start, audio_window_end, transcript, words[] (word, confidence, start, end), end_of_turn_confidence` (+ `trigger` on EndOfTurn); the state machine is Initial -> (StartOfTurn) TurnOngoing -> (EagerEndOfTurn) AwaitingEnd -> (TurnResumed) back, or (EndOfTurn) Initial; `EagerEndOfTurn`/`TurnResumed` only fire when `eager_eot_threshold` is set — [Flux state](https://developers.deepgram.com/docs/flux/state), [Flux force end turn](https://developers.deepgram.com/docs/flux/force-end-turn)
- Deepgram: `wss://api.deepgram.com/v2/listen?model=flux-general-en&encoding=linear16&sample_rate=16000`; "80ms audio chunks strongly recommended"; raw encodings `linear16, linear32, mulaw, alaw, opus, ogg-opus`; sample rates 8000/16000/24000/44100/48000, mono; `eot_timeout_ms` default 5000; `eot_threshold` range 0.5–1.0; word `start`/`end` are `double` "word-level timestamps"; Word-level timestamps are "All available" for Flux languages — [Flux quickstart](https://developers.deepgram.com/docs/flux/quickstart), [Nova-3 -> Flux migration](https://developers.deepgram.com/docs/flux/nova-3-migration), [Flux feature overview](https://developers.deepgram.com/docs/flux/feature-overview)

### Inferences
- Epoch-ms recipe that matches the task's definition and the framework's own math (`t0` = wall clock of the first frame that entered this STT stream):
  ```python
  import time, json
  from typing import AsyncIterable, Optional
  from livekit import rtc
  from livekit.agents import Agent, ModelSettings, stt


  class Receptionist(Agent):
      async def stt_node(
          self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
      ) -> Optional[AsyncIterable[stt.SpeechEvent]]:
          anchor: dict[str, float] = {}

          async def tapped() -> AsyncIterable[rtc.AudioFrame]:
              async for frame in audio:
                  if "t0" not in anchor:
                      anchor["t0"] = (
                          time.time() - frame.duration
                      )  # same rule as audio_recognition.py:807
                  yield frame

          async def events():
              async for ev in Agent.default.stt_node(self, tapped(), model_settings):
                  if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT and ev.alternatives:
                      sd = ev.alternatives[0]
                      t0 = anchor.get(
                          "t0", ev.created_at - sd.end_time
                      )  # fallback if no frame seen
                      words = [
                          {
                              "w": str(w),
                              "start_ms": round((t0 + w.start_time) * 1000),
                              "end_ms": round((t0 + w.end_time) * 1000),
                              "conf": w.confidence,
                          }
                          for w in (sd.words or [])
                      ]
                      self.session.userdata.pending_caller_turn = {
                          "text": sd.text,
                          "words": words,
                          "start_ms": round((t0 + sd.start_time) * 1000),
                          "end_ms": round((t0 + sd.end_time) * 1000),
                      }
                  yield ev

          return events()
  ```
  Because `word.start_time` already includes `start_time_offset`, this stays correct across Deepgram reconnects. Sanity check: `t0 + sd.end_time <= ev.created_at` should always hold (the framework applies the same check).
- One FINAL per Flux turn: with `turn_detection="stt"` the user turn is committed on `END_OF_SPEECH`, so the FINAL immediately preceding it is "the finished caller turn" to send to the detector; `on_user_turn_completed(turn_ctx, new_message)` is the right place to flush it (it has the final text and runs before the LLM reply).
- Each `Agent` handoff (e.g. switching to the enrollment agent) runs a new `stt_node` and therefore a new STT stream whose Deepgram clock restarts at 0, so capture `t0` per `stt_node` invocation (the snippet above does), never once per call.
- Browser-only caveat: `AudioInputOptions.pre_connect_audio=True` (default) can push a burst of audio captured before the agent subscribed; Deepgram's clock then starts at audio that was captured up to `pre_connect_audio_timeout=3.0` s earlier than `t0`, so early words in the very first turn can be stamped late by the burst length. Not an issue on SIP. Disable with `room_io.RoomOptions(audio_input=room_io.AudioInputOptions(pre_connect_audio=False))` if first-turn precision matters.
- `eot_timeout_ms`: the plugin only sends it when you set it, so the effective default is Deepgram's server default (5000 ms per Deepgram), not the 3000 ms the plugin docstring claims; set it explicitly if you care.
- Consider `audio_chunk_duration_ms=80` (Deepgram's recommendation) instead of the plugin default 50.

### Gaps
- Deepgram's docs never state the unit of `audio_window_start/end` and `words[].start/end`; "seconds" is inferred from example values (e.g. 0.42–0.56) and from the plugin adding them to second-based offsets — [Flux state](https://developers.deepgram.com/docs/flux/state).
- Whether `inference.STT(model="deepgram/flux-general-en")` (LiveKit Inference) also populates `words` was not verified; the plugin path (`deepgram.STTv2`) is the verified route.
- I did not verify the exact URL query encoding (`_to_deepgram_url`) or whether `start_time` is set by the plugin to the first-frame send time (the base class seeds it at stream init; the docstring says plugins "may override").

## Q4. Events and timing (states, timestamps, how to detect interruptions)

### Takeaway
Every `AgentSession` event in 1.8.10 is a pydantic model with a `created_at: float` (Unix seconds, `time.time()`) even though the docs page
omits it; `agent_state_changed` (`initializing|idle|listening|thinking|speaking`) and `user_state_changed` (`speaking|listening|away`) give
you `agent_speech_start/end` and `user_speech_start`. Whether the caller interrupted the agent is recorded on the assistant `ChatMessage.interrupted`
(delivered via `conversation_item_added`) and on `SpeechHandle.interrupted`; assistant items also carry wall-clock `metrics.started_speaking_at /
stopped_speaking_at`.

### Cited Findings
- Docs event table: `user_input_transcribed` -> `UserInputTranscribedEvent(transcript, is_final, speaker_id, language)`; `user_state_changed` -> `UserStateChangedEvent(old_state, new_state)`; `agent_state_changed` -> `AgentStateChangedEvent(old_state, new_state)`; `conversation_item_added` -> `ConversationItemAddedEvent(item: ChatMessage)`; `function_tools_executed` -> `FunctionToolsExecutedEvent(function_calls, function_call_outputs, has_tool_reply, has_agent_handoff; zipped(), cancel_tool_reply(), cancel_agent_handoff())`; `metrics_collected` ("deprecated at session level") -> `MetricsCollectedEvent(metrics)`; `speech_created` -> `SpeechCreatedEvent(user_initiated, source, speech_handle)` with `source` in `"say" | "generate_reply" | "tool_response"`; `error`, `close` (`CloseReason`: `error, job_shutdown, participant_disconnected, user_initiated, task_completed`); `AgentState = initializing|idle|listening|thinking|speaking`; `UserState = speaking|listening|away`; `ChatMessage` fields used: `role, text_content, interrupted, content, metrics`; the page documents `created_at` only on `UserTranscriptionTimeoutEvent`, `ToolExecutionUpdatedEvent`, `AgentFalseInterruptionEvent` — [Events reference](https://docs.livekit.io/reference/agents/events.md)
- Source (contradicts/extends the docs): all of `UserStateChangedEvent, AgentStateChangedEvent, UserInputTranscribedEvent, MetricsCollectedEvent, SessionUsageUpdatedEvent, ConversationItemAddedEvent, FunctionToolsExecutedEvent, SpeechCreatedEvent, ToolExecutionUpdatedEvent, UserTurnExceededEvent, ErrorEvent, CloseEvent, AgentFalseInterruptionEvent, EotPredictionEvent` define `created_at: float = Field(default_factory=time.time)`; `UserInputTranscribedEvent` also has `item_id: str | None`; `SpeechCreatedEvent.user_initiated: bool` ("True if the speech was created using public methods like `say` or `generate_reply`") and `source: Literal["say", "generate_reply"]` (no `"tool_response"` in 1.8.10); `ConversationItemAddedEvent.item: ChatMessage | AgentHandoff | _TypeDiscriminator`; `AgentFalseInterruptionEvent(resumed: bool, created_at, message, extra_instructions)`; `EventTypes` also includes `agent_false_interruption`, `user_turn_exceeded`, `session_usage_updated`, `tool_execution_updated`, `eot_prediction` — [events.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/voice/events.py) (lines 309-603)
- `ChatMessage` (source): `id, type="message", role, content: list[ChatContent], interrupted: bool = False, transcript_confidence: float | None, extra: dict, metrics: MetricsReport, created_at: float = time.time()`, property `text_content` — [chat_context.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/llm/chat_context.py) (lines 407-420)
- `MetricsReport` (TypedDict on `ChatMessage.metrics`): `started_speaking_at, stopped_speaking_at, transcription_delay, end_of_turn_delay, on_user_turn_completed_delay, llm_node_ttft, llm_node_tps, llm_node_ttfs, tts_node_ttfb, playback_latency, e2e_latency, provider_request_ids, llm_metadata, tts_metadata, stt_metadata` — [chat_context.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/llm/chat_context.py) (lines 286-361)
- Source: the spoken assistant message is added with `interrupted=speech_handle.interrupted, created_at=started_speaking_at if started_speaking_at is not None else time.time(), metrics=assistant_metrics` where `assistant_metrics["started_speaking_at"]`/`["stopped_speaking_at"]` are wall-clock playout bounds (`stopped_speaking_at = time.time()` when playout ends) and `e2e_latency = started_speaking_at - previous_user.stopped_speaking_at`; its `content` is the text actually heard (`playback_ev.synchronized_transcript` when interrupted) — [agent_activity.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/voice/agent_activity.py) (lines 3362-3431)
- `SpeechHandle` public API: `id, num_steps, scheduled, interrupted, allow_interruptions (get/set), chat_items, done(), exception(), interrupt(...), await wait_for_playout(), add_done_callback(cb), remove_done_callback(cb), await wait_if_not_interrupted([...])`; the handle is awaitable — [speech_handle.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/voice/speech_handle.py) (lines 88-274)
- Interruption semantics (docs): the framework "pauses the agent's speech whenever it detects user speech in the input audio"; `session.interrupt()` also works; when interrupted "the agent truncates its conversation history to what the user actually heard"; false-interruption handling via `turn_handling={"interruption": {"false_interruption_timeout": 2.0, "resume_false_interruption": True}}`; `agent_false_interruption` fires after that much silence — [Turn detection](https://docs.livekit.io/agents/logic/turns.md)
- Interruption defaults (reference): `interruption.enabled=True`, `mode` adaptive (on LiveKit Cloud/dev when conditions met) else `vad`, `discard_audio_if_uninterruptible=True`, `min_duration=0.5` s, `min_words=0`, `false_interruption_timeout=2.0` s, `resume_false_interruption=True`, `backchannel_boundary=(1.0, 1.0)` — [Turn handling options reference](https://docs.livekit.io/reference/agents/turn-handling-options.md)
- Docs import paths shown: `from livekit.agents import UserInputTranscribedEvent`, `from livekit.agents import ConversationItemAddedEvent`, `from livekit.agents.llm import ChatMessage, ImageContent, AudioContent`, `from livekit.agents import SessionUsageUpdatedEvent, ToolExecutionUpdatedEvent, ErrorEvent` — [Events reference](https://docs.livekit.io/reference/agents/events.md)
- 1.8.7 release note: "Adds realtime end-to-end latency measurement with local VAD. The example reads it from assistant message metrics in a `conversation_item_added` handler" — [releases](https://github.com/livekit/agents/releases)

### Inferences
- Minimal event wiring for requirement (b) (all names verified):
  ```python
  from livekit.agents import (
      AgentStateChangedEvent,
      UserStateChangedEvent,
      ConversationItemAddedEvent,
      SpeechCreatedEvent,
  )
  from livekit.agents.llm import ChatMessage


  @session.on("agent_state_changed")
  def _on_agent_state(ev: AgentStateChangedEvent):
      ms = round(ev.created_at * 1000)
      if ev.new_state == "speaking":
          publish({"type": "agent_speech_start", "t_ms": ms})
      elif ev.old_state == "speaking":
          publish({"type": "agent_speech_end", "t_ms": ms})


  @session.on("user_state_changed")
  def _on_user_state(ev: UserStateChangedEvent):
      if ev.new_state == "speaking":
          publish(
              {
                  "type": "user_speech_start",
                  "t_ms": round(ev.created_at * 1000),
                  "agent_was_speaking": session.agent_state == "speaking",
              }
          )


  @session.on("conversation_item_added")
  def _on_item(ev: ConversationItemAddedEvent):
      item = ev.item
      if isinstance(item, ChatMessage) and item.role == "assistant":
          m = item.metrics
          publish(
              {
                  "type": "turn",
                  "speaker": "agent",
                  "text": item.text_content,
                  "start_ms": round(m.get("started_speaking_at", item.created_at) * 1000),
                  "end_ms": round(m.get("stopped_speaking_at", ev.created_at) * 1000),
                  "interrupted": item.interrupted,
              }
          )
  ```
  `publish` must be fire-and-forget from these sync callbacks (e.g. `asyncio.create_task(room.local_participant.publish_data(...))`).
- "Did the caller interrupt the agent?" for the caller turn: `True` if the assistant item immediately before it has `interrupted=True`, or if `session.agent_state == "speaking"` when `user_state_changed -> speaking` fired. `AgentFalseInterruptionEvent.resumed` tells you the interruption was judged false (backchannel) and speech resumed.
- `user_state_changed` is VAD-driven and still fires with `turn_detection="stt"` because the bundled VAD stays active for interruptions; as a second signal, Flux's `StartOfTurn` arrives as `START_OF_SPEECH` inside `stt_node`.
- `agent_state_changed -> speaking` is the realtime signal; `ChatMessage.metrics.started_speaking_at/stopped_speaking_at` is the authoritative after-the-fact pair for the same turn (use it for `start_ms/end_ms` of agent turns).

### Gaps
- Word-level timestamps for the *agent's* words: Cartesia returns word timestamps (`word_timestamps=True` default) and `use_tts_aligned_transcript=True` forwards an aligned transcript, but I did not verify how/where `TimedString`s for TTS surface to user code (likely via `transcription_node` / the TTS stream); treat agent word times as unverified and fall back to `started_speaking_at`-based estimates — [Cartesia plugin page](https://docs.livekit.io/agents/models/tts/cartesia.md).
- The docs' `speech_created.source == "tool_response"` value does not exist in 1.8.10 source; docs may describe 2.0.

## Q5. Function tools: definition, errors, the async/long-running pattern, speaking mid-tool, limiting visible tools

### Takeaway
1.8.10 already ships the documented async-tool machinery: inside a `@function_tool` you get `RunContext` with `await ctx.update(msg)` (hands a
synthetic result to the LLM so it keeps talking), `async with ctx.with_filler("one moment while I verify", delay=...)` (plays `session.say`
filler when idle), `async with ctx.foreground()` (hold the floor for an interactive step), `ctx.disallow_interruptions()`, `ToolFlag.CANCELLABLE`,
`on_duplicate=`; the tool's return value is voiced as a deferred reply when the agent is next idle. To keep Ramp writes behind the gate, never
expose them as LLM tools at all, or restrict per reply with `generate_reply(tools=[...])` / per agent with `tools=[...]` / `agent.update_tools(...)`.

### Cited Findings
- Definition (docs, verbatim pattern):
  ```python
  from livekit.agents import function_tool, Agent, RunContext


  class MyAgent(Agent):
      @function_tool()
      async def lookup_weather(
          self, context: RunContext, location: str
      ) -> dict[str, Any]:
          """Look up weather information for a given location.

          Args:
              location: The location to look up weather information for.
          """
          return {"weather": "sunny", "temperature_f": 70}
  ```
  Decorator params `name, description, raw_schema, flags`; "Argument types are inferred from the function signature"; `RunContext` gives "access to the current `session`, `function_call`, `speech_handle`, and `userdata`"; `context.disallow_interruptions()`; `speech_handle.wait_if_not_interrupted([...])` / `.interrupted`; `raise ToolError("...")` "to return an error to the LLM in place of a response"; unexpected exceptions become a generic internal-error message to the LLM; bad arguments are forwarded as `ToolError`; "Return `None` or nothing at all to complete the tool silently"; handoff: `return SomeAgent(), "Transferring the user to SomeAgent"`; dynamic tools: `await agent.update_tools(agent.tools + [tool_a])`, `await agent.update_tools([t for t in agent.tools if t.id != tool_a.id])`, `await agent.update_tools([tool_a, tool_b])` ("This replaces _all_ tools, including those registered automatically within the agent class"); `ToolFlag.IGNORE_ON_ENTER` "Excludes the tool from any `generate_reply` calls made inside the agent's `on_enter` method"; programmatic `function_tool(func, name=..., description=...)`; raw tools `@function_tool(raw_schema=...)` with `raw_arguments: dict[str, object]`; imports `from livekit.agents.llm import ToolError`, `from livekit.agents.llm.tool_context import ToolFlag` — [Function tools](https://docs.livekit.io/agents/logic/tools/definition.md)
- Source signature: `function_tool(f=None, *, name=None, description=None, raw_schema=None, flags: ToolFlag = ToolFlag.NONE, on_duplicate: DuplicateMode = "allow", duplicate_scope: DuplicateScope = "name")`; `class ToolFlag(Flag): NONE, IGNORE_ON_ENTER, CANCELLABLE`; `class ToolError(Exception)`; `class StopResponse(Exception)` — [tool_context.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/llm/tool_context.py) (lines 142-375)
- Async tools (docs): "Tools that take more than a few seconds block the conversation until they return."; `ctx.update(message)` adds a status to the chat context and "The LLM reads it and voices a reply while the tool keeps running"; `ctx.with_filler(...)` (Python) "plays audio through `session.say()`, bypassing the LLM. Fillers play only after the session has been idle for `delay`"; filler params `source, delay, interval, max_steps` (Python in seconds); `ctx.foreground()` "plays any queued reply, waits for idle, then blocks other agent speech during the interactive step"; "The return value is voiced as a follow-up reply when the agent is next idle"; cancellation is opt-in via `ToolFlag.CANCELLABLE` ("most tools (orders, writes, payments) aren't safe to interrupt partway through") and auto-exposes `lk_agents_get_running_tasks` / `lk_agents_cancel_task`; `on_duplicate` in `allow | reject | replace | confirm` (detected by tool name only); "pending async tool updates are dropped during an agent handoff" unless wrapped in `AsyncToolset`; templates via `tool_handling` -> `async_options` keys `update_template, duplicate_reject_template, duplicate_confirm_template, reply_at_tail_template, reply_maybe_covered_template`; imports `from livekit.agents import Agent, RunContext, function_tool`, `from livekit.agents.llm import ToolFlag`, `from livekit.agents.llm.async_toolset import AsyncToolset`, `from livekit.agents.beta.workflows import GetEmailTask, WorkflowInstructions` — [Async tools](https://docs.livekit.io/agents/logic/tools/async.md)
- Source confirms all of it exists in 1.8.10: `RunContext` has properties `session, speech_handle, function_call, userdata` and methods `disallow_interruptions()`, `await wait_for_playout()` ("only waits for the assistant's spoken response prior running this tool to finish playing"), `async with with_filler(source, *, delay=0, interval=None, max_steps=None)` ("`source`: Either a string (spoken via `session.say`), or a callable `(step: int) -> SpeechHandle | str | None` ... `delay`: Continuous-idle dwell required before each fire. `0` = fire as soon as the session is next idle ... `interval=None` fires at most once"), `async with foreground()`, `await update(message, *, template=None, silent=False)` ("The first update releases control to the LLM with `message` as the tool's synthetic return; subsequent updates are coalesced into a deferred reply ... `silent`: ... True keeps it to the model alone"); `class AsyncToolset(Toolset)` exists; `ToolHandlingOptions(async_options: AsyncToolOptions)`; the deferred-reply system prompt is "New results arrived from background tool calls (call_ids: {call_ids}). Summarize the results naturally. Do NOT repeat information you have already told the user." — [events.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/voice/events.py) (lines 47-225), [tool_executor.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/voice/tool_executor.py) (lines 4-80), [async_toolset.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/llm/async_toolset.py)
- Per-reply tool restriction (source docstring): `generate_reply(tools=[...])` — "List of tool IDs to make available for this response. When set, only the specified tools can be used. Tool IDs must match registered tools on the agent. For function tools, the ID is the function name (accessible via `my_tool.id`)"; `tool_choice` — "If generate_reply is invoked within a function_tool, defaults to 'none'" — [agent_session.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/voice/agent_session.py) (lines 1608-1640)
- `max_tool_steps` default 3: "At the limit, the agent makes one final call with tools disabled" — [Agent sessions](https://docs.livekit.io/agents/logic/sessions.md)
- Tool-related events: `function_tools_executed`, `tool_execution_updated` (`ToolCallStarted | ToolCallUpdated(message, silent) | ToolCallEnded(status done|error|cancelled) | ToolReplyUpdated(status scheduled|completed|interrupted|skipped)`) — [events.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/voice/events.py) (lines 441-549)
- 1.8.3 release note: "adds reply_required support in tool results" — [releases](https://github.com/livekit/agents/releases)

### Inferences
- Gated sensitive action (requirement c), using only verified APIs:
  ```python
  from livekit.agents import function_tool, RunContext
  from livekit.agents.llm import ToolError


  class Receptionist(Agent):
      @function_tool()
      async def request_sensitive_action(
          self, ctx: RunContext[CallState], action: str, details: str
      ) -> str:
          """Request a sensitive account action (card reissue, limit change, payee add). Requires verification.

          Args:
              action: One of card_reissue, limit_change, add_payee.
              details: Any parameters the caller gave.
          """
          ctx.disallow_interruptions()  # don't let barge-in cancel the gate
          async with ctx.with_filler("One moment while I verify that.", delay=0.5):
              verdict = await detector_authorize(
                  ctx.userdata.call_id, action
              )  # HTTP to Eric's /authorize, may take seconds
          if not verdict.allowed:
              raise ToolError(
                  f"Verification failed ({verdict.reason}). Do not perform {action}; offer the secure callback flow."
              )
          result = await ramp_client.perform(
              action, details
          )  # the write happens in Python, never via an LLM-visible tool
          return f"Verified. {action} completed: {result}"
  ```
  The LLM sees exactly one tool for the whole class of sensitive actions; the Ramp SDK call is not a tool, so there is nothing to bypass. If you also want the LLM to narrate progress, `await ctx.update("Verification in progress")` makes it speak from the LLM while the tool continues; `with_filler` is the cheaper, deterministic "one moment" line.
- For a hard mode switch (e.g. elevated risk from the detector), either `await self.update_tools([t for t in self.tools if t.id != "request_sensitive_action"])` or hand off to a restricted `Agent` subclass whose `tools=[]`; for one reply only, `session.generate_reply(instructions=..., tools=["lookup_balance"])`.
- Plain `@function_tool`s defined on the Agent class do not survive handoffs for pending async updates; if a gate check may still be in flight when you switch agents, put the tool in an `AsyncToolset` passed to `AgentSession(tools=[...])`.

### Gaps
- `AsyncToolset`'s constructor/usage was not read in detail (docs show `AsyncToolset` wrapping tools; source only confirms the class exists with `aclose()`).
- Whether preemptive generation can execute a tool call on a *preflight* (eager) transcript before the turn is final was not verified; see Q10.

## Q6. Room data: receiving the detector's "risk" packets, sending turns, hidden participants

### Takeaway
Subscribe with `@ctx.room.on("data_received")` and filter `packet.topic == "risk"`; `DataPacket` has `data: bytes, kind, participant: RemoteParticipant | None, topic: str | None`.
The LiveKit server broadcasts data packets from a **hidden** participant to everyone (no `Hidden()` check in the data path), but hidden
participants are never announced to others, so the Python SDK cannot resolve the sender and `packet.participant` will be `None`; put the sender
in the JSON payload. Send turns with `await ctx.room.local_participant.publish_data(payload, reliable=True, topic="turn")` (15 KiB cap per reliable packet).

### Cited Findings
- Docs Python examples (verbatim):
  ```python
  @room.on("data_received")
  def on_data_received(data: rtc.DataPacket):
      logging.info("received data from %s: %s", data.participant.identity, data.data)


  # string payload will be encoded to bytes with UTF-8
  await room.local_participant.publish_data(
      "my payload",
      reliable=True,
      destination_identities=["identity1", "identity2"],
      topic="topic1",
  )
  ```
  Reliable: "each packet can be up to 15KiB in size", in-order, retransmitted, but best-effort (not buffered server-side; a receiver disconnected at send time never gets it); lossy: keep under ~1300 bytes; topics are free-form strings — [Data packets](https://docs.livekit.io/home/client/data/packets.md)
- Python SDK: `DataPacket(data: bytes, kind: proto_room.DataPacketKind.ValueType, participant: RemoteParticipant | None, topic: str | None = None)`; the docstring says `participant` is `None` when a server SDK sent it; the handler resolves the sender with `self._remote_participants.get(identity, None)` and emits `data_received` without a None check; `LocalParticipant.publish_data(payload: bytes | str, *, reliable: bool = True, destination_identities: list[str] = [], topic: str = "")`; also `send_text(text, *, topic=...)`, `register_text_stream_handler(topic, handler)` / `register_byte_stream_handler(topic, handler)` (one handler per topic, else `ValueError`); room events include `participant_connected, participant_disconnected, participant_attributes_changed, track_subscribed, data_received, sip_dtmf_received, connected, disconnected, reconnecting, reconnected` — [room.py](https://github.com/livekit/python-sdks/blob/main/livekit-rtc/livekit/rtc/room.py) (DataPacket at line 138 in the installed 1.1.21), [participant.py](https://github.com/livekit/python-sdks/blob/main/livekit-rtc/livekit/rtc/participant.py), local test log
- LiveKit server: `broadcastParticipantState` returns early for hidden participants ("hidden participant updates are sent only to the hidden participant itself"); `GetOtherParticipantInfo` filters `!op.Hidden()` so hidden participants are excluded from the participant list sent on join; the data path (`onDataMessage` -> `BroadcastDataPacketForRoom`) has **no** `Hidden()` check, only "skip the source" and optional destination filtering — [livekit/pkg/rtc/room.go](https://github.com/livekit/livekit/blob/master/pkg/rtc/room.go)
- JS SDK doc for `RoomEvent.DataReceived`: "All participants in the room will receive the messages sent to the room."; callback args `(payload, participant: Participant | undefined, kind, topic?, encryptionType?, participantIdentity?)` — [client-sdk-js events.ts](https://github.com/livekit/client-sdk-js/blob/main/src/room/events.ts)
- Docs: "A participant is hidden if their participant permissions has `hidden` set to `true`. You can set this field in the participant's access token. A hidden participant is not visible to other participants in the room." Participant kinds: `STANDARD, AGENT, SIP, EGRESS, INGRESS` — [Participants](https://docs.livekit.io/intro/basics/rooms-participants-tracks/participants.md)
- `livekit.api.VideoGrants` dataclass fields (1.2.2): `room_create, room_list, room_record, room_admin, room_join, room, destination_room, can_publish, can_subscribe, can_publish_data, can_publish_sources, can_update_own_metadata, ingress_admin, hidden, recorder, agent, can_manage_agent_session, can_subscribe_metrics` — local test log
- Agent Console observers join "as a hidden participant" and "can't publish audio or video, but can send text and execute RPC calls" (i.e. hidden participants sending data is a supported pattern) — [Agent Console](https://docs.livekit.io/testing/agent-console.md)
- RoomIO only auto-links participants of `participant_kinds` (default `DEFAULT_PARTICIPANT_KINDS = [PARTICIPANT_KIND_CONNECTOR, PARTICIPANT_KIND_SIP, PARTICIPANT_KIND_STANDARD]`) and links "the first participant" unless `participant_identity` is given — [room_io/types.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/voice/room_io/types.py) (lines 122-131), [job.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/job.py) (lines 169-173)

### Inferences
- Receiving risk updates (sync callback, parse JSON, guard `participant is None`):
  ```python
  @ctx.room.on("data_received")
  def _on_data(pkt: rtc.DataPacket):
      if pkt.topic != "risk":
          return
      update = json.loads(
          pkt.data
      )  # sender identity should be inside the payload; pkt.participant is None for hidden senders
      session.userdata.risk = update
  ```
- Eric's detector token should carry `VideoGrants(room_join=True, room=<room>, hidden=True, can_subscribe=True, can_publish=False, can_publish_data=True)`; because hidden participants are invisible to RoomIO, the detector can never be mistaken for the caller, and `ctx.wait_for_participant()` will not return it. If Eric decides not to use `hidden`, pin the caller with `room_io.RoomOptions(participant_identity=participant.identity)` (and for phone-only, `participant_kinds=[rtc.ParticipantKind.PARTICIPANT_KIND_SIP]`).
- Reliable packets are capped at 15 KiB: a 60-second caller turn with ~150 words of `{"w","start_ms","end_ms","conf"}` is roughly 8–10 KB of JSON, so use compact keys and, if a turn can exceed the cap, split it or use `send_text(json, topic="turn")` (text streams have no 15 KiB packet cap) — inferred from the size limits above.
- Prefer broadcasting on a topic over `destination_identities=[detector]`: delivery to a hidden *destination* identity was not verified, broadcast delivery was.

### Gaps
- Whether `VideoGrants.can_publish_data` defaults to true when omitted was not checked (set it explicitly).
- The Python SDK may add `participant_identity` to `DataPacket` in a newer `livekit` (the JS SDK already passes `participantIdentity`); 1.1.21 (pinned by agents 1.8.10) does not have it.

## Q7. Participant info: caller identity, SIP kind and attributes, phone vs browser, hidden participants in `remote_participants`

### Takeaway
A phone caller is `participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP` with `sip.*` attributes (`sip.phoneNumber`, `sip.trunkPhoneNumber`,
`sip.callID`, `sip.callStatus`, ...); a browser caller is `PARTICIPANT_KIND_STANDARD` with none of them. Hidden participants never appear in
`room.remote_participants` (the server does not announce them), so the detector will not show up there.

### Cited Findings
- SIP participant attributes: `sip.callID` (LiveKit's SIP call ID), `sip.callIDFull` (provider's globally unique SIP call ID), `sip.callStatus` (`active`, `automation`, `dialing`, `hangup`, `ringing`), `sip.phoneNumber` (caller's number for inbound; absent if `HidePhoneNumber` is set in the dispatch rule), `sip.ruleID`, `sip.trunkID`, `sip.trunkPhoneNumber`, `sip.twilio.accountSid` / `sip.twilio.callSid` (Twilio trunks only); custom attributes via the dispatch rule or trunk `headers_to_attributes` (header-based ones arrive asynchronously); Python check (verbatim):
  ```python
  if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
      if participant.attributes["sip.phoneNumber"] == "+15105550100":
          logger.info("Caller phone number is +1-510-555-0100")
  ```
  — [SIP participant](https://docs.livekit.io/sip/sip-participant.md)
- `ctx.wait_for_participant(*, identity: str | None = None, kind: list[ParticipantKind] | ParticipantKind = DEFAULT_PARTICIPANT_KINDS) -> rtc.RemoteParticipant` ("If identity is None, the first participant that joins the room will be returned. If the participant has already joined, the function will return immediately."); `DEFAULT_PARTICIPANT_KINDS = [PARTICIPANT_KIND_CONNECTOR, PARTICIPANT_KIND_SIP, PARTICIPANT_KIND_STANDARD]` — [job.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/job.py) (lines 169-173, 650-660)
- `session.room_io.linked_participant` identifies the linked participant; `participant.attributes.get("user.language")`; `@ctx.room.on("participant_attributes_changed") def ...(changed_attrs: dict[str, str], p: rtc.Participant)`; `ctx.add_participant_entrypoint(entrypoint_fnc=..., kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP)` — [Agent sessions](https://docs.livekit.io/agents/logic/sessions.md), [Job lifecycle](https://docs.livekit.io/agents/server/job.md)
- Dispatch rule for the LiveKit number (verbatim JSON, create with `lk sip dispatch create dispatch-rule.json`):
  ```json
  {"dispatch_rule": {"rule": {"dispatch_rule_individual": {"room_prefix": "call-"}},
    "name": "My dispatch rule",
    "room_config": {"agents": [{"agent_name": "inbound-agent", "metadata": "job dispatch metadata"}]}}}
  ```
  Individual rules create one room per caller named `<prefix><phone number><random suffix>` (phone number in the room name is PII; `no_randomness` drops the suffix); `metadata` arrives as `ctx.job.metadata`; `hide_phone_number` controls visibility of the caller number; `attributes` are inherited by every SIP participant the rule creates — [Dispatch rule](https://docs.livekit.io/telephony/accepting-calls/dispatch-rule.md)
- LiveKit phone numbers: US only, inbound only; rent via dashboard (Telephony -> Phone Numbers -> Rent a number) or `lk number search --country-code US --area-code 415` then `lk number purchase --numbers +14155550100`; every number must be assigned to a dispatch rule (`lk number update --id <PHONE_NUMBER_ID> --sip-dispatch-rule-id <DISPATCH_RULE_ID>`); "doesn't require separate SIP trunk configuration"; every plan includes 1 free US local number, Build plan includes 50 free inbound minutes; `TransferSIPParticipant` not supported on these numbers — [Phone numbers](https://docs.livekit.io/telephony/start/phone-numbers.md)
- Hidden participants are excluded from the join response participant list and their state updates are not broadcast (server), and the Python SDK's `remote_participants` is populated only from the join response and `participant_connected` events — [room.go](https://github.com/livekit/livekit/blob/master/pkg/rtc/room.go), [room.py](https://github.com/livekit/python-sdks/blob/main/livekit-rtc/livekit/rtc/room.py)

### Inferences
- Channel detection for `CallState`: `is_phone = participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP`; `caller_number = participant.attributes.get("sip.phoneNumber")`; `dialed = participant.attributes.get("sip.trunkPhoneNumber")`; `sip_call_id = participant.attributes.get("sip.callID")`. Browser callers: use `participant.identity`/`name` from the token instead.
- Because the room name for individual dispatch rules embeds the caller's number, prefer `sip.callID` (or your own UUID in `job.metadata`) as the call identifier you share with the detector.

### Gaps
- The exact identity string LiveKit assigns to inbound SIP participants (commonly reported as `sip_<E.164>`) is not stated on the pages fetched; read `participant.identity` at runtime rather than parsing it.
- `sip.callStatus` transitions for inbound calls (whether it is already `active` when the agent joins) were not documented on the fetched page.

## Q8. Browser testing: fastest path to talk to the agent from a browser

### Takeaway
The hosted `agents-playground.livekit.io` has been retired in favor of the LiveKit Cloud **Agent Console** (`lk agent dev` prints its link); for a
real frontend, enable the project's **Development token server** and run `agent-starter-react` (`pnpm dev`), passing `agentName="callback"` so the
explicit-dispatch agent is requested. `python agent.py console` is the fastest *local* loop but has no room, so Eric's detector cannot join it.

### Cited Findings
- `lk agent dev` "runs the agent in development mode. It prints an Agent Console link." — [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md)
- Agent Console: "a web-based tool that allows you to debug your agents in realtime", works with agents running on LiveKit Cloud, self-hosted, "or running locally on your machine"; open via "Launch Console" on an agent or at `https://cloud.livekit.io/projects/p_/agents/console`; shows events, tool execution, model timing; observers join as hidden participants and "can't publish audio or video, but can send text and execute RPC calls"; needs Python SDK >= 1.5.2; DTMF pane exists but "Simulating a SIP participant ... is an upcoming feature" — [Agent Console](https://docs.livekit.io/testing/agent-console.md)
- Development token server: "LiveKit Sandbox is no longer available" but the token server remains; enable the **Development token server** toggle at `https://cloud.livekit.io/projects/p_/settings/project`, copy the Token server ID, use `TokenSource.developmentTokenServer("<id>")` in JS/React, then `tokenSource.fetch({ roomName, agentName: "my-agent" })` (or `useSession` with `agentName`); it accepts `agent_name` plus optional `deployment` in token requests; "for development and testing only" — [Development token server](https://docs.livekit.io/frontends/build/authentication/development-token-server.md)
- React starter (verbatim):
  ```shell
  git clone https://github.com/livekit-examples/agent-starter-react.git
  cd agent-starter-react
  pnpm install
  cp .env.example .env.local
  pnpm dev
  ```
  then open http://localhost:3000; "Token server support for quick development" is listed as a feature — [React starter app](https://docs.livekit.io/frontends/start/starter-apps/react.md)
- `console` mode runs the agent "locally without connecting to LiveKit" with mic/text input; `connect --room my-test-room` joins a specific room — [Startup modes](https://docs.livekit.io/agents/server/startup-modes.md)
- Community threads report the hosted agents playground was replaced by the Agent Console / gated behind Cloud sign-in; the open-source repo `livekit/agents-playground` remains for local runs — [community thread 1127](https://community.livekit.io/t/removing-of-https-agents-playground-livekit-io/1127), [community thread 1133](https://community.livekit.io/t/missing-old-livekit-playground-ui/1133), [livekit/agents-playground](https://github.com/livekit/agents-playground)
- Explicit dispatch from a CLI without a frontend: `lk dispatch create --agent-name test-agent --room my-room --metadata '{...}'` creates the room and dispatches the agent — [Agent dispatch](https://docs.livekit.io/agents/server/agent-dispatch.md)

### Inferences
- Day-1 loop for Veer alone: `uv run agent.py console` (no network to LiveKit needed except model APIs). Day-1 loop with Eric's detector in the room: `lk agent dev` -> open the printed Agent Console link -> talk; Eric's detector joins the same room name with a hidden token. For the demo UI: dev token server + `agent-starter-react` with `agentName="callback"`.
- For the Nov 19 live final, run the phone path (LiveKit number -> dispatch rule -> `agent_name: callback`) and the browser path (token with `RoomAgentDispatch(agent_name="callback")`) against the *same* deployed agent; both paths use explicit dispatch so one `agent_name` serves both.

### Gaps
- The Agent Console page does not explicitly say the person who launches it can publish microphone audio (only that *observers* cannot); the quickstart's "prints an Agent Console link" implies a talk-to-your-agent flow, but I could not confirm the mic UX from docs.
- `agent-starter-react`'s `.env.example` variable names (and whether it needs the token server ID vs API key/secret) were not on the docs page; check the repo README.

## Q9. Multi-agent / handoff pattern for "enrollment mode" vs normal assistant mode

### Takeaway
Model enrollment as a second `Agent` subclass: a tool on the receptionist returns `EnrollmentAgent(chat_ctx=self.chat_ctx), "Switching to voice enrollment"`;
`EnrollmentAgent.on_enter` speaks the consent script with `say(..., allow_interruptions=False)`, then drives the 7 phrases from
`on_user_turn_completed` (raising `StopResponse()` so the LLM never replies mid-enrollment), and finally `session.update_agent(Receptionist(...))`.
Shared per-call state lives in `AgentSession[CallState](userdata=...)` and is reachable as `ctx.userdata` in tools and `self.session.userdata` in agents.

### Cited Findings
- Handoffs (docs): a tool returns the new agent (`return BillingAgent(chat_ctx=self.chat_ctx), "Transferring to billing"`) or `None` to stay; `session.update_agent(CustomerServiceAgent())` switches directly; `session.current_agent` reads the active one; by default "A new agent starts with a fresh history and sees only its own instructions"; pass `chat_ctx=self.chat_ctx` for full history, `self.chat_ctx.copy(exclude_instructions=True)` for turns only, `chat_ctx.copy().truncate(max_items=6)` to keep the last turns; `session.history` always holds the complete conversation; per-agent model overrides via constructor (`tts=...`) and `turn_handling={"endpointing": {"min_delay": 1.2}}`; `on_enter` example `await self.session.generate_reply(instructions="Greet the user and ask how you can help them.")`; shared state:
  ```python
  @dataclass
  class MySessionInfo:
      user_name: str | None = None
      age: int | None = None

  session = AgentSession[MySessionInfo](userdata=MySessionInfo(), ...)
  # in tools: context: RunContext[MySessionInfo]; context.userdata.user_name = name
  # in agents: self.session.userdata
  ```
  — [Agents & handoffs](https://docs.livekit.io/agents/logic/agents-handoffs.md)
- Workflows overview: "Agents hold long-lived control of a session", "Tools can also trigger agent handoffs", context handling is "Explicit: pass `chat_ctx`, summarize, or start fresh", task groups share context and return a summarized result; linked pages: agents-handoffs, tasks (`https://docs.livekit.io/agents/logic/tasks.md`) — [Workflows](https://docs.livekit.io/agents/logic/workflows.md)
- `on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage)`; "raise StopResponse to stop the agent from generating a reply"; imports `from livekit.agents import ChatContext, ChatMessage` — [Pipeline nodes](https://docs.livekit.io/agents/build/nodes.md); `class StopResponse(Exception)` lives in `livekit.agents.llm.tool_context` — [tool_context.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/llm/tool_context.py) (line 160)
- `Agent.__init__` accepts per-agent `stt, vad, llm, tts, turn_handling, tool_handling, tools, chat_ctx, id`; hooks `on_enter`, `on_exit`, `on_user_turn_completed`; `AgentTask` is exported from `livekit.agents` and `livekit.agents.beta.workflows` provides e.g. `GetEmailTask` — [agent.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/voice/agent.py), [Async tools](https://docs.livekit.io/agents/logic/tools/async.md)
- `session.update_agent(agent: Agent) -> None` and `session.current_agent` exist in 1.8.10 — [agent_session.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/voice/agent_session.py) (lines 849, 1788)

### Inferences
- Enrollment sketch (verified names only):
  ```python
  from livekit.agents import Agent, ChatContext, function_tool, RunContext
  from livekit.agents.llm import ChatMessage, StopResponse

  PHRASES = ["My voice is my password", ...]  # 7 prompted phrases


  class EnrollmentAgent(Agent):
      def __init__(self, chat_ctx=None):
          super().__init__(
              instructions="You are running voice enrollment. Do not chat.",
              chat_ctx=chat_ctx,
              tools=[],  # no Ramp tools visible here
              turn_handling=TurnHandlingOptions(turn_detection="stt"),
          )
          self.idx = 0

      async def on_enter(self):
          await self.session.say(
              "To enroll your voice, I need your consent. ... Say yes to continue.",
              allow_interruptions=False,
          )

      async def on_user_turn_completed(
          self, turn_ctx: ChatContext, new_message: ChatMessage
      ):
          st = self.session.userdata
          text = (new_message.text_content or "").lower()
          if not st.consented:
              if "yes" not in text:
                  await self.session.say(
                      "No problem. Returning to your call.", allow_interruptions=False
                  )
                  self.session.update_agent(Receptionist(chat_ctx=self.chat_ctx))
                  raise StopResponse()
              st.consented = True
          else:
              st.enrollment_samples.append(
                  st.pending_caller_turn
              )  # text + word ms from stt_node
              self.idx += 1
          if self.idx < len(PHRASES):
              await self.session.say(
                  f"Please repeat: {PHRASES[self.idx]}", allow_interruptions=False
              )
          else:
              await self.session.say(
                  "Enrollment complete. Returning to your call.",
                  allow_interruptions=False,
              )
              self.session.update_agent(Receptionist(chat_ctx=self.chat_ctx))
          raise StopResponse()  # never let the LLM answer during enrollment


  class Receptionist(Agent):
      @function_tool()
      async def start_voice_enrollment(self, ctx: RunContext[CallState]):
          """Begin voice enrollment when the caller asks to enroll or set up voice verification."""
          return EnrollmentAgent(chat_ctx=self.chat_ctx), "Starting voice enrollment"
  ```
  Enrollment turns should be flagged `"mode": "enrollment"` in the packets you send to the detector so Eric's service treats them as reference samples.
- Each handoff restarts the STT stream (new `stt_node`), so the timestamp anchor logic from Q3 must live in a base class shared by both agents.

### Gaps
- `AgentTask` / `livekit.agents.beta.workflows` (a task that returns a result to the calling agent, e.g. for the consent question) was not researched beyond confirming it exists — see https://docs.livekit.io/agents/logic/tasks.md.
- Whether `say()` inside `on_user_turn_completed` followed by `StopResponse()` leaves the session in `listening` state cleanly was not tested; the alternative is to do the enrollment prompts from `on_enter`/a background task driven by `session.wait_for_idle()`.

## Q10. Gotchas: Apple Silicon wheels, cold start, preemptive_generation, metrics, per-call userdata, deprecations

### Takeaway
Apple Silicon is fine as long as the interpreter is arm64 (all deps had wheels; `onnxruntime 1.31.1`, `av 19.0.2`). LiveKit Cloud's free Build plan
scales production agents to zero and cold starts cost 10–20 s, so keep `lk agent dev` running or use a paid plan for the live final. Turn-taking,
interruption and preemptive-generation knobs all moved into `TurnHandlingOptions` (old kwargs deprecated until 2.0). Use `ChatMessage.metrics`
and `session.usage` for per-turn/per-call numbers; `metrics_collected` still works but is marked deprecated. Per-call state = `AgentSession[T](userdata=...)`.

### Cited Findings
- Cold start: on the Build (free) plan "production agents can scale to zero replicas after all active sessions end"; a cold start "adds 10 to 20 seconds before the agent joins the room"; paid plans (Ship, Scale) keep production agents warm; non-production deployments always scale to zero when idle; new instances must pass a health check within 5 minutes, so "make sure the `prewarm` function doesn't take longer than 5 minutes"; `lk agent config` / `lk agent deploy` / `lk agent rollback` (rollback needs a paid plan); `livekit.toml` has `[project] subdomain` and `[agent] id`; rolling deploys give old instances up to 1 hour to finish sessions — [Managing deployments](https://docs.livekit.io/deploy/agents/managing-deployments.md)
- `TurnHandlingOptions` (source TypedDict): `turn_detection: TurnDetectionMode | None`, `endpointing: EndpointingOptions(mode "fixed"|"dynamic", min_delay, max_delay, alpha)`, `interruption: InterruptionOptions(enabled, mode "adaptive"|"vad", discard_audio_if_uninterruptible, min_duration, min_words, resume_false_interruption, false_interruption_timeout, backchannel_boundary)`, `preemptive_generation: PreemptiveGenerationOptions(enabled, preemptive_tts, max_speech_duration, max_retries)`, `user_turn_limit: UserTurnLimitOptions(max_words, max_duration)` — [turn.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/voice/turn.py) (lines 113-285)
- Defaults (reference): endpointing `min_delay=0.5` s, `max_delay=3.0` s (0.3/2.5 with the audio turn detector), `alpha=0.9`; `preemptive_generation.enabled=True`, `preemptive_tts=False`, `max_speech_duration=10.0` s, `max_retries=3`; `user_turn_limit` disabled by default; example `TurnHandlingOptions(turn_detection=inference.TurnDetector(), endpointing={"mode": "fixed", "min_delay": 0.5, "max_delay": 3.0}, interruption={"min_duration": 0.5, "resume_false_interruption": True}, preemptive_generation={"preemptive_tts": True})` — [Turn handling options reference](https://docs.livekit.io/reference/agents/turn-handling-options.md)
- Sessions doc: `preemptive_generation` is "set via `turn_handling`. Default is enabled. It can raise LLM token usage."; `user_away_timeout` default 15.0 s; `transcription_timeout` needs a VAD; `use_tts_aligned_transcript` default off — [Agent sessions](https://docs.livekit.io/agents/logic/sessions.md)
- 1.8.10 release notes (29 Oct): "Silero is no longer loaded automatically." (listed under behavior changes together with Mistral default changes), streaming STT now "detects stalled connections and caps recovery retries", worker health endpoint returns 503 before registration, Deepgram "Drop the language hint the model cannot take, and add it for nova-3 multi", OpenAI TTS sample-rate option; 1.8.9: Deepgram "adds a speed parameter for Flux TTS", Cartesia "adds WebSocket timestamp parameters", adaptive interruptions fall back to VAD; 1.8.7: Deepgram "adds a configurable Flux audio chunk duration and a `language_hint` option for Flux", fix for "tools still running when interrupted speech times out"; 1.8.6: preemptive-generation fixes — [release 1.8.10](https://github.com/livekit/agents/releases/tag/livekit-agents@1.8.10), [releases](https://github.com/livekit/agents/releases)
- Metrics: `MetricsCollectedEvent(metrics: AgentMetrics, created_at)` and `SessionUsageUpdatedEvent(usage: AgentSessionUsage, created_at)` exist; `metrics.UsageCollector().collect(metrics)` / `.get_summary()` and `metrics.log_metrics(metrics)` exist; `EOUMetrics` has `end_of_utterance_delay, transcription_delay, on_user_turn_completed_delay`; `LLMMetrics.ttft`, `TTSMetrics.ttfb`; `session.usage` returns `AgentSessionUsage(model_usage: list[ModelUsage])` — [events.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/voice/events.py), [metrics/base.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/metrics/base.py), [metrics/usage_collector.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/metrics/usage_collector.py); docs label session-level `metrics_collected` deprecated — [Events reference](https://docs.livekit.io/reference/agents/events.md)
- Userdata: `AgentSession` is `Generic[Userdata_T]` with `userdata` constructor kwarg and a `userdata` property (get/set); `RunContext.userdata` returns `self.session.userdata` — [agent_session.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/voice/agent_session.py) (lines 392, 431, 793-800), [events.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/voice/events.py) (lines 93-95)
- `RoomOptions` (unified since 1.3.1 per docs) fields: `text_input, audio_input: AudioInputOptions(sample_rate=24000, num_channels=1, frame_size_ms=50, noise_cancellation, auto_gain_control, pre_connect_audio=True, pre_connect_audio_timeout=3.0), video_input, audio_output: AudioOutputOptions(sample_rate=24000, track_publish_options, track_name), text_output: TextOutputOptions(sync_transcription, transcription_speed_factor=1.0, json_format=False), participant_kinds, participant_identity, close_on_disconnect=True, delete_room_on_close=False`; legacy `RoomInputOptions`/`RoomOutputOptions` still exist — [room_io/types.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/voice/room_io/types.py) (lines 53-285), [Agent sessions](https://docs.livekit.io/agents/logic/sessions.md)
- Deprecated `AgentSession` kwargs and their replacements (`target_version="v2.0"`): `turn_detection`, `preemptive_generation`, `min/max_endpointing_delay`, `allow_interruptions`, `min_interruption_duration/words`, `false_interruption_timeout`, `resume_false_interruption`, `discard_audio_if_uninterruptible`, `agent_false_interruption_timeout` -> `turn_handling=TurnHandlingOptions(...)`; `keyterms_options` -> `stt_context_options=STTContextOptions(...)` — [agent_session.py](https://github.com/livekit/agents/blob/livekit-agents@1.8.10/livekit-agents/livekit/agents/voice/agent_session.py) (lines 393-408)
- Local test: arm64 install resolved `onnxruntime==1.31.1`, `av==19.0.2`, `numpy==2.5.4`, `livekit==1.1.21` with no source builds (`Installed 78 packages in 283ms` after resolution) — local test log

### Inferences
- Preemptive generation + Flux: `eager_eot_threshold` makes the plugin emit `PREFLIGHT_TRANSCRIPT`, which is what preemptive generation consumes ("stable enough for preemptive generation" per the `SpeechEventType` docstring). It lowers latency but costs tokens; if you want zero chance of a gate tool firing on a not-yet-final turn, start with `turn_handling=TurnHandlingOptions(turn_detection="stt", preemptive_generation={"enabled": False})` and leave `eager_eot_threshold` unset, then enable both once the gate is proven.
- "Silero is no longer loaded automatically" most plausibly refers to the `livekit-plugins-silero` plugin no longer being auto-imported; the *bundled* VAD (`inference.VAD(model="silero")` from `livekit-local-inference`) is still created when you omit `vad=`. Passing `vad=ctx.proc.userdata["vad"]` from a `setup_fnc` prewarm sidesteps the ambiguity.
- For the live final on Nov 19, do not rely on a Build-plan production deployment answering within the first ring; either upgrade for that day or keep a `lk agent dev` worker alive (dev mode registers with the same `agent_name`).
- Per-call userdata should be built from `ctx.job.metadata` (dispatch metadata), `ctx.room.name`, and the caller's `sip.*` attributes, and should hold the latest risk update plus the pending caller-turn payload from `stt_node`.

### Gaps
- Dispatch arbitration when a local `dev` worker and a Cloud deployment register the same `agent_name` simultaneously is not documented on the pages fetched; run only one during demos.
- The precise meaning of "Silero is no longer loaded automatically" in 1.8.10 is not spelled out in the release notes.
- I did not measure real cold-start or first-response latency; the 10–20 s figure is LiveKit's own statement.
