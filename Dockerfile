# The cloud backup of the Callback worker, built by `lk agent create` (see docs/demo-runbook.md).
# Build from the repo root, not from agent/. agent/ installs ../shared as a path dependency, so
# both folders have to be in the build context:
#
#   docker build -t callback-agent .
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Log lines must not sit in a buffer if the worker dies. Compile the bytecode now, so the first
# call after a cold start does not pay for it.
ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1

# The worker runs as a normal user, not root.
RUN adduser --disabled-password --gecos "" --home /app --shell /sbin/nologin \
    --no-create-home --uid 10001 appuser

# Keep the two folders side by side, like in the repo: agent/uv.lock points at ../shared.
WORKDIR /app
COPY shared/ shared/
COPY agent/ agent/
RUN chown -R appuser:appuser /app

USER appuser
WORKDIR /app/agent

# Every locked package has a Linux wheel, so there is no compiler in this image. If a new
# dependency has none, install gcc, g++ and python3-dev first.
RUN uv sync --frozen

# Run the module, not `agent.py download-files`: agent.py reads its settings when it is imported,
# and the build has no environment.
RUN uv run -m livekit.agents download-files

# LiveKit Cloud supplies LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET. The rest comes from
# the secrets given to `lk agent create`, including CALLBACK_AGENT_NAME=callback-cloud.
CMD ["uv", "run", "agent.py", "start"]
