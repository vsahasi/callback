"""Entry point. Run it from agent/: `uv run agent.py dev` (or `start`)."""

from pathlib import Path

from dotenv import load_dotenv
from livekit import agents

# The worker reads its settings when it is imported, so the repo-root .env has to be loaded first.
# Variables already set in the shell win over the file.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from callback_agent.worker import server  # noqa: E402

if __name__ == "__main__":
    agents.cli.run_app(server)
