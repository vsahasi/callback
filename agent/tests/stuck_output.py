import asyncio

from livekit.agents.voice import io


class StuckTranscription(io.TextOutput):
    """A transcript output that never finishes taking one line, the way a hung TTS never plays it.

    A roomless session plays a line out through its transcript output, and say() waits until the
    output has taken all of it. So this line never finishes playing until LiveKit cancels it.
    """

    def __init__(self, stuck_on: str) -> None:
        super().__init__(label="stuck", next_in_chain=None)
        self._stuck_on = stuck_on

    async def capture_text(self, text: str) -> None:
        if text == self._stuck_on:
            await asyncio.Event().wait()  # until the speech is cancelled

    def flush(self) -> None:
        pass
