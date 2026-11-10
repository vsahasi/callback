"""close_background bounds the BackgroundAudioPlayer's close at the job's shutdown.

In livekit-agents 1.8.10 the job's shutdown disconnects the room before it runs the shutdown
callbacks, and the room is usually gone already (every end of a call deletes it). The player's
aclose() ends by unpublishing its track, and on a room that is no longer connected that wait never
returns: left as a bare shutdown callback it held the job open until the worker killed the process
at 10 s. The player here is a stand-in with the one method that matters; two tests pin the two
facts about LiveKit that the bound rests on.
"""

import asyncio
import inspect
import logging
from types import SimpleNamespace

import pytest
from livekit.agents import BackgroundAudioPlayer
from livekit.agents.ipc import job_proc_lazy_main

import callback_agent.session_wiring as wiring
from callback_agent.session_wiring import close_background

LOGGER = "callback_agent.session_wiring"


class Player:
    """Stands in for a BackgroundAudioPlayer: closes, hangs in its close, or raises from it."""

    def __init__(self, *, hang: bool = False, error: Exception | None = None) -> None:
        self._hang = hang
        self._error = error
        self.closed = False
        self.cut_off = False

    async def aclose(self) -> None:
        if self._error is not None:
            raise self._error
        if self._hang:
            try:
                await asyncio.Event().wait()  # the unpublish whose answer never comes
            except asyncio.CancelledError:
                self.cut_off = True
                raise
        self.closed = True


def room(*, connected: bool) -> SimpleNamespace:
    return SimpleNamespace(isconnected=lambda: connected)


@pytest.fixture
def quick(monkeypatch):
    monkeypatch.setattr(wiring, "BACKGROUND_CLOSE_S", 0.05)


async def close(player: Player, *, connected: bool) -> None:
    await asyncio.wait_for(close_background(player, room(connected=connected)), timeout=5)


async def test_a_player_that_closes_is_closed_and_nothing_is_logged(caplog):
    player = Player()
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        await close(player, connected=False)
    assert player.closed
    assert caplog.records == []


async def test_a_close_that_hangs_once_the_room_is_gone_is_cut_off_without_a_warning(quick, caplog):
    player = Player(hang=True)
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        await close(player, connected=False)
    assert player.cut_off
    assert all(r.levelno < logging.WARNING for r in caplog.records)
    assert any("background audio" in r.getMessage() for r in caplog.records)


async def test_a_close_that_hangs_on_a_live_room_is_cut_off_with_a_warning(quick, caplog):
    player = Player(hang=True)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        await close(player, connected=True)
    assert player.cut_off
    assert "the background audio was still closing after 0.05 s" in caplog.text


async def test_a_close_that_raises_is_logged_by_type_and_swallowed(caplog):
    player = Player(error=RuntimeError("TR_1234 could not be unpublished"))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        await close(player, connected=False)
    assert "could not close the background audio: RuntimeError" in caplog.text
    assert "TR_1234" not in caplog.text  # the type and the place, never the message


def test_the_default_bound_leaves_room_for_the_rest_of_the_shutdown():
    # The parent kills the process 10 s after the job reports that it is shutting down, and
    # finish_call runs next to this one; a second is plenty for a close that is going to finish.
    assert 0 < wiring.BACKGROUND_CLOSE_S <= 2.0


# --- what the bound rests on, in livekit-agents 1.8.10 -----------------------------------------


def test_livekit_disconnects_the_room_before_it_runs_the_shutdown_callbacks():
    # If this order ever changes, the player's own close may finish on its own again.
    source = inspect.getsource(job_proc_lazy_main._JobProc._shutdown_job)
    assert source.index("await self._room.disconnect()") < source.index("_shutdown_callbacks")


def test_the_players_close_ends_with_an_unpublish_on_the_room():
    source = inspect.getsource(BackgroundAudioPlayer.aclose)
    assert "unpublish_track" in source
