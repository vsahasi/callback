"""How the session takes turns and fills its silences: the options worker.py builds it with."""

from livekit.agents import AgentSession, BackgroundAudioPlayer, BuiltinAudioClip

from callback_agent.session_wiring import THINKING_SOUND, TURN_HANDLING


async def test_livekit_accepts_the_turn_handling():
    AgentSession(turn_handling=TURN_HANDLING)  # never started: this only checks the options


def test_flux_still_decides_turns_and_nothing_is_generated_early():
    # A tool must never fire on a half-finished turn.
    assert TURN_HANDLING["turn_detection"] == "stt"
    assert TURN_HANDLING["preemptive_generation"] == {"enabled": False}


def test_a_turn_is_committed_soon_after_flux_ends_it():
    # The fixed default waits 0.5 s on top of Flux's own end of turn: silence on every turn.
    assert TURN_HANDLING["endpointing"]["min_delay"] <= 0.3


def test_a_lone_word_or_line_noise_does_not_stop_the_agent():
    assert TURN_HANDLING["interruption"]["min_words"] >= 2


def test_a_false_interruption_pauses_the_agent_for_at_most_a_second():
    assert TURN_HANDLING["interruption"]["false_interruption_timeout"] <= 1.0


async def test_hold_music_plays_quietly_while_the_agent_thinks():
    [sound] = THINKING_SOUND
    assert sound.source is BuiltinAudioClip.HOLD_MUSIC
    assert 0 < sound.volume <= 0.4  # under the agent's voice and the caller's
    player = BackgroundAudioPlayer(thinking_sound=THINKING_SOUND)  # never started
    await player.aclose()
