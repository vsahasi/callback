"""How the session takes turns and fills its silences: the options worker.py builds it with."""

from livekit.agents import AgentSession, BackgroundAudioPlayer, BuiltinAudioClip

from callback_agent.session_wiring import (
    BACKGROUND_STREAM_TIMEOUT_MS,
    THINKING_SOUND,
    TURN_HANDLING,
)


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


def test_interruptions_come_from_the_local_vad_not_the_cloud_model():
    # On the rehearsal calls the cloud "adaptive" detector paused the agent three times a call on
    # handset echo, and its 100 ms polls cost a network round trip each. The VAD runs here, and
    # the thresholds below are what decide, nothing else.
    assert TURN_HANDLING["interruption"]["mode"] == "vad"


def test_handset_echo_shorter_than_most_of_a_second_is_not_an_interruption():
    assert TURN_HANDLING["interruption"]["min_duration"] >= 0.8


async def test_hold_music_is_about_as_loud_as_the_agents_voice():
    # The built-in clip is -22 dBFS. At 0.25 it went out at -34 dBFS and no caller heard it over
    # a narrowband phone line. 0.8 is -24 dBFS, just under speech.
    [sound] = THINKING_SOUND
    assert sound.source is BuiltinAudioClip.HOLD_MUSIC
    assert 0.6 <= sound.volume <= 1.0
    player = BackgroundAudioPlayer(
        thinking_sound=THINKING_SOUND, stream_timeout_ms=BACKGROUND_STREAM_TIMEOUT_MS
    )  # never started
    await player.aclose()


def test_the_music_mixer_waits_for_a_slow_frame_instead_of_dropping_it():
    # The player's default gives the clip decoder 200 ms per frame. The first frame of a call on a
    # cold process took longer, and the mixer logged a timeout and put out silence for that block.
    assert BACKGROUND_STREAM_TIMEOUT_MS >= 1000
