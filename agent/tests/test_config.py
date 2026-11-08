import pytest
from callback_shared.contract import Scenario

from callback_agent.config import Settings

# Everything the worker needs to start: the default `required` of Settings.from_env.
REQUIRED_ENV = {
    "CARTESIA_VOICE_ID": "v",
    "DETECTOR_URL": "http://detector.test",
    "DETECTOR_TOKEN": "token",
    "RAMP_CLIENT_ID": "client-id",
    "RAMP_CLIENT_SECRET": "client-secret",
}


def test_the_worker_needs_the_voice_the_detector_and_ramp_by_default():
    with pytest.raises(ValueError) as error:
        Settings.from_env({})
    assert str(error.value).startswith(
        "Missing required environment variables: CARTESIA_VOICE_ID, DETECTOR_URL, DETECTOR_TOKEN, "
        "RAMP_CLIENT_ID, RAMP_CLIENT_SECRET"
    )


@pytest.mark.parametrize("name", list(REQUIRED_ENV))
def test_each_default_required_name_is_enforced(name):
    env = {**REQUIRED_ENV, name: " "}  # blank counts as missing
    with pytest.raises(ValueError, match=name):
        Settings.from_env(env)


def test_a_full_environment_builds_the_settings():
    s = Settings.from_env(REQUIRED_ENV)
    assert (s.cartesia_voice_id, s.detector_url, s.detector_token) == (
        "v",
        "http://detector.test",
        "token",
    )
    assert (s.ramp_client_id, s.ramp_client_secret) == ("client-id", "client-secret")


def test_missing_required_names_are_all_reported():
    with pytest.raises(ValueError, match="CARTESIA_VOICE_ID.*DETECTOR_URL"):
        Settings.from_env({}, required=("CARTESIA_VOICE_ID", "DETECTOR_URL"))


def test_scenario_and_defaults():
    s = Settings.from_env({**REQUIRED_ENV, "DETECTOR_SCENARIO": "replay"})
    assert (s.agent_name, s.detector_scenario, s.ramp_base_url) == (
        "callback",
        Scenario.REPLAY,
        "https://demo-api.ramp.com",
    )


def test_empty_values_count_as_missing():
    # A .env copied from .env.example holds CALLBACK_AGENT_NAME= and RAMP_BASE_URL=, which load as "".
    s = Settings.from_env({**REQUIRED_ENV, "CALLBACK_AGENT_NAME": "", "RAMP_BASE_URL": " "})
    assert (s.agent_name, s.ramp_base_url) == ("callback", "https://demo-api.ramp.com")
    with pytest.raises(ValueError, match="CARTESIA_VOICE_ID"):
        Settings.from_env({**REQUIRED_ENV, "CARTESIA_VOICE_ID": ""})


def test_every_setting_reads_its_own_variable():
    env = {
        "CALLBACK_AGENT_NAME": "callback-cloud",
        "CARTESIA_VOICE_ID": "voice",
        "DETECTOR_URL": "http://detector.test",
        "DETECTOR_TOKEN": "token",
        "DETECTOR_SCENARIO": "ai_agent",
        "RAMP_BASE_URL": "https://ramp.test",
        "RAMP_CLIENT_ID": "client-id",
        "RAMP_CLIENT_SECRET": "client-secret",
        "LIVEKIT_SIP_OUTBOUND_TRUNK_ID": "trunk-id",
        "TELNYX_NUMBER": "telnyx-number",
    }
    assert Settings.from_env(env) == Settings(
        agent_name="callback-cloud",
        cartesia_voice_id="voice",
        detector_url="http://detector.test",
        detector_token="token",
        detector_scenario=Scenario.AI_AGENT,
        ramp_base_url="https://ramp.test",
        ramp_client_id="client-id",
        ramp_client_secret="client-secret",
        sip_outbound_trunk_id="trunk-id",
        telnyx_number="telnyx-number",
    )


def test_unset_optional_values_are_none():
    s = Settings.from_env(REQUIRED_ENV)
    assert (s.detector_scenario, s.sip_outbound_trunk_id, s.telnyx_number) == (None, None, None)


def test_a_script_can_ask_for_less():
    # scripts/seed_ramp.py needs the Ramp credentials and nothing else
    s = Settings.from_env({"RAMP_CLIENT_ID": "id", "RAMP_CLIENT_SECRET": "secret"}, required=())
    assert (s.ramp_client_id, s.detector_url, s.cartesia_voice_id) == ("id", None, "")


def test_unknown_scenario_names_the_variable_and_the_choices():
    with pytest.raises(ValueError, match=r"DETECTOR_SCENARIO.*replay"):
        Settings.from_env({**REQUIRED_ENV, "DETECTOR_SCENARIO": "repaly"})
