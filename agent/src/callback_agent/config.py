"""Settings read from the environment. agent.py loads the repo-root .env before anything imports this."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from callback_shared.contract import Scenario

DEFAULT_AGENT_NAME = "callback"
DEFAULT_RAMP_BASE_URL = "https://demo-api.ramp.com"

# What the worker cannot run without: a voice, the detection service, and Ramp. Without the detector
# no caller can be identified, and without Ramp there is nothing to act on.
REQUIRED_ENV = (
    "CARTESIA_VOICE_ID",
    "DETECTOR_URL",
    "DETECTOR_TOKEN",
    "RAMP_CLIENT_ID",
    "RAMP_CLIENT_SECRET",
)


@dataclass(frozen=True)
class Settings:
    agent_name: str
    cartesia_voice_id: str
    detector_url: str | None = None
    detector_token: str | None = None
    detector_scenario: Scenario | None = None
    ramp_base_url: str = DEFAULT_RAMP_BASE_URL
    ramp_client_id: str | None = None
    ramp_client_secret: str | None = None
    sip_outbound_trunk_id: str | None = None
    telnyx_number: str | None = None

    @classmethod
    def from_env(
        cls, env: Mapping[str, str], *, required: Iterable[str] = REQUIRED_ENV
    ) -> "Settings":
        """Build the settings from environment variables.

        A blank value counts as unset, because a .env copied from .env.example holds `NAME=` lines and
        python-dotenv loads those as "". `required` names environment variables (the worker's
        REQUIRED_ENV unless a script asks for fewer); every one that is unset or blank is reported
        in a single ValueError.
        """

        def get(name: str) -> str | None:
            return (env.get(name) or "").strip() or None

        missing = [name for name in required if get(name) is None]
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)} (see .env.example)"
            )

        scenario = get("DETECTOR_SCENARIO")
        try:
            detector_scenario = Scenario(scenario) if scenario else None
        except ValueError:
            choices = ", ".join(s.value for s in Scenario)
            raise ValueError(f"DETECTOR_SCENARIO must be one of {choices}, not {scenario!r}") from None

        return cls(
            agent_name=get("CALLBACK_AGENT_NAME") or DEFAULT_AGENT_NAME,
            cartesia_voice_id=get("CARTESIA_VOICE_ID") or "",
            detector_url=get("DETECTOR_URL"),
            detector_token=get("DETECTOR_TOKEN"),
            detector_scenario=detector_scenario,
            ramp_base_url=get("RAMP_BASE_URL") or DEFAULT_RAMP_BASE_URL,
            ramp_client_id=get("RAMP_CLIENT_ID"),
            ramp_client_secret=get("RAMP_CLIENT_SECRET"),
            sip_outbound_trunk_id=get("LIVEKIT_SIP_OUTBOUND_TRUNK_ID"),
            telnyx_number=get("TELNYX_NUMBER"),
        )
