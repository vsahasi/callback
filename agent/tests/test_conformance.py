"""The five demo calls of the final contract ruling (section 5), against a running detector.

Skipped unless DETECTOR_URL and DETECTOR_TOKEN are set. Each test drives the real Gate, StepUps,
Enrollment and DetectorClient through one call (see conformance_support.py), and the same cases
are proved against a stub in test_conformance_stub.py. The detector must be in mock mode, with
users_on_file.example.json:

    # terminal 1, repo root
    DETECTOR_TOKEN=dev-local CALLBACK_MOCK_MODE=1 USERS_ON_FILE_PATH=users_on_file.example.json \\
      uv run uvicorn detector.app:app --port 8000 --no-access-log
    # terminal 2
    cd agent && DETECTOR_URL=http://127.0.0.1:8000 DETECTOR_TOKEN=dev-local \\
      uv run pytest -m mock_detector tests/test_conformance.py tests/test_detector.py -v

Calls 2 and the browser variants wait 6.5 s after create for the cloned voice to give itself away.
"""

import os

import pytest
from conformance_support import (
    case_ai_agent,
    case_cloned_voice,
    case_detector_stopped,
    case_enrollment,
    case_human_impostor,
    case_missing_version_header,
    case_real_cfo,
    case_replay,
    case_wrong_token,
)

URL = os.environ.get("DETECTOR_URL", "").strip()
TOKEN = os.environ.get("DETECTOR_TOKEN", "").strip()

pytestmark = [
    pytest.mark.mock_detector,
    pytest.mark.skipif(
        not (URL and TOKEN),
        reason="DETECTOR_URL and DETECTOR_TOKEN are not set: this needs a running detector in "
        "mock mode",
    ),
]


async def test_call_1_real_cfo_steps_up_then_allows():
    await case_real_cfo(URL, TOKEN)


async def test_call_2_cloned_voice_is_denied_and_the_vendor_held():
    await case_cloned_voice(URL, TOKEN)


async def test_call_3_replay_is_denied_and_the_vendor_held():
    await case_replay(URL, TOKEN)


async def test_call_4_ai_agent_is_denied_and_the_vendor_held():
    await case_ai_agent(URL, TOKEN)


async def test_call_5_human_impostor_is_denied_and_the_vendor_held():
    await case_human_impostor(URL, TOKEN)


async def test_enrollment_completes_with_seven_prompts():
    await case_enrollment(URL, TOKEN)


async def test_a_wrong_token_is_a_deny_that_holds_the_vendor():
    await case_wrong_token(URL, TOKEN)


async def test_a_missing_version_header_is_a_400():
    await case_missing_version_header(URL, TOKEN)


async def test_a_stopped_detector_is_a_deny_that_holds_the_vendor():
    await case_detector_stopped(TOKEN)
