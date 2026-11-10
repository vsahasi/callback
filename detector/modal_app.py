"""Modal deployment for the detector, mock mode, CPU only.

NOT DEPLOYED. Day-1 deliberately stops at the file. To deploy it yourself:

    modal secret create callback-detector \
        DETECTOR_TOKEN=<long-random-string> \
        USERS_ON_FILE_JSON='{"users": [...]}' \
        TWILIO_ACCOUNT_SID=... TWILIO_AUTH_TOKEN=... TWILIO_VERIFY_SERVICE_SID=...

    modal deploy detector/modal_app.py

Secrets this needs (Modal secret name "callback-detector"):
    DETECTOR_TOKEN              the bearer token the LiveKit agent presents.
                                The old name CALLBACK_API_TOKEN still works as
                                a fallback; if both are set they must match,
                                or the app refuses to start.

What the /v1 agent contract reads from the same secret:
    USERS_ON_FILE_JSON          the users_on_file directory, inline JSON (see
                                users_on_file.example.json). Holds the real
                                phone number, so it lives only here, never in
                                git. Without it the directory is empty and
                                every session create is a 404.
    TWILIO_ACCOUNT_SID          Twilio Verify, which sends and checks the SMS
    TWILIO_AUTH_TOKEN           code. Real mode answers 503 otp_unavailable
    TWILIO_VERIFY_SERVICE_SID   without all three. Mock mode ignores them.
    CALLBACK_MOCK_SMS           optional. "twilio" makes mock mode send and
                                check real codes through Twilio Verify, for
                                the recording. Blank means the mock code.
    CALLBACK_BANK_VERIFY_ENABLED  optional. "1" mounts POST
                                /verify/bank-details. Off by default.

Optional overrides, same secret or env:
    CALLBACK_MOCK_MODE                 default "true" here
    CALLBACK_OTP_MAX_ATTEMPTS          default 3
    CALLBACK_OTP_TTL_SECONDS           default 120
    CALLBACK_SESSION_TTL_SECONDS       default 900: the whole call, caller
                                       leg and callback leg, under one id
    CALLBACK_MOCK_JITTER_MIN_MS        default 150
    CALLBACK_MOCK_JITTER_MAX_MS        default 400

Real mode needs two more, and both of them:
    CALLBACK_MOCK_MODE=0
    CALLBACK_AUDIO_ENABLED=1
    CALLBACK_AUDIO_ALLOWED_HOSTS       comma separated, no default

For voice_match to run at all, vendors have to be enrolled, and for those
enrolments to outlive a deploy:
    CALLBACK_ENROLMENT_DICT_NAME       e.g. callback-voiceprints

Turning real mode on without naming hosts gives a detector that denies every
call. That is the safe direction rather than a bug: `uri` arrives in a request
body, so an unrestricted fetcher is a server-side request forgery primitive
inside the VPC. See the module docstring of detector/audio.py. A deployment
that forgot the allowlist says so in the reason on every decision.

Pinned to a single container (min_containers=1, max_containers=1) because the
default SessionStore keeps sessions in process memory, so the same call routed
to two containers would become two unrelated sessions, each with its own OTP
attempt budget.

detector/store.py now has the seam to lift that pin: ModalDictSessionStore
shares sessions across replicas. Before raising max_containers, read its
docstring on the residual race. modal.Dict has no compare-and-swap, so a lease
stolen at the wrong moment can lose one update. A monotonic guard keeps the
cost to one extra OTP attempt recorded, never a fabricated allow, but it is a
real trade and should be a deliberate decision rather than a config tweak.

The enrolment store has the same seam and none of that difficulty, because a
voice print is written whole and never built up across requests. Set
CALLBACK_ENROLMENT_DICT_NAME and reference voices survive a deploy and are
shared between replicas. Leave it unset and every enrolment is lost on
restart, which means voice_match silently stops running for every vendor until
they are enrolled again: that is a fail-closed direction, but a quiet one, so
set the name before anyone relies on it.

Run it locally instead, no Modal account needed:

    uv run uvicorn detector.app:app --reload --no-access-log --port 8000
"""

from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi>=0.115",
        "httpx>=0.27",
        # detector/analysis needs this on the serving path.
        "numpy>=1.26",
        # callback_shared needs 2.11 or later for its schema snapshot.
        "pydantic>=2.11,<3",
        # detector/audio decodes flac with this. The wheel carries
        # libsndfile, so the image needs no apt package for it.
        "soundfile>=0.12",
        "uvicorn[standard]>=0.32",
    )
    # callback_shared is the agent contract the /v1 router serves. It lives
    # in shared/src, installed editable, so it is added by module name.
    .add_local_python_source("shared", "detector", "callback_shared")
)

app = modal.App("callback-detector")


@app.function(
    image=image,
    secrets=[
        # required_keys turns a secret that exists but lacks the token into a
        # clear deploy-time error, rather than a container that boots healthy
        # and then rejects every request against a generated token.
        modal.Secret.from_name("callback-detector", required_keys=["DETECTOR_TOKEN"])
    ],
    min_containers=1,
    max_containers=1,
    timeout=60 * 10,
)
# One container, but many requests at once: without this Modal hands the
# container one input at a time, so turn posts, challenge polls, authorize and
# the review dashboard would queue behind each other, and the 2.2 s authorize
# deadline does not count time in that queue. Safe on one container: all /v1
# state is touched on one event loop.
@modal.concurrent(max_inputs=32)
@modal.asgi_app()
def fastapi_app():
    """Serve the detector in mock mode."""

    import os

    os.environ.setdefault("CALLBACK_MOCK_MODE", "true")

    from detector.app import create_app
    from detector.config import Settings

    return create_app(Settings.from_env())
