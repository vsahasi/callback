"""Modal deployment for the detector, mock mode, CPU only.

NOT DEPLOYED. Day-1 deliberately stops at the file. To deploy it yourself:

    modal secret create callback-detector \
        CALLBACK_API_TOKEN=<long-random-string>

    modal deploy detector/modal_app.py

Secrets this needs (Modal secret name "callback-detector"):
    CALLBACK_API_TOKEN   the bearer token the LiveKit agent presents

Optional overrides, same secret or env:
    CALLBACK_MOCK_MODE                 default "true" here
    CALLBACK_OTP_MAX_ATTEMPTS          default 3
    CALLBACK_OTP_TTL_SECONDS           default 120
    CALLBACK_SESSION_TTL_SECONDS       default 300
    CALLBACK_MOCK_JITTER_MIN_MS        default 150
    CALLBACK_MOCK_JITTER_MAX_MS        default 400

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

Run it locally instead, no Modal account needed:

    uv run uvicorn detector.app:app --reload --port 8000
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
        "pydantic>=2.9",
        "uvicorn[standard]>=0.32",
    )
    .add_local_python_source("shared", "detector")
)

app = modal.App("callback-detector")


@app.function(
    image=image,
    secrets=[
        # required_keys turns a secret that exists but lacks the token into a
        # clear deploy-time error, rather than a container that boots healthy
        # and then rejects every request against a generated token.
        modal.Secret.from_name(
            "callback-detector", required_keys=["CALLBACK_API_TOKEN"]
        )
    ],
    min_containers=1,
    max_containers=1,
    timeout=60 * 10,
)
@modal.asgi_app()
def fastapi_app():
    """Serve the detector in mock mode."""

    import os

    os.environ.setdefault("CALLBACK_MOCK_MODE", "true")

    from detector.app import create_app
    from detector.config import Settings

    return create_app(Settings.from_env())
