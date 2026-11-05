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

Pinned to a single container (min_containers=1, max_containers=1) because
session state lives in process memory. Do not raise max_containers before
SessionStore is backed by Redis or a Modal Dict.

Run it locally instead, no Modal account needed:

    uv run uvicorn detector.app:app --reload --port 8000
"""

from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi>=0.115",
        "pydantic>=2.9",
        "uvicorn[standard]>=0.32",
    )
    .add_local_python_source("shared", "detector")
)

app = modal.App("callback-detector")


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("callback-detector")],
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
