"""/v1 paths collapse to route templates in metrics and logs.

Session, challenge, enrollment and user ids never become metric labels or
logged paths, and the name query on GET /v1/users never reaches a log line.
"""

from __future__ import annotations

import io
import json

import httpx
import pytest

from detector.observability import (
    BASE_ENDPOINTS,
    UNMATCHED_ENDPOINT,
    V1_ENDPOINTS,
    normalise_endpoint,
)

KNOWN = BASE_ENDPOINTS | V1_ENDPOINTS


@pytest.mark.parametrize(
    ("path", "template"),
    [
        ("/v1/users", "/v1/users"),
        ("/v1/users/u_eric", "/v1/users/{user_id}"),
        ("/v1/sessions", "/v1/sessions"),
        ("/v1/sessions/ses_0123/caller", "/v1/sessions/{session_id}/caller"),
        ("/v1/sessions/ses_0123/turns", "/v1/sessions/{session_id}/turns"),
        ("/v1/sessions/ses_0123/events", "/v1/sessions/{session_id}/events"),
        ("/v1/sessions/ses_0123/challenges", "/v1/sessions/{session_id}/challenges"),
        (
            "/v1/sessions/ses_0123/challenges/chl_ab",
            "/v1/sessions/{session_id}/challenges/{challenge_id}",
        ),
        ("/v1/sessions/ses_0123/otp/send", "/v1/sessions/{session_id}/otp/send"),
        ("/v1/sessions/ses_0123/otp/check", "/v1/sessions/{session_id}/otp/check"),
        ("/v1/sessions/ses_0123/authorize", "/v1/sessions/{session_id}/authorize"),
        ("/v1/sessions/ses_0123/factors", "/v1/sessions/{session_id}/factors"),
        ("/v1/sessions/ses_0123/end", "/v1/sessions/{session_id}/end"),
        ("/v1/enrollments", "/v1/enrollments"),
        ("/v1/enrollments/enr_9/complete", "/v1/enrollments/{enrollment_id}/complete"),
        ("/v1/enrollments/u_eric", "/v1/enrollments/{user_id}"),
        ("/v1/sessions/ses_0123/", UNMATCHED_ENDPOINT),
        ("/v1/sessions/ses_0123/nope", UNMATCHED_ENDPOINT),
        ("/v1/a/b/c", UNMATCHED_ENDPOINT),
    ],
)
def test_ids_collapse_to_templates(path: str, template: str) -> None:
    assert normalise_endpoint(path, KNOWN) == template


def test_v1_templates_need_registering() -> None:
    assert normalise_endpoint("/v1/users/u_eric") == UNMATCHED_ENDPOINT


def _requests(stream: io.StringIO) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in stream.getvalue().splitlines()
        if '"http_request"' in line
    ]


async def test_logged_path_is_the_template(
    v1: httpx.AsyncClient, log_stream: io.StringIO
) -> None:
    await v1.get("/v1/users/u_eric")
    await v1.get("/v1/users", params={"name": "Eric Catalano"})
    records = _requests(log_stream)
    assert [r["endpoint"] for r in records] == ["/v1/users/{user_id}", "/v1/users"]
    assert [r["path"] for r in records] == ["/v1/users/{user_id}", "/v1/users"]
    text = log_stream.getvalue()
    assert "u_eric" not in text
    assert "Eric" not in text
    assert "Catalano" not in text


async def test_metric_labels_are_templates(v1: httpx.AsyncClient) -> None:
    await v1.get("/v1/users/u_eric")
    response = await v1.get("/metrics")
    assert response.status_code == 200
    assert 'endpoint="/v1/users/{user_id}"' in response.text
    assert "u_eric" not in response.text
