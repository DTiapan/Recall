"""Unit tests for API key authentication guard."""

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from recall.api.auth import build_api_key_guard


def _build_test_app(required_api_key: str | None) -> FastAPI:
    app = FastAPI()
    guard = build_api_key_guard(required_api_key)

    @app.post("/protected", dependencies=[Depends(guard)])
    async def protected_route() -> dict[str, str]:
        return {"status": "ok"}

    return app


def test_api_key_guard_allows_open_access_when_unconfigured():
    client = TestClient(_build_test_app(None))
    response = client.post("/protected")
    assert response.status_code == 200


def test_api_key_guard_rejects_missing_key():
    client = TestClient(_build_test_app("secret-key"))
    response = client.post("/protected")
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"


def test_api_key_guard_accepts_valid_bearer_token():
    client = TestClient(_build_test_app("secret-key"))
    response = client.post("/protected", headers={"Authorization": "Bearer secret-key"})
    assert response.status_code == 200


def test_api_key_guard_accepts_x_api_key_header():
    client = TestClient(_build_test_app("secret-key"))
    response = client.post("/protected", headers={"X-API-Key": "secret-key"})
    assert response.status_code == 200
