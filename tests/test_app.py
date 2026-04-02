from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_home_page_loads() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Ops Demo Launchpad" in response.text


def test_apps_api_returns_twelve_apps() -> None:
    response = client.get("/api/apps")
    assert response.status_code == 200
    assert len(response.json()["apps"]) == 12


def test_enabled_redirect_returns_target() -> None:
    response = client.get("/go/release-risk-copilot", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "https://release-risk-copilot-3f95ebdcfcc5.herokuapp.com/"


def test_healthcheck() -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"ok": True}
