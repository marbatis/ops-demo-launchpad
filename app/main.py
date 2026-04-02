import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "apps.json"

app = FastAPI(title="Ops Webapp Hub", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))


def _config_path() -> Path:
    raw_value = os.getenv("APP_CONFIG_PATH")
    return Path(raw_value) if raw_value else DEFAULT_CONFIG_PATH


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    return json.loads(_config_path().read_text())


def load_apps() -> list[dict[str, Any]]:
    apps: list[dict[str, Any]] = []
    for item in load_config()["apps"]:
        record = dict(item)
        record["launch_path"] = f"/go/{record['slug']}"
        record["is_enabled"] = bool(record.get("enabled") and record.get("target_url"))
        apps.append(record)
    return apps


def get_app_config(slug: str) -> dict[str, Any]:
    for item in load_apps():
        if item["slug"] == slug:
            return item
    raise HTTPException(status_code=404, detail=f"Unknown app: {slug}")


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    apps = load_apps()
    configured_count = sum(1 for item in apps if item.get("target_url"))
    enabled_count = sum(1 for item in apps if item["is_enabled"])
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "apps": apps,
            "title": load_config()["title"],
            "subtitle": load_config()["subtitle"],
            "audit_date": load_config()["audit_date"],
            "configured_count": configured_count,
            "enabled_count": enabled_count,
            "total_count": len(apps),
        },
    )


@app.get("/api/apps")
def apps_api() -> dict[str, Any]:
    return {"apps": load_apps()}


@app.get("/go/{slug}")
def go(slug: str) -> RedirectResponse:
    item = get_app_config(slug)
    if not item["is_enabled"]:
        raise HTTPException(status_code=404, detail=f"{item['name']} is not enabled yet.")
    return RedirectResponse(item["target_url"], status_code=307)


@app.get("/healthz")
def healthz() -> dict[str, bool]:
    return {"ok": True}
