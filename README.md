# Ops Demo Launchpad

Single-entry Heroku launchpad for the 12 demo apps.

## What this solves

You wanted one public Heroku URL that acts as the front door for the full app set. This project gives you:

- a landing page with all 12 apps
- one redirect route per app at `/go/<slug>`
- a simple JSON registry with the live Heroku URLs
- a Heroku-ready setup with `Procfile` and `app.json`

## Current audit baseline

Audit date: `2026-04-01`

- All 12 GitHub repos include `Procfile`, `app.json`, and a web UI entrypoint.
- All 12 target apps now have live Heroku web URLs in [`config/apps.json`](/Users/marcelosilveira/Documents/Playground/config/apps.json).
- `false-green-dashboard-simulator` maps to the Heroku app `false-green-sim-marbatis`.

## Local run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Configuration

Edit [`config/apps.json`](/Users/marcelosilveira/Documents/Playground/config/apps.json) for each card:

- `target_url`: where the real deployed app lives
- `enabled`: controls whether `/go/<slug>` redirects to the app

The launchpad redirects from `/go/<slug>` to the configured target.

## Heroku deploy

```bash
heroku create ops-demo-launchpad
git push heroku main
```

## Recommended rollout

1. Deploy each of the 12 apps and confirm the public URL.
2. Update [`config/apps.json`](/Users/marcelosilveira/Documents/Playground/config/apps.json) with the confirmed URL.
3. Flip `enabled` to `true` for that app.
4. Deploy this launchpad app to Heroku.
