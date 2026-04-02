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

## Weekly Security Audit

This repo includes a scheduled security audit workflow:

- Workflow: `.github/workflows/weekly-security-audit.yml`
- Script: `scripts/weekly_security_audit.sh`
- Schedule: every Monday at 13:00 UTC (`cron: 0 13 * * 1`)
- Also runnable manually via `workflow_dispatch`

The audit does three checks:

- scans every public `marbatis/*` repo with `gitleaks`
- reads GitHub secret-scanning open alerts per public repo
- audits Heroku app config vars for suspicious secret-like values

Recommended GitHub secrets for this repo:

- `AUDIT_GH_TOKEN`: PAT with enough read scope to query secret-scanning alerts across your repos
- `HEROKU_API_KEY`: API key for reading app config-vars through the Heroku API

Local run:

```bash
chmod +x scripts/weekly_security_audit.sh
scripts/weekly_security_audit.sh
```

Outputs are written to `reports/security-audit/`.

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
