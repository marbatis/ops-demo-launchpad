#!/usr/bin/env bash
set -euo pipefail

GH_OWNER="${GH_OWNER:-marbatis}"
WORKDIR="${WORKDIR:-/tmp/public-security-audit}"
REPORT_ROOT="${REPORT_ROOT:-$(pwd)/reports/security-audit}"
TIMESTAMP="$(date -u +"%Y-%m-%dT%H-%M-%SZ")"
RUN_DIR="$REPORT_ROOT/$TIMESTAMP"

mkdir -p "$WORKDIR/repos" "$RUN_DIR"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing_required_command: $1" >&2
    exit 2
  fi
}

require_cmd gh
require_cmd git
require_cmd jq
require_cmd curl
require_cmd gitleaks

if [[ -n "${AUDIT_GH_TOKEN:-}" ]]; then
  export GH_TOKEN="$AUDIT_GH_TOKEN"
fi
if [[ -z "${GH_TOKEN:-}" && -n "${GITHUB_TOKEN_FALLBACK:-}" ]]; then
  export GH_TOKEN="$GITHUB_TOKEN_FALLBACK"
fi
if [[ -z "${GH_TOKEN:-}" ]]; then
  token="$(gh auth token 2>/dev/null || true)"
  if [[ -n "$token" ]]; then
    export GH_TOKEN="$token"
  fi
fi

if [[ -z "${HEROKU_API_KEY:-}" ]] && command -v heroku >/dev/null 2>&1; then
  heroku_token="$(heroku auth:token 2>/dev/null || true)"
  if [[ -n "$heroku_token" ]]; then
    export HEROKU_API_KEY="$heroku_token"
  fi
fi

repos_file="$RUN_DIR/public_repos.txt"
gh api --paginate "/users/$GH_OWNER/repos?type=public&per_page=100" | jq -r ".[].name" | sort > "$repos_file"

gitleaks_csv="$RUN_DIR/github_gitleaks_summary.csv"
echo "repo,findings_count,state,note" > "$gitleaks_csv"

while IFS= read -r repo; do
  [[ -z "$repo" ]] && continue
  repo_dir="$WORKDIR/repos/$repo"
  report_json="$RUN_DIR/gitleaks_${repo}.json"

  if [[ ! -d "$repo_dir/.git" ]]; then
    if ! git clone "https://github.com/$GH_OWNER/$repo.git" "$repo_dir" >/dev/null 2>&1; then
      echo "$repo,0,clone_error,clone_failed" >> "$gitleaks_csv"
      continue
    fi
  else
    git -C "$repo_dir" fetch --all --prune >/dev/null 2>&1 || true
  fi

  rm -f "$report_json"
  set +e
  gitleaks detect --no-banner --source "$repo_dir" --report-format json --report-path "$report_json" >/dev/null 2>&1
  code=$?
  set -e

  findings_count=0
  if [[ -f "$report_json" ]]; then
    findings_count="$(jq "length" "$report_json" 2>/dev/null || echo 0)"
  fi

  if [[ "$code" -eq 0 ]]; then
    echo "$repo,0,clean,-" >> "$gitleaks_csv"
  elif [[ "$code" -eq 1 && "$findings_count" -gt 0 ]]; then
    echo "$repo,$findings_count,findings,-" >> "$gitleaks_csv"
  else
    echo "$repo,$findings_count,scan_error,exit_code_$code" >> "$gitleaks_csv"
  fi
done < "$repos_file"

secret_scan_csv="$RUN_DIR/github_secret_scanning_summary.csv"
echo "repo,open_alerts,state,note" > "$secret_scan_csv"

while IFS= read -r repo; do
  [[ -z "$repo" ]] && continue
  set +e
  response="$(gh api "/repos/$GH_OWNER/$repo/secret-scanning/alerts?state=open&per_page=100" 2>/tmp/gh_secret_scan_err.$$)"
  code=$?
  set -e

  if [[ "$code" -ne 0 ]]; then
    note="$(tr '\n' ' ' < /tmp/gh_secret_scan_err.$$ | sed 's/,/;/g' | sed 's/[[:space:]]\\+/ /g')"
    echo "$repo,-1,api_error,${note:-unknown_error}" >> "$secret_scan_csv"
  else
    count="$(printf "%s" "$response" | jq "length")"
    if [[ "$count" -gt 0 ]]; then
      echo "$repo,$count,alerts,-" >> "$secret_scan_csv"
    else
      echo "$repo,0,clean,-" >> "$secret_scan_csv"
    fi
  fi
done < "$repos_file"
rm -f /tmp/gh_secret_scan_err.$$

heroku_csv="$RUN_DIR/heroku_config_summary.csv"
echo "app,key_count,sensitive_key_names,suspicious_value_key_names,state,note" > "$heroku_csv"

if [[ -n "${HEROKU_API_KEY:-}" ]]; then
  apps_json="$(curl -sS "https://api.heroku.com/apps" \
    -H "Accept: application/vnd.heroku+json; version=3" \
    -H "Authorization: Bearer $HEROKU_API_KEY")"

  printf "%s" "$apps_json" | jq -r ".[].name" | sort > "$RUN_DIR/heroku_apps.txt"

  while IFS= read -r app; do
    [[ -z "$app" ]] && continue
    config_json="$(curl -sS "https://api.heroku.com/apps/$app/config-vars" \
      -H "Accept: application/vnd.heroku+json; version=3" \
      -H "Authorization: Bearer $HEROKU_API_KEY")"

    key_count="$(printf "%s" "$config_json" | jq "keys | length")"
    sensitive_keys="$(
      printf "%s" "$config_json" | jq -r '
        to_entries[]
        | select(.key | test("(?i)(KEY|TOKEN|SECRET|PASSWORD|PRIVATE|CREDENTIAL|DATABASE_URL|CONNECTION|URI|WEBHOOK|AUTH)"))
        | .key
      ' | paste -sd';' - || true
    )"
    suspicious_value_keys="$(
      printf "%s" "$config_json" | jq -r '
        to_entries[]
        | select((.value | tostring) | test("(sk-[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]+PRIVATE KEY-----|Bearer [A-Za-z0-9._-]{20,})"))
        | .key
      ' | paste -sd';' - || true
    )"

    [[ -z "$sensitive_keys" ]] && sensitive_keys="-"
    [[ -z "$suspicious_value_keys" ]] && suspicious_value_keys="-"

    state="clean"
    note="-"
    if [[ "$suspicious_value_keys" != "-" ]]; then
      state="suspicious_value"
      note="review_required"
    fi

    echo "$app,$key_count,$sensitive_keys,$suspicious_value_keys,$state,$note" >> "$heroku_csv"
  done < "$RUN_DIR/heroku_apps.txt"
else
  echo "no_heroku_api_key,0,-,-,skipped,HEROKU_API_KEY_unavailable" >> "$heroku_csv"
fi

gitleaks_findings_repos="$(awk -F',' 'NR>1 && $3=="findings" {count++} END {print count+0}' "$gitleaks_csv")"
gitleaks_total_findings="$(awk -F',' 'NR>1 && $3=="findings" {sum+=$2} END {print sum+0}' "$gitleaks_csv")"
secret_alert_repos="$(awk -F',' 'NR>1 && $3=="alerts" {count++} END {print count+0}' "$secret_scan_csv")"
secret_alert_total="$(awk -F',' 'NR>1 && $3=="alerts" {sum+=$2} END {print sum+0}' "$secret_scan_csv")"
secret_scan_api_errors="$(awk -F',' 'NR>1 && $3=="api_error" {count++} END {print count+0}' "$secret_scan_csv")"
heroku_suspicious_apps="$(awk -F',' 'NR>1 && $5=="suspicious_value" {count++} END {print count+0}' "$heroku_csv")"

report_md="$RUN_DIR/report.md"
{
  echo "# Weekly Security Audit"
  echo
  echo "- Timestamp (UTC): $TIMESTAMP"
  echo "- GitHub owner: \`$GH_OWNER\`"
  echo "- Public repos scanned: \`$(wc -l < "$repos_file" | tr -d " ")\`"
  echo "- Heroku apps scanned: \`$(awk -F',' 'NR>1 {count++} END {print count+0}' "$heroku_csv")\`"
  echo
  echo "## Headline Result"
  echo
  echo "- Gitleaks findings: \`$gitleaks_total_findings\` across \`$gitleaks_findings_repos\` repos"
  echo "- GitHub secret-scanning open alerts: \`$secret_alert_total\` across \`$secret_alert_repos\` repos"
  echo "- GitHub secret-scanning API errors: \`$secret_scan_api_errors\` repos"
  echo "- Heroku suspicious config-value matches: \`$heroku_suspicious_apps\` apps"
  echo
  echo "## Artifacts"
  echo
  echo "- \`github_gitleaks_summary.csv\`"
  echo "- \`github_secret_scanning_summary.csv\`"
  echo "- \`heroku_config_summary.csv\`"
  echo
  if [[ "$gitleaks_findings_repos" -gt 0 ]]; then
    echo "## Repos With Gitleaks Findings"
    echo
    awk -F',' 'NR>1 && $3=="findings" {printf "- %s (%s findings)\n", $1, $2}' "$gitleaks_csv"
    echo
  fi
  if [[ "$secret_alert_repos" -gt 0 ]]; then
    echo "## Repos With Open GitHub Secret Alerts"
    echo
    awk -F',' 'NR>1 && $3=="alerts" {printf "- %s (%s open alerts)\n", $1, $2}' "$secret_scan_csv"
    echo
  fi
  if [[ "$secret_scan_api_errors" -gt 0 ]]; then
    echo "## Repos With GitHub Secret API Errors"
    echo
    awk -F',' 'NR>1 && $3=="api_error" {printf "- %s (%s)\n", $1, $4}' "$secret_scan_csv"
    echo
  fi
  if [[ "$heroku_suspicious_apps" -gt 0 ]]; then
    echo "## Heroku Apps With Suspicious Value Matches"
    echo
    awk -F',' 'NR>1 && $5=="suspicious_value" {printf "- %s (keys: %s)\n", $1, $4}' "$heroku_csv"
    echo
  fi
} > "$report_md"

cp "$report_md" "$REPORT_ROOT/latest.md"
cp "$gitleaks_csv" "$REPORT_ROOT/latest_gitleaks_summary.csv"
cp "$secret_scan_csv" "$REPORT_ROOT/latest_secret_scanning_summary.csv"
cp "$heroku_csv" "$REPORT_ROOT/latest_heroku_summary.csv"

echo "report_dir=$RUN_DIR"
echo "latest_report=$REPORT_ROOT/latest.md"

if [[ "$gitleaks_total_findings" -gt 0 || "$secret_alert_total" -gt 0 || "$heroku_suspicious_apps" -gt 0 ]]; then
  exit 1
fi

