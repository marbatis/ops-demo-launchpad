#!/usr/bin/env bash
# Install the Android Agent Kit into an Android project.
#
# Usage: install.sh [--force] [TARGET_DIR]
#        install.sh --user [--force]
#   TARGET_DIR  project root (default: current directory)
#   --user      install for Claude Code on this computer (~/.claude): subagents
#               and the android-ai-dev skill, active in every project
#   --force     overwrite kit-owned files that already exist
#
# Never overwrites a project's own docs/journeys.md, and only appends an
# @AGENTS.md import to an existing CLAUDE.md.
set -euo pipefail

KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORCE=0
USER_MODE=0
TARGET="."

for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    --user) USER_MODE=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) TARGET="$arg" ;;
  esac
done

copy() {
  local src="$1" dst="$2"
  mkdir -p "$(dirname "$dst")"
  if [[ -e "$dst" && $FORCE -eq 0 ]]; then
    echo "  skip   ${dst#$TARGET/} (exists; use --force to update)"
  else
    cp "$src" "$dst"
    echo "  write  ${dst#$TARGET/}"
  fi
}

if [[ $USER_MODE -eq 1 ]]; then
  TARGET="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
  echo "Installing Android Agent Kit for Claude Code into $TARGET"
  for agent in "$KIT_DIR"/templates/.claude/agents/*.md; do
    copy "$agent" "$TARGET/agents/$(basename "$agent")"
  done
  copy "$KIT_DIR/skills/android-ai-dev/SKILL.md" "$TARGET/skills/android-ai-dev/SKILL.md"
  for doc in "$KIT_DIR"/docs/*.md; do
    copy "$doc" "$TARGET/skills/android-ai-dev/docs/$(basename "$doc")"
  done
  echo "Done. Restart Claude Code; the android-ai-dev skill and subagents are now available everywhere."
  exit 0
fi

TARGET="$(cd "$TARGET" && pwd)"
echo "Installing Android Agent Kit into $TARGET"

copy "$KIT_DIR/AGENTS.md" "$TARGET/AGENTS.md"

for doc in "$KIT_DIR"/docs/*.md; do
  copy "$doc" "$TARGET/docs/agent/$(basename "$doc")"
done

for agent in "$KIT_DIR"/templates/.claude/agents/*.md; do
  copy "$agent" "$TARGET/.claude/agents/$(basename "$agent")"
done

# journeys.md is project-specific: seed it once, never overwrite.
if [[ -e "$TARGET/docs/journeys.md" ]]; then
  echo "  keep   docs/journeys.md"
else
  copy "$KIT_DIR/templates/journeys.md" "$TARGET/docs/journeys.md"
fi

# Claude Code reads CLAUDE.md; import AGENTS.md so there is one source of truth.
if [[ ! -e "$TARGET/CLAUDE.md" ]]; then
  printf '@AGENTS.md\n' > "$TARGET/CLAUDE.md"
  echo "  write  CLAUDE.md"
elif ! grep -qx '@AGENTS.md' "$TARGET/CLAUDE.md"; then
  printf '\n@AGENTS.md\n' >> "$TARGET/CLAUDE.md"
  echo "  append CLAUDE.md (@AGENTS.md import)"
else
  echo "  keep   CLAUDE.md"
fi

echo "Done. Next: describe your app's journeys in docs/journeys.md."
