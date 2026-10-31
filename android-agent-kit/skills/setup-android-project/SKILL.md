---
name: setup-android-project
description: Installs the Android Agent Kit into the current Android project — AGENTS.md, CLAUDE.md import, docs/agent/ references, docs/journeys.md template and .claude/agents subagents. Use when starting a new Android app or when asked to set up/update the agent kit in a project.
disable-model-invocation: true
---

# Set up the Android Agent Kit in this project

1. Confirm the current directory is the project root (has `settings.gradle(.kts)` or is an empty new repo).
2. Run the installer from this plugin, two levels above this skill's base directory:
   `bash <base>/../../install.sh .`
   Add `--force` only if the user asked to update existing kit files.
3. Show the installer output.
4. Open `docs/journeys.md` and replace the example journeys with ones for this app, based on its screens and
   features (ask the user if the app has no code yet).
5. Remind the user to commit the new files.
