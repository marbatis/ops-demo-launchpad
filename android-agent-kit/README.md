# Android Agent Kit

A set of documents to feed to AI coding agents (Claude Code, Codex, Antigravity/Gemini, Qwen, Kimi, or any
ACP-compatible agent in Android Studio) when building **new Android apps with AI features**.

Source material: the **Android Dev Summit 2026 keynote** (28 Oct 2026, https://www.youtube.com/live/8Lmv7LGgL4E),
its full transcript, and Google's follow-up developer blog posts. Timestamps in these docs refer to that video.

## What's in here

| File | Purpose | Feed it when |
|---|---|---|
| [`AGENTS.md`](AGENTS.md) | Drop-in root instructions for any coding agent. Short, imperative rules. | Always. Copy to the root of every Android project (also as `CLAUDE.md`). |
| [`docs/00-adsummit-2026-reference.md`](docs/00-adsummit-2026-reference.md) | Everything announced, with status (stable/alpha/canary/preview/coming) and timestamps. | When the agent needs to know what exists and how mature it is. |
| [`docs/01-architecture-standards.md`](docs/01-architecture-standards.md) | Tech stack, module layout, Compose-first, state, DI, data. | Project setup and any architectural change. |
| [`docs/02-agentic-dev-workflow.md`](docs/02-agentic-dev-workflow.md) | How the agent should build, verify, profile and fix: Android CLI, skills, preview, prototype, journeys, leak loop, AI review. | Every coding session. |
| [`docs/03-ai-features-playbook.md`](docs/03-ai-features-playbook.md) | AppFunctions / Android MCP, Gemini Nano 4 Agentic API, A2UI, Connected Apps, cloud fallback. | Building any AI or agent feature into the app. |
| [`docs/04-adaptive-desktop-xr.md`](docs/04-adaptive-desktop-xr.md) | Foldables, Google Book (desktop), Android XR / XREAL Aura, Play badges. | UI work, layouts, new form factors. |
| [`docs/05-quality-and-release-checklist.md`](docs/05-quality-and-release-checklist.md) | Definition of done, security, accessibility, performance, Play readiness. | Before every PR and every release. |
| [`docs/06-migration-xml-to-compose.md`](docs/06-migration-xml-to-compose.md) | Converting legacy View/XML code. | Working on older codebases. |
| [`templates/journeys.md`](templates/journeys.md) | Template of user journeys the agent runs for testing/leak detection. | Copy to `docs/journeys.md` in each project. |
| [`templates/.claude/agents/android-release-validator.md`](templates/.claude/agents/android-release-validator.md) | Claude Code subagent that audits release readiness (as demoed in the keynote). | Copy to `.claude/agents/` in each project. |
| [`templates/.claude/agents/android-leak-hunter.md`](templates/.claude/agents/android-leak-hunter.md) | Subagent that runs journeys, finds and fixes memory leaks. | Copy to `.claude/agents/`. |
| [`templates/.claude/agents/android-adaptive-reviewer.md`](templates/.claude/agents/android-adaptive-reviewer.md) | Subagent that makes/validates adaptive + desktop layouts. | Copy to `.claude/agents/`. |

## How to get it into every project

There are four ways, and they combine:

### 1. New app: start from a template repo
Keep a GitHub template repo (e.g. `marbatis/android-ai-starter`) that was created by running
`install.sh` on an empty repo and ticking **Settings → Template repository**. New apps: "Use this template",
then create the Android project in that folder with Android Studio (Empty Activity, Compose).
Without a template repo, run `install.sh` on the new project folder; the result is the same.

### 2. Claude Code plugin: available in every project, every session
```
/plugin install android-ai-kit --marketplace marbatis/ops-demo-launchpad
```
This gives Claude Code, in any project:
- the `android-ai-dev` skill, which loads this playbook automatically when doing Android work
- the `android-release-validator`, `android-leak-hunter` and `android-adaptive-reviewer` subagents
- `/android-ai-kit:setup-android-project`, which installs the kit files into the current project

Update later with `/plugin marketplace update marbatis-tools`.

### 3. This computer, without the plugin marketplace
```bash
bash path/to/ops-demo-launchpad/android-agent-kit/install.sh --user
```
Installs the subagents and the `android-ai-dev` skill (with the docs) into `~/.claude`, so local Claude Code
sessions have them in every project. Re-run with `--force` after pulling kit updates.

### 4. Any existing project, any agent: the installer
```bash
bash path/to/ops-demo-launchpad/android-agent-kit/install.sh path/to/my-android-app
```
It writes `AGENTS.md`, `docs/agent/*.md`, `.claude/agents/*.md`, seeds `docs/journeys.md`, and adds an
`@AGENTS.md` import to `CLAUDE.md`. It skips files that already exist; `--force` updates kit-owned files but
never overwrites your `docs/journeys.md`.

`AGENTS.md` is read by Codex, Antigravity and agents in Android Studio; Claude Code reads it via the
`CLAUDE.md` import. It references the `docs/agent/*.md` files by path, so agents load the deep docs on demand
rather than all at once (saves tokens).

For tools that take a single context file (e.g. a chat UI), paste `AGENTS.md` first, then the specific doc for
the task.

## Accuracy notes — read before trusting any API name

- Many features are **alpha, Canary, preview, or "coming soon"** as of Oct 2026. Each doc marks status.
- Code labelled **"pattern"** reflects what was shown on stage; exact class/method names may differ from the
  shipped library. Agents are instructed to verify names against developer.android.com or the library sources
  before writing production code.
- Auto-captions in the transcript mangle names: "Cloud" = Claude, "Kelly" = CLI, "A tui" = A2UI,
  "Quinn/Kimmy" = Qwen/Kimi.
- Re-check status of each item every few months and update `docs/00-adsummit-2026-reference.md`.
