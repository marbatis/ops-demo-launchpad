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

## How to use it in a new Android project

```bash
KIT=path/to/ops-demo-launchpad/android-agent-kit
cd my-new-android-app
cp $KIT/AGENTS.md ./AGENTS.md
cp $KIT/AGENTS.md ./CLAUDE.md            # Claude Code reads CLAUDE.md
mkdir -p docs/agent .claude/agents
cp $KIT/docs/*.md docs/agent/
cp $KIT/templates/journeys.md docs/journeys.md
cp $KIT/templates/.claude/agents/*.md .claude/agents/
```

`AGENTS.md` references the `docs/agent/*.md` files by path, so agents load the deep docs on demand rather
than all at once (saves tokens).

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
