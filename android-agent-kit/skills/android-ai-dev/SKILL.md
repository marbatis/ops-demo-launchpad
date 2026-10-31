---
name: android-ai-dev
description: Android app development playbook (Android Dev Summit 2026). Use whenever building, changing, reviewing, testing or releasing a native Android app — Kotlin, Jetpack Compose, Gradle Android modules, AndroidManifest.xml — or adding AI/agent features (Gemini Nano, ML Kit GenAI, AppFunctions, Android MCP, A2UI, Connected Apps), adaptive/foldable/desktop (Google Book) or Android XR support.
---

# Android AI development playbook

Follow these rules for any Android work. Detailed references are in `<base>/docs/` when installed with
`install.sh --user`, otherwise in the plugin's `docs/` folder two levels up (`<base>/../../docs/`), where
`<base>` is this skill's base directory. Read only the doc the task needs.

| Task | Read |
|---|---|
| What exists and how mature it is (stable/alpha/canary/preview) | `docs/00-adsummit-2026-reference.md` |
| Project setup, stack, modules, screen pattern, leak-safe and security rules | `docs/01-architecture-standards.md` |
| How to build/verify: Android CLI, `android preview`, journeys, leak loop, AI review | `docs/02-agentic-dev-workflow.md` |
| AI features: AppFunctions, Gemini Nano 4 Agentic API, A2UI, Connected Apps | `docs/03-ai-features-playbook.md` |
| Foldable, desktop/Google Book, XR | `docs/04-adaptive-desktop-xr.md` |
| Definition of done, release to Google Play | `docs/05-quality-and-release-checklist.md` |
| XML/View code | `docs/06-migration-xml-to-compose.md` |

If the project has its own `AGENTS.md` or `docs/agent/` copy, the project's version wins.

## Non-negotiables
- Never guess Android APIs; many 2026 APIs are alpha/preview. Verify names against developer.android.com or
  library sources and say when something is unverified.
- Prefer Android CLI commands and skills over exploring logcat or guessing (`android --help` for exact names).
- Kotlin + Jetpack Compose only; no new XML layouts. Material 3, Navigation 3, adaptive from screen one.
  ViewModel + StateFlow, Hilt, Room/DataStore, coroutines. Resolve latest stable library versions.
- Every screen: previews for phone/foldable/desktop, light/dark, font scale; `testTag`s on interactive elements.
- Loop per change: implement → `android preview` → build/tests → run affected journeys from `docs/journeys.md`
  on phone, foldable, desktop → leak check if lifecycle code changed → report evidence. Never claim a check
  passed that you didn't run.
- AI features: business actions as `@AppFunction` first (serves in-app agent, Android MCP, Connected Apps);
  on-device Gemini Nano with cloud fallback; A2UI with a fixed component catalog; user confirmation for
  destructive or financial actions.

## Subagents in this plugin
- `android-release-validator` — Play policy, security, accessibility, performance, adaptive checks, smoke tests.
- `android-leak-hunter` — run journeys with leak detection, fix, re-verify all journeys.
- `android-adaptive-reviewer` — adaptive and desktop-ready layouts, keyboard/pointer, state on resize.

## Project without journeys or AGENTS.md?
Suggest running the `setup-android-project` skill to install the kit's files into the project.
