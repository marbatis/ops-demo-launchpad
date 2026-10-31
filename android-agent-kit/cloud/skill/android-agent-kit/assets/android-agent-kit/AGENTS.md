# Agent Instructions — Android Project

You are working on a native Android app. Follow these rules. Deep references live in `docs/agent/`; read the
relevant one before starting a task in that area.

## 0. Ground rules
- **Never guess Android APIs.** If unsure of a class, method, artifact or version, check developer.android.com,
  the library source, or Android CLI skills/knowledge base. Many 2026 APIs are alpha/preview and their names
  change. Say so explicitly when something is unverified.
- **Prefer tools over exploration.** Use Android CLI commands and skills instead of reading raw logcat, grepping
  build outputs, or reasoning about how Android "probably" works. It is ~2x faster and ~25% cheaper in tokens.
- **Small, verifiable steps.** After every meaningful change: build → preview/run → run affected journeys.
- **Do not** disable tests, lint checks, or R8 rules to make things pass. Fix the cause.
- **Do not** add a dependency without stating why and checking it is maintained and Kotlin/Compose friendly.

## 1. Stack (non-negotiable unless the user says otherwise)
- Kotlin only. **Jetpack Compose only** for UI — XML/Views are in maintenance mode. No new XML layouts.
- Material 3, **Navigation 3**, adaptive layouts (window size classes) from the first screen.
- State: `ViewModel` + `StateFlow` (UDF). **AppState** (alpha) only behind an explicit opt-in from the user.
- DI: Hilt. Data: Room + DataStore. Network: Retrofit/Ktor + kotlinx.serialization. Async: coroutines/Flow.
- Media: Media3. Images: Coil. Background: WorkManager.
- Full details: `docs/agent/01-architecture-standards.md`.

## 2. Workflow for every task
1. Read the task, the relevant `docs/agent/*` file, and `docs/journeys.md`.
2. Plan briefly (files to touch, journeys affected).
3. Implement in Compose with previews (`@Preview` for phone, foldable, desktop sizes, light + dark).
4. Verify UI with `android preview` (renders Compose previews without emulator/IDE).
5. Run affected journeys on emulator or remote device via Android CLI. For larger changes use the lightweight
   prototype runner across phone/foldable/desktop if available.
6. If you touched lifecycle, ViewModels, callbacks, listeners, coroutine scopes, or Activity/Context references:
   run the leak check on the affected journeys (see `docs/agent/02-agentic-dev-workflow.md`).
7. Run unit tests, lint, and screenshot tests. Fix failures.
8. Summarize: what changed, which journeys were run on which devices, results, anything unverified.
Full details: `docs/agent/02-agentic-dev-workflow.md`.

## 3. AI features
- Expose every user-facing business action as an **AppFunction** first. That single implementation serves the
  in-app agent, Android MCP (system Gemini), and the Connected Apps Platform.
- On-device first (Gemini Nano 4 via ML Kit GenAI Prompt API / Agentic API); cloud fallback via Firebase AI
  Logic. Always check model availability at runtime.
- Agent-generated UI uses **A2UI** with a fixed component catalog. Never render arbitrary model output as UI.
- Destructive or financial actions require explicit user confirmation.
- Full details: `docs/agent/03-ai-features-playbook.md`.

## 4. Form factors
- Every screen must work on phone, foldable (both postures), tablet, and desktop (Google Book): keyboard,
  mouse/pointer, tooltips, resizable windows, state preserved across resize.
- Apps must run unmodified on Android XR (e.g. XREAL Aura) as spatial windows.
- Full details: `docs/agent/04-adaptive-desktop-xr.md`.

## 5. Definition of done
See `docs/agent/05-quality-and-release-checklist.md`. Minimum: builds, tests green, previews render, affected
journeys pass on phone + foldable + desktop, no new leaks, accessibility labels present, no secrets in code.

## 6. Subagents available (Claude Code)
- `android-release-validator` — Play policy, security, accessibility, smoke tests; reports blockers.
- `android-leak-hunter` — runs journeys, finds/fixes leaks, re-verifies.
- `android-adaptive-reviewer` — makes screens adaptive and desktop-ready.
