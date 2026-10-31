# Android Dev Summit 2026 — Reference for Agents

Keynote: 28 Oct 2026, Mountain View. Video: https://www.youtube.com/live/8Lmv7LGgL4E (1:01:55).
Timestamps below are `mm:ss` into that video. Status is as stated on stage / in Google's blog at that date.

**Status legend:** `STABLE` production-ready · `CANARY` Android Studio Canary only · `ALPHA` API may change ·
`PREVIEW` program/early access · `COMING` announced, not shipped · `EARLY` sneak peek, name not final.

---

## 1. Themes (closing, 1:00:16)
1. **Choice in how you build** — any model, any agent, any environment; capabilities for every agent via Android CLI.
2. **Intelligent experiences** inside apps, and **Connected Apps** to reach users across Google surfaces.
3. **New hardware** (Google Book laptops, XREAL Aura glasses); Compose apps are ready for all of it.

## 2. AI tooling

| Item | Status | Time | Key facts |
|---|---|---|---|
| Bring Your Own Model (Android Studio) | STABLE | 2:09 | Shipped last year. |
| **Bring Your Own Agent** via **Agent Client Protocol (ACP)** | STABLE (latest stable Studio) | 2:15, 16:49 | Settings → *Agents & marketplace*; 40+ integrations (Antigravity, Codex, Claude Agent, Qwen, Kimi…). Multiple agents can be installed side by side. Full harness of each agent runs inside Studio with IDE tools and shared agent context. |
| Antigravity default in Studio | COMING (next months) | 18:04 | Gemini Enterprise subscriptions picked up automatically. |
| **Android knowledge base** | STABLE | 2:38 | Up-to-date Android best practices for agents. |
| **Android skills** | STABLE | 7:32, 8:44 | 30+ skills across 20+ areas. Built with an evals framework so they only teach what models don't already know. Targets fast-moving areas: Compose, performance, advanced camera APIs, security, adaptive. |
| **Android CLI** | STABLE | 2:54, 7:25 | Tool-first interface for agents: code analysis, device management, UI debugging, run app journeys, inspect heaps, diagnose errors. 10x usage growth in 2 months. Works with Claude Code, Codex, Pi, Antigravity, etc. |
| Remote devices (Android CLI) | STABLE | 10:17 | Run journeys on real hardware (demo: Galaxy Z Fold 8) from the CLI. |
| `android preview` | STABLE | 12:49 | Renders Compose previews as screenshots without emulator or IDE. Blog: agents 77% faster on UI tasks, up to 3x fewer tokens. |
| **LeakCanary in Android CLI** (`android profiler leakcanary`-style command) | COMING ("next few weeks") | 10:30 | Runs a journey, analyzes leak trace, follows reference chain, proposes fix. Demo: 4 leaks across 6 journeys found, fixed, re-verified. With CLI: ~2x faster, 25% fewer tokens (Opus 5.5, medium effort). |
| **Android prototype** (working name) | EARLY | 13:24 | `android prototype <apk>` runs the app in a lightweight local process (laptop RAM), interactive, real DB writes (Room). Multiple instances at once: phone, foldable, desktop. Agents can drive it to verify behaviour. |
| **Review all changes with AI** | Android Studio (new) | 15:20 | Reviews an agent's changeset, flags extra issues, produces a walkthrough grouped by issue fixed. |
| Custom subagents via ACP | STABLE | 18:28 | Demo: Claude Code subagent "Android release validator" — runs Play policy insights skill, security, accessibility checks, emulator smoke tests in parallel; reports blockers / medium / high. Same subagent works in CLI and Studio. |
| **Migrate to Compose** (right-click XML → Migrate to Compose) | CANARY | 20:30 | Agent builds a step-by-step plan, produces artifacts, outputs Compose + preview. Works with any connected agent. |
| Fix with AI for memory leaks (Studio) | CANARY | (blog) | Studio-side counterpart of the CLI leak flow. |

## 3. Architecture & libraries

| Item | Status | Time | Key facts |
|---|---|---|---|
| **Compose-first** | Policy | 19:58 | All Android UI should be Compose; Views are in **maintenance mode**. |
| Navigation 3 | STABLE | 22:34, 40:32 | Used for adaptive list-detail "scene strategy". |
| **AppState** | ALPHA | 22:47 | Compose's reactive model brought to the app/state layer; e.g. share session state across multiple UIs/components. Feedback wanted. |
| Media3 + Dawn/WebGPU | STABLE | 6:04 | Adobe Premiere: multitrack 4K editing + AI noise removal. |
| Evidence | — | 3:28–4:49 | Instagram Direct: Compose AI-native UI → >33% less token cost & agent time; UI code −50%. Shopify: React Native → Kotlin, +50% startup. |

## 4. In-app intelligence

| Item | Status | Time | Key facts |
|---|---|---|---|
| **A2UI** (Agent-to-UI) open protocol | Protocol | 24:21 | Agent describes UI as declarative JSON in its responses. |
| **A2UI Compose renderer** | ALPHA | 24:31–27:47 | Renders with *your* component catalog (Material 3 + custom domain components). A surface composable renders streamed UI; body slots swap content in place with shimmer; server pushes small **JSON patches**; properties **bind** to Compose state so only observers recompose. Native performance, input, accessibility, theming. |
| **Gemini Nano 4** | Available on supported devices | 28:07 | Native **function calling**. |
| ML Kit GenAI Prompt API (expanded) | — | 28:28 | Access to Nano 4 capabilities. |
| **Agentic API** (ML Kit GenAI) | New | 28:14, 28:33 | Auto-executes functions, saves conversation history, manages orchestration loop, automatic guardrails. Fully offline multi-turn agent (demo in airplane mode). |
| **Android MCP** powered by **AppFunctions** | New | 31:09 | Unified tool-calling framework: share app capabilities with your own agent *and* system agents (Gemini) without building twice. |

## 5. Distribution: Connected Apps Platform

| Item | Status | Time | Key facts |
|---|---|---|---|
| **Connected Apps Platform** | PREVIEW (apply for production) | 32:48 | Apps take actions for users in Search and Gemini app. 50+ apps live (e.g. Instacart). Branded interactive widgets (logo, colours, typography). Your brand, auth, business logic stay yours. |
| Sensitive action confirmation | Built-in | 33:01 | Money transfers, reservations etc. show an interactive confirmation card. |
| **Gemini dev mode** | Available today | 34:31 | Simulate agent interactions, inspect schemas, validate tool calls. |
| Remote MCP | — | 34:45 | Validate/test/deploy MCP servers; reach Gemini on web, mobile, Search. |
| Android MCP (on-device) | — | 35:03 | AppFunctions Jetpack libraries connect apps to Gemini on device. |

## 6. Form factors & Google Play

| Item | Status | Time | Key facts |
|---|---|---|---|
| Play foldable quality review | COMING (next year) | 5:25 | Play will look closely at app/game quality on foldables; more showcase opportunities for adaptive apps. Details later this year. |
| Android Auto | — | 5:02 | 250M+ compatible cars; 700 apps for cars with Google built-in. |
| **Google Book** laptops | Launched Oct 2026 | 35:51 | Built on Android stack; 5 OEMs; clamshell, convertible, detachable soon. ≥16 GB RAM, NPU 45+ TOPS. Continue On: one-click handoff phone→Google Book (app-to-app, app-to-web). Linux terminal VM with sudo; Antigravity preinstalled (beta); one-click VM reset keeping host files. |
| Large-screen economics | — | 36:17 | Phone+large-screen users spend 9x more; Notability +12x buyer conversion after Compose optimisation. |
| **Adaptive skill** (desktop support) | STABLE | 40:03–42:09 | Agent prompt "make my app adaptive and desktop ready" → Nav3 list-detail scene strategy, keyboard shortcuts, tooltips, custom pointer cursors, state preserved on resize. |
| Play **Made for / Optimized for desktop** badge | New | 42:14 | Granted when fully large-screen optimized incl. expected desktop journeys; boosts large-screen recommendations. |

## 7. Android XR

| Item | Status | Time | Key facts |
|---|---|---|---|
| Android XR platform | STABLE (existing, headsets already shipping) | 43:25 | One platform for headsets, glasses, everything in between. |
| **XREAL Aura** | Orderable now | 44:33 | First wired XR glasses on Android XR. <95 g frames; compute puck (doubles as trackpad); Snapdragon Reality Elite; optical see-through; hand + voice input; USB-C plug-and-play mirroring (laptop/console as spatial window); Gemini sees what you see. Existing phone/tablet apps work out of the box; 150+ XR experiences, 30+ new made-for-XR titles. $1,279 ($1,099 through end of 2026). US/CA/KR now, JP 2 Nov; ships mid-Nov. |
| **Jetpack XR SDK core + Android XR emulators** | STABLE | 59:12 | Production-ready, backward-compatible. Extend Compose apps into 3D; anchor panels alongside spatial elements. |
| Android XR engine hub (Unity, Unreal, Godot) | STABLE | 59:46 | For engine-based apps/games. |
