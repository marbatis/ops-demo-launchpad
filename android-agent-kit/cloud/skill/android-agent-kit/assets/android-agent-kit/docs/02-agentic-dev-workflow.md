# Agentic Development Workflow

How an AI coding agent should build, verify, profile, fix and hand off Android work.

Core principle from the keynote: *"Even the smartest models get dramatically more efficient when you give them
the right tools."* With Android CLI, the same model found and fixed the same leaks ~2x faster with 25% fewer
tokens, because it stopped polling logcat and guessing.

---

## 1. Tooling map

| Need | Use | Avoid |
|---|---|---|
| Android best practice | Android skills / knowledge base | Answering from memory on fast-moving APIs |
| See a UI change | `android preview` (Compose previews → screenshots, no emulator/IDE) | Building + launching emulator for every tweak |
| Interact with the app quickly | `android prototype <apk>` (EARLY; lightweight local instances, phone/foldable/desktop) | Full emulator boot when not needed |
| Run on emulator / real device | Android CLI device management, **remote devices** | Hand-written adb scripts |
| Run user journeys | Android CLI journeys + `docs/journeys.md` | Ad-hoc clicking |
| Find memory leaks | Android CLI profiler + LeakCanary command (COMING); until then LeakCanary in debug build + heap dump analysis via CLI | Watching leak notifications / logcat in a loop |
| Review the agent's diff | Android Studio **Review all changes with AI** + walkthrough | Merging large agent PRs unread |
| Release readiness | `android-release-validator` subagent | Manual one-off checks |

Command names change while features are new. Before using a command, run the CLI's help (e.g.
`android --help`, `android <command> --help`) and use what it reports.

## 2. Standard task loop

```
read task + AGENTS.md + relevant doc + docs/journeys.md
   → plan (files, affected journeys, devices)
   → implement (Compose + previews)
   → android preview  ──(visual wrong?)──┐
   → build + unit tests                  │ fix
   → run affected journeys (phone, foldable, desktop)
   → leak check if lifecycle-sensitive code changed
   → lint + screenshot tests
   → summary with evidence
```

The summary must list: files changed, journeys run × devices × result, leak results, previews checked, anything
not verified and why.

## 3. Journeys

Journeys are the contract between you and your agents: the agent can only test and fix what is described.
Keep them in `docs/journeys.md` (template in `templates/journeys.md`).

Rules for good journeys:
- One user goal per journey, numbered steps, explicit **Expect:** assertions.
- Reference elements by `testTag` or visible text.
- Include **stress steps** for leaks: open/close a screen 10–20x, rotate, fold/unfold, resize window, background
  and foreground.
- Mark which devices each journey must pass on: `phone`, `foldable`, `desktop`, `xr` (optional).
- Provide test data/credentials via a debug-only path, never real accounts.

## 4. Memory-leak loop (as demoed 9:06–11:30)

1. Ensure journeys exist and `leakcanary-android` is a `debugImplementation` dependency.
2. Prompt: *"Find and fix memory leaks in my journeys."*
3. For each journey × device: run journey with leak detection → collect leak traces.
4. For each leak: follow the reference chain to the GC root, identify the owning object that outlives its
   lifecycle (common: static/singleton holding Activity, unremoved listener, coroutine outliving scope,
   ViewModel holding Context, lambda capture).
5. Apply the smallest fix that respects the lifecycle (see architecture doc §5).
6. **Re-run every journey** (not only the leaking one) to confirm leaks are gone and behaviour is unchanged.
7. Report table: journey | device | leaks before | leaks after | fix commit.

Status: LeakCanary command in Android CLI is COMING (announced "in the next few weeks" on 28 Oct 2026). Until
available, use LeakCanary's heap analysis output pulled via the CLI/device tools, or Android Studio's Canary
"Fix with AI" for leaks.

## 5. UI iteration loop
1. Edit composable + its previews.
2. `android preview` → inspect screenshots at all preview sizes.
3. Compare against design / previous screenshots; fix spacing, overflow, contrast, truncation.
4. Only then build and run journeys.

## 6. Multi-device validation
- Phone (compact), foldable (medium, both postures), desktop/Google Book (expanded, keyboard + mouse, resizable).
- Use remote devices for real-hardware checks (e.g. foldables) before release.
- Optional: prototype runner with several sizes at once for quick behaviour checks.

## 7. Working inside Android Studio (ACP)
- Agents connect via Settings → Agents & marketplace. Multiple agents can be installed; pick per task.
- Custom subagents defined for Claude Code (`.claude/agents/*.md`) are usable both in the CLI and in Studio.
- XML → Compose: right-click layout → **Migrate to Compose** (Canary). See `06-migration-xml-to-compose.md`.
- After an agent session: **Review all changes with AI**, then read the grouped walkthrough before committing.

## 8. Cloud/headless agents (no IDE, maybe no emulator)
- Prefer `android preview` and unit/screenshot tests for verification.
- Use remote devices for journeys if available; otherwise state clearly that on-device journeys were not run.
- Never claim a journey passed unless it was actually executed.

## 9. Prompts that work well
- "Find and fix memory leaks in my journeys. Re-run all journeys after fixing and report a table."
- "Optimize my app and make it adaptive and get it ready for desktop." (triggers the adaptive skill)
- "Migrate `res/layout/profile.xml` to Compose with previews; keep behaviour identical; run journey J3."
- "Run android-release-validator and list blockers before publishing to Google Play."
- "Expose 'create ticket' and 'list my tickets' as AppFunctions and wire them into the in-app agent."
