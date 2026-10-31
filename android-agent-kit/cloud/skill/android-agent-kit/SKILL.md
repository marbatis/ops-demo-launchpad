---
name: android-agent-kit
description: Use automatically for Android projects, builds, debugging, validation and release preparation. Kotlin/Compose defaults, API caveats, journeys and scoped Android subagents.
---

# Android Agent Kit

Apply this guidance to new native Android projects and existing Android development/builds. The skill is self-contained: resolve all relative paths from this skill's directory, not the working project. Load only the reference needed for the task.

Source: Marcelo's [Android agent kit PR #1](https://github.com/marbatis/ops-demo-launchpad/pull/1), pinned to `2982ccfe100bb98ce87c50e385b3a6081d0c4a3c`, retrieved 2026-10-31. The 13 original files are preserved under `assets/android-agent-kit/`. This snapshot does not update when the source PR changes.

## Apply within the project scope

- Read existing project instructions, build configuration, supported devices and journeys when present. No project-local kit files are required. Prefer applicable project details; qualify draft source recommendations using this skill and [adoption notes](references/adoption-notes.md).
- New native Android apps default to Kotlin, Compose, Material 3, adaptive layouts and ViewModel/StateFlow with lifecycle-aware collection. Scale architecture to the product. Resolve a compatible toolchain and libraries from current official sources.
- Existing apps retain their architecture, dependency versions and build setup unless the task requests changes. A Java/XML build does not authorize Compose migration. Do not automatically add AI, cloud services, AppFunctions, XR or new form factors.
- Automatic skill loading does not install project files, change dependencies or run a build. Perform only actions required by the user's current task.
- Read the adoption notes on first use and before implementing summit-specific APIs. All draft command names, API snippets, artifact versions and readiness claims need current evidence when material, even if not labeled `PATTERN`.

## Reference routing

| Task | Read |
|---|---|
| Understand an announcement | [Summit reference](assets/android-agent-kit/docs/00-adsummit-2026-reference.md) |
| Architecture or new app | [Architecture](assets/android-agent-kit/docs/01-architecture-standards.md) |
| Implementation, builds, debugging, leak checks | [Workflow](assets/android-agent-kit/docs/02-agentic-dev-workflow.md) |
| Requested AI/AppFunctions/A2UI/Connected Apps work | [AI playbook](assets/android-agent-kit/docs/03-ai-features-playbook.md) plus adoption notes |
| Adaptive UI or requested desktop/XR support | [Form factors](assets/android-agent-kit/docs/04-adaptive-desktop-xr.md) |
| Relevant PR checks or release preparation | [Quality checklist](assets/android-agent-kit/docs/05-quality-and-release-checklist.md) |
| Requested XML migration | [Migration](assets/android-agent-kit/docs/06-migration-xml-to-compose.md) |
| Product-specific verification flows | [Journey template](assets/android-agent-kit/templates/journeys.md) |

## Build and verify

1. Discover the project's Gradle wrapper, module/variant tasks, AGP/Kotlin versions, compatible JDK, required SDK packages and available devices. Check actual installed CLI identity/version/help before using newer Android commands.
2. Use the project's wrapper and intended variant. Build, run relevant existing tests/lint, inspect affected previews and execute affected journeys where the required tools/devices exist. Do not assume `app` or `debug` task names.
3. Fall back to supported Gradle, adb, emulator, IDE or screenshot-test workflows when new CLI preview/prototype/journey commands are unavailable. Never invent a command or count an unexecuted check as passed.
4. For lifecycle-sensitive changes, inspect ownership/cancellation and use available debug LeakCanary/heap checks. Fix causes, then re-run relevant regression journeys. Do not suppress leaks or disable tests, lint or R8 to get a passing result.
5. Adapt supplied journeys to the actual app. Login, expense actions, test tags and timing are examples. Use the project's supported-device matrix. Previews and prototypes do not establish device, leak or release readiness.
6. Report commands, build variant/artifact, results, journeys by device, leak evidence when relevant and checks not verified with reasons. An APK build alone does not require a full Play release audit. Verify current Play requirements when release readiness is actually requested.

Cloud-specific: the skill supplies guidance, not the SDK or device access. Inspect the configured cloud environment. Use the matching JDK and installed Android SDK packages. Do not assume an emulator, remote device, credentials, model availability or network permission exists.

## Android UI/UX

For substantial UI work, plan key flows and screen states before implementation. Reuse the app's design system; organize content around user tasks; handle loading/empty/error states; check accessibility, text scaling, keyboard/focus and adaptive sizing. Add appropriate Compose previews and verify on supported devices. Use relevant UI skills if actually available; this kit does not depend on a separate Codex skill or tool.

## AI features when requested

- Expose selected authorized domain capabilities, not every action automatically. Enforce authentication, permissions, confirmation for destructive/financial actions and idempotency in application/domain code.
- Verify exact Agent/AgentRunner/AppFunctions/A2UI signatures and artifacts, runtime device/model/download availability, lifecycle cleanup and execution restrictions. AppState remains opt-in while alpha; evaluate other preview dependencies explicitly.
- Choose Android AppFunctions versus remote MCP for the intended surfaces. One AppFunction does not automatically provide web/Search distribution. Connected Apps onboarding remains separate.
- Cloud fallback must honor the product's privacy, consent, credentials and cost constraints; otherwise provide a clear offline/unavailable state. Render model UI through a validated allow-listed catalog.

## Scoped subagents

Use these when delegation helps the current task; routine builds do not need all three. Cloud templates are bundled in `templates/agents/`, but uploading a skill does not register them. The cloud setup installs them in the runtime's user agent directory.

- `android-release-validator`: inspect and run scoped validation without editing source/configuration or publishing. Local build/test outputs are allowed for requested checks.
- `android-leak-hunter`: diagnose lifecycle/heap issues; apply fixes only when the task authorizes them.
- `android-adaptive-reviewer`: audit supported layouts, restoration, input and accessibility; edit only when improvements/fixes are requested.

Pass the project path, task scope, supported devices and relevant evidence to each delegate. If named agents are unavailable, report that and use an available general-purpose delegate or perform the scoped work directly; do not claim they ran.

## Boundaries

Never commit or push without Marcelo's explicit direction. In a task that says no commits or pushes, do neither, including temporary commits. Do not infer authority to merge, upload releases, publish, enroll in previews, purchase services or start background monitoring from this kit. Preserve current project instructions and user constraints.

The original README's copy script is archival: it overwrites files and mishandles relative paths/spaces. Do not execute it. Copy or merge project-local guidance only when the user's task calls for it, preserving existing files.
