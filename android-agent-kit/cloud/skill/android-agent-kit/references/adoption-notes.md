# Adoption context and source checks

Requested by Marcelo on 2026-10-31 for future Android projects and builds.

## Provenance

- Source: https://github.com/marbatis/ops-demo-launchpad/pull/1
- Repository: `marbatis/ops-demo-launchpad`
- Pinned head: `2982ccfe100bb98ce87c50e385b3a6081d0c4a3c`
- Retrieved: 2026-10-31. PR open, draft, unmerged at retrieval.
- Snapshot: all 13 `android-agent-kit/` files, including its README. Original bytes verified against GitHub's Git blob SHA values.
- The source describes the Android Dev Summit presentation of 2026-10-28. Its claims are presentation notes, not a continuously maintained API contract. The original transcript was not independently audited here.
- This integration does not change, merge, monitor, or publish the source PR. Refresh the snapshot deliberately from a verified commit and review differences before replacing it.

## How to resolve the draft's over-broad rules

Project instructions, actual requirements, and current official documentation determine implementation. The original kit is a reference. New native apps default to Kotlin/Compose; existing builds do not trigger wholesale stack migration or automatic dependency upgrades. AI, AppFunctions, desktop additions, XR, cloud services, and preview enrollment are scoped product decisions.

Verify compatibility as a set (JDK, Gradle, AGP, Kotlin, Compose, SDK and libraries), rather than independently choosing every latest version. Do not assume the source's fixed module layout is necessary for small apps. Verify current secure-storage recommendations rather than blindly adopting the source's `EncryptedFile` suggestion.

Adapt journeys to real product behavior and test data. Source timing, authentication, expenses and test tags are examples. A rendered preview/prototype is not proof of runtime, device, leak, or release readiness. Label missing device/tool checks as not verified.

The source expense `PATTERN` directly executes a mutation without the auth/confirmation checks described in its prose. Enforce those checks before side effects, and use a suitable exact money representation instead of copying its `Double` example.

## Bounded official-source verification — 2026-10-31

These checks correct the draft's highest-risk assumptions. Re-check the linked docs when implementing; this is not an exhaustive validation of the kit.

1. **CLI:** official preview docs list `android preview list` and `android preview render`. Confirm the installed tool supports them. This check did not locate an official command reference for `android prototype <apk>`; keep it a demonstration until help/docs confirm availability. The summit post describes LeakCanary integration but says its CLI subcommand is coming soon, so inspect installed capabilities. Sources: https://developer.android.com/tools/agents/android-cli/commands/preview and https://developer.android.com/blog/posts/android-dev-summit-26-developer-tools-performance

2. **ML Kit agent API:** `Agent` and `AgentRunner` now have official alpha documentation, but this does not validate the kit's reconstructed snippet. The official guide uses the builder, `getOrCreateSession(...)`, `session.runAsync(Content)`, and `runner.close()`. Resolve exact imports, dependencies, signatures, and cleanup from the current guide: https://developers.google.com/ml-kit/genai/agent/android/get-started

3. **Nano availability:** the checked setup guide requires Android 14+ and AICore Developer Preview enrollment for this agent setup, despite library `minSdk` 26. Device support/model versions vary, with foreground and quota restrictions. Check runtime availability and product fallback behavior; offline capability does not imply universal availability. Sources: https://developers.google.com/ml-kit/genai/agent/android/get-started and https://developers.google.com/ml-kit/genai

4. **Connected Apps:** remote MCP and Android MCP/AppFunctions are complementary integration paths. Remote MCP is the documented default for cross-platform capabilities. Sharing domain logic is useful; a single AppFunction is not automatic web/Search reach. Preview onboarding/review and connection-type selection remain separate requirements. Sources: https://developers.google.com/connected-apps/mcp/ and https://developers.google.com/connected-apps/gpc/perform-the-steps

5. **AppState:** the official guide identifies it as alpha. Retain ViewModel/StateFlow by default and use AppState when opted into for the project. Source: https://developer.android.com/topic/libraries/architecture/appstate

6. **XR:** stability differs by artifact. At this check Runtime, ARCore, and SceneCore had stable 1.0.0 releases while XR Compose was `1.0.0-beta02`. XR Compose notes flagged a transitive SceneCore RC leak and documented a stable override. Resolve current dependency versions and release notes before implementing; do not apply a historical override blindly. Sources: https://developer.android.com/jetpack/androidx/versions and https://developer.android.com/jetpack/androidx/releases/xr-compose

## Cloud package layout

Resolve reference paths from the loaded skill directory. The archive preserves the original source at `assets/android-agent-kit/`; the seven source docs remain under its `docs/` folder. Active cloud agent templates are under `templates/agents/`. Their bootstrap installer puts the skill at the cloud runtime's `$HOME/.claude/skills/android-agent-kit` and the three agent definitions in `$HOME/.claude/agents`. It may set `ANDROID_AGENT_KIT_ROOT` to select another installation root.

The original raw copy script and source AGENTS.md are archival references, not automatic project configuration. No source PR mutation or repository commit/push is needed to use this account-enabled skill.
