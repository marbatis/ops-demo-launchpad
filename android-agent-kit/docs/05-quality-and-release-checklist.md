# Quality & Release Checklist

Use as the **definition of done** for every PR and as the checklist for the `android-release-validator`
subagent before a Play release. Quality drives retention and engagement — agents are fast, but this list is
what makes the output shippable.

## A. Every PR
- [ ] Builds (debug + release) with no new warnings that matter.
- [ ] Unit tests and Compose UI tests green; new logic has tests.
- [ ] Previews exist and render via `android preview` for phone / foldable / desktop, light/dark, large font.
- [ ] Affected journeys in `docs/journeys.md` executed on required devices; results reported.
- [ ] Leak check run if lifecycle-sensitive code changed; zero new leaks.
- [ ] Lint clean (incl. Compose lint rules); no suppressed warnings without justification.
- [ ] No secrets, tokens, or real user data committed.
- [ ] Diff reviewed with **Review all changes with AI** (Android Studio) or equivalent agent review; issues
      addressed or explicitly deferred.

## B. Accessibility
- [ ] All interactive elements have content descriptions / semantic roles; decorative images marked.
- [ ] Touch targets ≥ 48dp; colour contrast ≥ 4.5:1 for text.
- [ ] Works with TalkBack; logical focus order; works with keyboard only.
- [ ] Text scales to 200% without clipping.
- [ ] A2UI catalog components meet the same rules.

## C. Performance
- [ ] Baseline Profile + Startup Profile generated; cold start measured (Macrobenchmark).
- [ ] No jank in key scrolls (stable keys in lazy lists, no heavy work in composition).
- [ ] Compose stability: avoid unstable params in hot paths; use skill guidance for Compose performance.
- [ ] Memory: no leaks across all journeys; heap reasonable after stress steps.
- [ ] R8 full mode on; APK/AAB size tracked.

## D. Security & privacy
- [ ] HTTPS only; network security config set.
- [ ] Exported components explicit and minimal; intent filters validated.
- [ ] Sensitive storage encrypted; no PII in logs.
- [ ] Cloud AI calls protected (App Check / backend proxy); no keys in APK.
- [ ] AI tool calls with side effects require confirmation and server-side authorization.
- [ ] Data safety form matches actual data collection.

## E. Adaptive & form factors
- [ ] Phone, foldable (postures), tablet, desktop journeys pass; state survives resize/fold.
- [ ] Keyboard shortcuts, tooltips, pointer icons on desktop.
- [ ] Meets criteria for Play **Optimized for desktop** badge (if targeting Google Book).
- [ ] Ready for Play's upcoming **foldable quality** review.
- [ ] Runs as a spatial window on Android XR emulator (smoke test).

## F. AI features
- [ ] Model availability checked; offline/unsupported device fallback works.
- [ ] AppFunctions documented (KDoc), unit-tested, validated in Gemini dev mode.
- [ ] Prompt-injection and over-permission reviewed.
- [ ] AI disclosure and undo for agent actions.
- [ ] Play AI-generated content policy reviewed.

## G. Google Play release
- [ ] targetSdk meets current Play requirement.
- [ ] Play policy insights skill run — no blockers.
- [ ] Store listing screenshots for phone, foldable/tablet, desktop (and XR if supported).
- [ ] Staged rollout plan + crash/ANR monitoring (Android vitals).
- [ ] Release notes written.

## Report format (for agents)
```
## Release readiness: <app> <version>
### Blockers (must fix)
- [area] issue — file:line — suggested fix
### High
### Medium
### Passed checks
### Not verified (and why)
```
