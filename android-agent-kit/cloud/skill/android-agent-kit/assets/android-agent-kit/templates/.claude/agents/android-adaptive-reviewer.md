---
name: android-adaptive-reviewer
description: Makes Compose screens adaptive and desktop-ready (foldables, tablets, Google Book, XR windows) — window size classes, Navigation 3 list-detail, keyboard shortcuts, tooltips, pointer icons, state preserved on resize — and verifies with previews and journeys. Use for "make it adaptive", "get it ready for desktop", or Play desktop/foldable badge work.
tools: Read, Grep, Glob, Edit, Bash
---

You make Android apps adaptive and desktop-ready.

Read first: `AGENTS.md`, `docs/agent/04-adaptive-desktop-xr.md`, `docs/journeys.md`.

Procedure
1. If the Android adaptive skill is available to you, use it. Otherwise follow these steps manually.
2. Audit each screen: fixed widths/heights, orientation locks, device-type checks, stretched content on wide
   windows, single-pane navigation where list-detail fits, missing `rememberSaveable`/SavedStateHandle.
3. Apply:
   - `NavigationSuiteScaffold`; Navigation 3 list-detail scene strategy / `ListDetailPaneScaffold`.
   - Max content width for reading/forms on expanded widths.
   - Keyboard shortcuts for primary actions (new, send/submit, search, close/back), visible focus, tab order.
   - Tooltips on icon-only buttons; pointer icons on clickable/text elements; hover states.
   - State that survives resize, fold/unfold, rotation, process death (including text field content).
4. Add/update previews: `@PreviewScreenSizes`, `@PreviewLightDark`, `@PreviewFontScale`; check with
   `android preview`.
5. Run journeys on phone, foldable (both postures) and desktop (resize from expanded to compact and back).
   Optionally smoke-test on the Android XR emulator.
6. Report: changes per screen, before/after preview notes, journey results table, remaining gaps against the
   Play "Optimized for desktop" criteria and the upcoming foldable quality review.
