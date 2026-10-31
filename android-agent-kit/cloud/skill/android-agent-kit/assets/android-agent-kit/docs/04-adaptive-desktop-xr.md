# Adaptive, Desktop (Google Book) and XR

Compose is adaptive at its core: build once, run on phone, foldable, tablet, car, desktop and XR.
Business reasons from the keynote: phone + large-screen users spend **9x more**; Notability saw **12x** higher
buyer conversion after optimising; Google Play will **review foldable quality next year** and grants a
**Made for / Optimized for desktop** badge that boosts large-screen recommendations.

---

## 1. Requirements for every screen

| Area | Requirement |
|---|---|
| Layout | Driven by window size class (compact / medium / expanded), never by device type or `isTablet` flags. |
| Navigation | `NavigationSuiteScaffold` (bottom bar → rail → drawer). List-detail via Navigation 3 scene strategy / `ListDetailPaneScaffold`. |
| Content width | Constrain reading width on expanded screens; no full-width stretched text or forms. |
| Foldables | Handle table-top and book postures; avoid placing critical content across the hinge. |
| State | Survives rotation, fold/unfold, window resize, multi-window, process death (`rememberSaveable`, ViewModel, SavedStateHandle). Text field contents must persist through resize. |
| Input | Keyboard navigation and focus order; keyboard shortcuts for primary actions; mouse hover states; right-click/context menu where natural; stylus where relevant. |
| Pointer | Correct pointer icons (hand on clickable, text cursor on text). |
| Tooltips | On icon-only buttons. |
| Orientation | No orientation locks; no fixed aspect ratios. |
| Resizable | `resizeableActivity` true; test free-form windows. |

## 2. Desktop / Google Book specifics
- Hardware baseline: ≥16 GB RAM, NPU 45+ TOPS — enables local ML and heavy media workloads.
- **Continue On**: users hand off from phone to Google Book in one click (app-to-app or app-to-web). Make deep
  links and state restoration robust so a handed-off task resumes where it left off.
- Expected desktop journeys: open/create with keyboard shortcuts (e.g. Ctrl+N new item, Enter send, Esc
  close), multi-pane browsing, drag-and-drop where sensible, copy/paste, window resize without state loss.
- Developer environment: Google Book ships a Linux terminal VM with sudo and Antigravity (beta) preinstalled;
  one-click VM reset keeps host files — suitable for running agent workflows locally.

### Using the adaptive skill
Prompt your agent: *"Optimize my app, make it adaptive and get it ready for desktop."* In the demo (40:03) the
skill produced: Nav3 list-detail scene strategy, state handling fixes on resize, keyboard shortcuts, tooltips,
custom pointer cursors. Always review the diff and re-run journeys on phone + foldable + desktop afterwards.

```kotlin
// Keyboard shortcut example (verify against current Compose APIs)
Modifier.onPreviewKeyEvent { e ->
    if (e.type == KeyEventType.KeyDown && e.isCtrlPressed && e.key == Key.N) { onNewConversation(); true }
    else false
}
```

## 3. Android XR
- **Android XR** is one platform for headsets and glasses. **XREAL Aura** (wired glasses, optical see-through,
  hand + voice input, puck trackpad) runs existing phone/tablet apps **out of the box** as spatial windows.
- Baseline for every app: be fully adaptive and resizable → it already works well on XR.
- When XR adds real value (dashboards, multi-panel workspaces, media, 3D product views, hands-free field work):
  - Use **Jetpack XR SDK** (STABLE since 28 Oct 2026) to extend Compose UI into 3D: spatial panels, anchor
    panels alongside 3D elements, orbiters, environments.
  - Test on the **Android XR emulator** (STABLE).
  - Engine-based apps: Android XR engine hub for Unity / Unreal / Godot (STABLE).
- Design notes: users are often seated/moving (plane, hotel desk, sofa); system moves content aside when
  walking; keep text large, contrast high, interactions forgiving; support Gemini "sees what you see" by
  keeping UI semantics/accessibility labels accurate.

## 4. Other form factors
- **Android Auto / cars with Google built-in**: 250M+ compatible cars; use Car App Library templates where
  the app category qualifies.
- **Wear OS, TV**: only when the product needs them; share domain/data modules.

## 5. Verification matrix (minimum)

| Journey type | Phone | Foldable (both postures) | Desktop (resize + keyboard/mouse) | XR (emulator, optional) |
|---|---|---|---|---|
| Core flows | ✅ | ✅ | ✅ | ◻️ |
| State on resize/fold | — | ✅ | ✅ | ◻️ |
| Keyboard shortcuts | — | — | ✅ | — |

Use Android CLI remote devices for real foldable hardware and the prototype runner (EARLY) for quick
multi-size checks.
