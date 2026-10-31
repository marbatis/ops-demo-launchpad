# Migrating XML/Views to Compose

Google's position (19:58): all Android UI should be Compose; Views are in **maintenance mode**. Benefits seen:
UI code ~50% smaller and >33% lower agent token cost/time (Instagram Direct).

## 1. Tools
- **Android Studio (Canary): right-click an XML layout → Migrate to Compose.** The connected agent (Claude,
  Antigravity, Codex…) builds a step-by-step migration plan, produces artifacts, outputs Compose code and a
  Compose preview. Can be run per layout or across the project.
- Headless agents: do the same steps manually following this doc.

## 2. Strategy
1. **Inventory:** list layouts, Fragments, custom Views, adapters; rank by change frequency and complexity.
2. **Safety net first:** write/extend journeys and screenshot tests for each screen *before* migrating.
3. **Bottom-up or screen-by-screen:**
   - Leaf components first (`ComposeView` inside existing XML), or
   - Whole screens (Fragment → composable destination) when the screen is self-contained.
4. **Theme bridge:** create the Compose `AppTheme` mapped from existing XML theme attributes/colours so both
   look identical during transition.
5. **State:** move logic into ViewModel + `StateFlow`; the composable stays stateless.
6. **Interop:** `AndroidView` for Views that can't move yet (maps, legacy custom views); `ComposeView` for
   Compose inside Views.
7. **Navigation:** migrate to Navigation 3 once most screens are Compose.
8. **Delete** XML, adapters, ViewBinding and unused resources after each screen is verified.

## 3. Per-screen checklist
- [ ] Previews for phone / foldable / desktop, light/dark.
- [ ] Visual parity confirmed with `android preview` vs old screenshots.
- [ ] Same `testTag`s or updated journeys.
- [ ] Accessibility parity (content descriptions, focus order).
- [ ] Journeys pass; leak check run.
- [ ] Old XML and code removed.

## 4. Common mappings
| View | Compose |
|---|---|
| RecyclerView + Adapter | `LazyColumn` / `LazyVerticalGrid` with stable `key` |
| ConstraintLayout | `Column`/`Row`/`Box`; `ConstraintLayout` for Compose only when truly needed |
| ViewPager2 | `HorizontalPager` |
| Toolbar | `TopAppBar` |
| BottomNavigationView | `NavigationSuiteScaffold` |
| DialogFragment | `AlertDialog` / `ModalBottomSheet` |
| LiveData | `StateFlow` + `collectAsStateWithLifecycle` |
| SharedPreferences | DataStore |

## 5. Prompt
"Migrate `<layout>.xml` (and its Fragment) to Compose. Keep behaviour and visuals identical. Add previews for
phone/foldable/desktop. Move state to the ViewModel as StateFlow. Run journeys <ids> before and after. Delete
the old XML and binding code. Report a before/after screenshot comparison."
