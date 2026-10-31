# User Journeys

Agents run these journeys to verify features, catch regressions and detect memory leaks (Android CLI journeys,
remote devices, prototype runner). Keep them current: **a feature without a journey is not verified.**

Conventions
- IDs are stable (`J1`, `J2`…). Never renumber; deprecate instead.
- Steps reference `testTag`s in backticks or visible text in quotes.
- `Expect:` lines are assertions. Every journey ends with at least one.
- `Devices:` any of `phone`, `foldable`, `desktop`, `xr`.
- `Leak-stress:` steps that repeat lifecycle transitions; always run during leak checks.
- Test data: debug build only — user `test@example.com`, password from `local.properties` key `TEST_PASSWORD`
  (never commit real credentials).

---

## J1 — Launch and sign in
Devices: phone, foldable, desktop
1. Cold start the app.
   Expect: `login_screen` visible within 2s.
2. Enter test credentials in `email_field`, `password_field`; tap `sign_in_button`.
   Expect: `home_screen` visible; `home_list` has ≥ 1 item.
Leak-stress: sign out and sign in again 5x.

## J2 — Browse list and open detail
Devices: phone, foldable, desktop
1. From `home_screen` tap the first item in `home_list`.
   Expect: phone → `detail_screen` full screen; foldable/desktop → list and detail side by side.
2. Navigate back (system back / Esc on desktop).
   Expect: list scroll position preserved.
Leak-stress: open and close detail 20x; rotate (phone) / fold-unfold (foldable) / resize window (desktop) 5x.

## J3 — Create item
Devices: phone, desktop
1. Tap `fab_create` (desktop: press Ctrl+N).
   Expect: `create_sheet` visible, focus in `title_field`.
2. Type "Journey test item", choose priority "High", tap `save_button` (desktop: Enter).
   Expect: item appears at top of `home_list` with "High" badge.
3. Resize window / rotate.
   Expect: item still present; no duplicate.

## J4 — In-app agent (AI)
Devices: phone (with and without network)
1. Open `assistant_screen`. Enable airplane mode.
2. Send "Add a 47 euro expense for the Eiffel Tower ticket".
   Expect: progress events shown; confirmation card if amount-affecting; expense appears in `expenses_list`
   with 47.00 EUR.
3. Send "Actually I paid 30".
   Expect: same expense updated to 30.00 EUR (conversation context kept).
4. On a device without Gemini Nano: Expect graceful fallback message or cloud path (when online).
Leak-stress: open/close assistant 10x mid-response.

## J5 — Background and restore
Devices: phone, foldable
1. Start J3 step 2 but don't save; send app to background; trigger process death (CLI/dev option); return.
   Expect: draft text restored.

## J6 — Desktop keyboard and pointer
Devices: desktop
1. Navigate entire home screen with Tab/Shift+Tab.
   Expect: visible focus indicator, logical order.
2. Hover icon-only buttons.
   Expect: tooltips; hand pointer on clickables.
3. Resize window from expanded to compact and back.
   Expect: layout adapts; text in any open field preserved.

<!-- Add one journey per feature. Copy a block above and adapt. -->
