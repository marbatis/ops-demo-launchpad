# Architecture Standards

Why these choices: Compose is declarative, state-driven and pure Kotlin — "what's easy for you is easy for your
agents". Google reports Compose migrations cut agent token cost/time by >33% and UI code by ~50% (Instagram
Direct), and native Kotlin beat React Native on startup by ~50% (Shopify). Agents reason best about one
language, one UI paradigm, unidirectional data flow.

## 1. Stack

| Concern | Choice | Notes |
|---|---|---|
| Language | Kotlin (latest stable), K2 compiler | No Java in new code. |
| UI | Jetpack Compose + Material 3 | No XML layouts, no Fragments for new screens. |
| Navigation | Navigation 3 | Use scene strategies for list-detail / supporting pane on large screens. |
| Adaptive | `material3-adaptive` (window size classes, `ListDetailPaneScaffold`, `NavigationSuiteScaffold`) | Required from the first screen. |
| State | `ViewModel` + `StateFlow`, UDF | `AppState` (ALPHA) only on explicit opt-in. |
| DI | Hilt | Constructor injection; no service locators. |
| Persistence | Room, DataStore (Preferences/Proto) | Room migrations tested. |
| Network | Retrofit or Ktor + kotlinx.serialization + OkHttp | Timeouts + retry policy set explicitly. |
| Concurrency | Coroutines + Flow | Inject dispatchers; no `GlobalScope`. |
| Background | WorkManager | |
| Images | Coil 3 | |
| Media | Media3 (ExoPlayer, Transformer) | GPU effects: Dawn/WebGPU where needed. |
| AI on device | ML Kit GenAI (Prompt API, Agentic API), Gemini Nano | See `03-ai-features-playbook.md`. |
| AI cloud | Firebase AI Logic (hybrid inference) | |
| Agent interop | AppFunctions (Android MCP), A2UI Compose | |
| Testing | JUnit, Turbine, Compose UI test, Roborazzi/Paparazzi or Compose Preview Screenshot Testing | |
| Build | Gradle Kotlin DSL, version catalog (`libs.versions.toml`), convention plugins | |
| Debug tooling | LeakCanary (debugImplementation), StrictMode in debug | |

Always resolve the **latest stable** version of each library when creating a project; do not copy versions
from memory.

## 2. Module layout

```
app/                      # Application, MainActivity, nav graph wiring, DI root
core/
  designsystem/           # Theme, tokens, shared components (also the A2UI catalog source)
  ui/                     # Reusable composables not tied to a feature
  model/                  # Pure Kotlin domain models
  data/                   # Repositories (interfaces + impl)
  database/               # Room
  network/                # API clients
  datastore/
  ai/                     # Model availability, Nano/Firebase AI wrappers, agent runner setup
  appfunctions/           # @AppFunction declarations (thin; delegate to domain use cases)
  testing/                # Fakes, test rules
feature/
  <name>/                 # One module per feature: screen(s), ViewModel, navigation entry
```

Rules:
- Features depend on `core:*`, never on other features.
- Domain logic lives in use cases/repositories, **not** in AppFunctions or ViewModels. AppFunctions, the in-app
  agent, and the UI all call the same use cases.

## 3. Screen pattern

```kotlin
data class TicketsUiState(
    val isLoading: Boolean = true,
    val tickets: List<TicketUi> = emptyList(),
    val error: String? = null,
)

@HiltViewModel
class TicketsViewModel @Inject constructor(
    private val observeTickets: ObserveOpenTicketsUseCase,
) : ViewModel() {
    val uiState: StateFlow<TicketsUiState> = observeTickets()
        .map { TicketsUiState(isLoading = false, tickets = it.map(Ticket::toUi)) }
        .catch { emit(TicketsUiState(isLoading = false, error = it.message)) }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), TicketsUiState())
}

@Composable
fun TicketsRoute(onOpen: (String) -> Unit, vm: TicketsViewModel = hiltViewModel()) {
    val state by vm.uiState.collectAsStateWithLifecycle()
    TicketsScreen(state = state, onOpen = onOpen)
}

@Composable
fun TicketsScreen(state: TicketsUiState, onOpen: (String) -> Unit) { /* stateless */ }
```

- Route composable = wiring; Screen composable = stateless, previewable, testable.
- Use `collectAsStateWithLifecycle`, `rememberSaveable` for UI state that must survive resize/rotation/process death.
- Add `Modifier.testTag("...")` to interactive and assertion-relevant elements (journeys rely on them).

## 4. Previews (mandatory)

Every screen ships with previews for at least: phone, foldable, desktop/tablet; light & dark; large font.

```kotlin
@PreviewScreenSizes          // phone, foldable, tablet, desktop
@PreviewLightDark
@PreviewFontScale
@Composable
private fun TicketsScreenPreview() = AppTheme { TicketsScreen(sampleState, onOpen = {}) }
```

These are what `android preview` renders for agent verification.

## 5. Leak-safe coding rules
- Never store `Activity`, `View`, or `Context` (other than `applicationContext`) in ViewModels, singletons, or
  companion objects.
- Unregister listeners/callbacks in `DisposableEffect`/`onDispose` or lifecycle observers.
- Use `viewModelScope` / `lifecycleScope`; never `GlobalScope`.
- Don't capture composable lambdas in long-lived objects.
- Bitmaps and media players: release in `onDispose` / `onCleared`.

## 6. Security baseline
- No secrets in source or `BuildConfig`. Use a backend or Secrets Gradle plugin for non-sensitive keys only.
- Network security config: HTTPS only. Certificate pinning only where justified.
- `android:exported` explicit on every component; least-privilege permissions.
- EncryptedFile / Keystore for sensitive local data.
- R8 enabled for release, with tested keep rules.

## 7. AppState (ALPHA) guidance
Use only when the user opts in. Good fit: session state shared across multiple UIs or app components (e.g.
signed-in user, active trip/booking shared between phone UI, widget, and agent). Keep a ViewModel/StateFlow
fallback path until it is stable. Verify API names against the current alpha docs.
