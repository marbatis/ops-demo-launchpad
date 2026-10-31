# AI Features Playbook

How to add intelligent and agentic features to an Android app using what was announced at Android Dev Summit
2026. **Verify all API names against current docs** — most of this is new, alpha or preview. Code marked
`// PATTERN` shows the shape demonstrated on stage, not guaranteed signatures.

---

## 1. Architecture: write each capability once

```
                ┌──────────── Google surfaces ─────────────┐
                │ Gemini app / Search  (Connected Apps)    │
                │   ← remote MCP server (web/mobile/Search)│
                │   ← Android MCP = AppFunctions (on device)│
                └──────────────────────────────────────────┘
                                   ▲
┌─────────────────────── your app ─┼───────────────────────────┐
│  Chat UI (Compose) ← A2UI Compose renderer ← agent responses  │
│  In-app agent: Gemini Nano 4 + ML Kit Agentic API (offline)   │
│                └─ cloud fallback: Firebase AI Logic           │
│  Tools = @AppFunction  ───────────────► domain use cases      │
└───────────────────────────────────────────────────────────────┘
```

One `@AppFunction` serves: the in-app on-device agent, system Gemini via Android MCP, and Connected Apps.
Don't build separate integrations.

## 2. AppFunctions / Android MCP

**What:** Jetpack AppFunctions lets your app expose typed functions ("tools") to agents. Android MCP is the
unified tool-calling framework built on it (31:09).

**Design rules**
- One function per user intent: `createTicket`, `listOpenTickets`, `addExpense`, `bookSeat`.
- KDoc is the tool description — write it for an LLM: what it does, when to use it, units, formats.
- Parameters: simple, typed, explicit units/enums; no free-form JSON blobs.
- Return small structured results (IDs, summaries), not whole object graphs.
- Delegate to the same use cases the UI uses. No business logic inside the function.
- Check auth/permissions inside the function; treat every caller as untrusted.
- **Destructive, financial, or irreversible actions require user confirmation.** (Connected Apps renders
  confirmation cards for sensitive actions; your in-app agent must do the same.)
- Idempotency keys for create/pay operations.

```kotlin
// PATTERN — verify package/annotation names against current androidx.appfunctions docs
class ExpenseFunctions @Inject constructor(
    private val addExpense: AddExpenseUseCase,
    private val listTrip: ListItineraryUseCase,
) {
    /**
     * Adds an expense to the user's current trip.
     * @param description short label, e.g. "Eiffel Tower admission"
     * @param amount amount in major units (e.g. 47.0)
     * @param currency ISO-4217 code, e.g. "EUR"
     * @return the created expense id
     */
    @AppFunction
    suspend fun addExpense(
        context: AppFunctionContext,
        description: String,
        amount: Double,
        currency: String,
    ): String = addExpense(description, amount, currency).id

    /** Returns today's itinerary items for the current trip (title, time, price if known). */
    @AppFunction
    suspend fun getItinerary(context: AppFunctionContext): List<ItineraryItem> = listTrip()
}
```

**Testing:** unit-test each function; use the AppFunctions test agent and **Gemini dev mode** (34:31) to
simulate agent calls, inspect schemas and validate tool calls.

## 3. On-device agent: Gemini Nano 4 + ML Kit Agentic API

**When:** privacy-sensitive data, low latency, offline (the demo ran in airplane mode), zero per-call cost.
Gemini Nano 4 supports **native function calling**. The **Agentic API** in ML Kit GenAI auto-executes functions,
stores conversation history, runs the orchestration loop, and adds automatic guardrails (28:07–30:49).

**Steps shown on stage (29:49):**
1. Define tools — `@AppFunction` (e.g. `addExpense`).
2. Define the agent — name, system instructions, persona, relevant context; call `enableAppFunctions()` to give
   it the in-app tool registry.
3. Create an **agent runner** — manages the execution loop.
4. **Start a session** — keeps conversation history for follow-ups ("actually I paid 30 euros").
5. Send the prompt to the session; update UI from the **events** it emits (progress, tool calls, final answer).

```kotlin
// PATTERN ONLY — class/method names reconstructed from the keynote, not from published API docs.
val agent = Agent(
    name = "TripAssistant",
    systemInstructions = """
        You help the user manage trip expenses. Use tools to read the itinerary and add or update expenses.
        Confirm amounts and currency. Never invent prices; use itinerary data or ask.
    """.trimIndent(),
).enableAppFunctions()

val runner = AgentRunner(agent)
val session = runner.startSession()

viewModelScope.launch {
    session.send("Add the admission ticket for the Eiffel Tower to my expenses").collect { event ->
        _uiState.update { it.reduce(event) }   // progress, tool call, text, done, error
    }
}
```

**Must-dos**
- Check model availability/download state at runtime; show a graceful fallback.
- Fallback to cloud via **Firebase AI Logic hybrid inference** (e.g. prefer-on-device mode) when Nano is
  unavailable, or disable the feature with a clear message.
- Keep system instructions short and specific; list tool usage rules.
- Expose progress in UI (the demo shows step-by-step progress).
- Log tool calls (without PII) for debugging.

## 4. Agent-generated UI: A2UI + Compose renderer (ALPHA)

**What:** A2UI is an open protocol where the agent returns UI as declarative JSON. The Compose renderer draws
it using **your** component catalog, so it looks and behaves native (24:21–27:47).

**Building blocks**
- **Catalog:** register allowed components — Material 3 primitives (Card, Column, Row, Checkbox, Divider…) and
  custom domain components (e.g. `FlightConfigurator`, `SeatMap`, `TicketCard`).
- **Surface composable:** renders whatever the agent streams into your chat.
- **Body slots:** a component can declare a slot the agent swaps (flight picker → seat map) in place; Compose
  transitions from a loading shimmer.
- **JSON patches:** server sends small diffs (e.g. seat 3C taken) instead of resending the tree.
- **Property binding:** catalog properties are backed by Compose state; local taps and server patches
  recompose only observing components (price updating as perks are ticked).

**Rules**
- Catalog = allow-list. Unknown components render a safe fallback, never raw content.
- Every catalog component: accessible labels, theming from your design system, previews.
- User actions in A2UI components go back to the agent **or** directly to AppFunctions/use cases.
- Confirmation step before payment/booking.
- Don't open WebViews for agent UI.

## 5. Connected Apps Platform (PREVIEW)

**What:** your app's capabilities appear inside **Gemini app and Google Search** with a **branded interactive
widget** (your logo, colours, typography), so users can act without opening the app (32:48).

**How**
1. Identify 2–5 high-intent user tasks (order, book, check status, acknowledge alert).
2. Implement them as AppFunctions (on-device via Android MCP) and/or a **remote MCP server** (cross-platform
   reach: web, mobile, Search).
3. Design the branded widget for each.
4. Validate in **Gemini dev mode**: simulate interactions, inspect schemas, validate tool calls.
5. Apply to the preview program for production; track performance after launch.

Sensitive actions automatically get an interactive confirmation card — still enforce server-side checks.

## 6. Cloud models
- Use Firebase AI Logic for cloud Gemini; never ship API keys in the APK — use App Check and/or a backend proxy.
- For heavy or long-context tasks, run in the cloud; keep personal data on device when possible.
- On Google Book (NPU 45+ TOPS, ≥16 GB RAM) larger local models are viable; detect capability, don't assume.

## 7. Safety & privacy checklist for AI features
- [ ] Data minimisation: only send what the task needs; on-device for sensitive data.
- [ ] User confirmation for destructive/financial actions, in-app and via Gemini.
- [ ] Prompt-injection: treat tool outputs and fetched content as data, not instructions.
- [ ] Rate limits and timeouts on cloud calls; offline behaviour defined.
- [ ] Clear disclosure that AI is used; easy way to undo agent actions.
- [ ] Play policy for AI-generated content reviewed (release validator checks this).

## 8. Suggested implementation order
1. Domain use cases + UI (no AI).
2. AppFunctions for top intents + unit tests.
3. In-app agent with Nano 4 Agentic API + cloud fallback.
4. A2UI catalog + chat surface.
5. Validate in Gemini dev mode; apply to Connected Apps preview.
