# Force Multiplier, Not Autopilot: What AI Actually Does in a One-Person Software Company

*Course 2 mini-project. Draft 1, 2026-10-28.*

---

## 1. My Field

I'm a solo technical founder. I run a one-person software company: I decide what to build, I build it, I get it through Apple and Google review, I keep it running, and I handle the ads, payments, sales tax, and support myself. The workflows are the whole org chart compressed into one person, so my day is a mix of coding, release management, security and ops checks, store listings, ad campaigns, and bookkeeping.

That's exactly why it's a good case study. A one-person company is the purest test of "force multiplier." There's no team to absorb AI's output and no one to catch its mistakes but me. If AI genuinely multiplies work, I ship more with the same hours. If it only looks like it does, I pay for it in bugs, bad decisions, or a tax problem. The test is simple: what shipped, and what broke.

---

## 2. What AI Can Do Today

I use Claude (claude.ai and Claude Code) daily, plus Perplexity when I need sources. Here are three real tasks.

**Task 1: Store localization.** Old way: write the App Store listing in English, translate it myself, check Apple's character limits by hand. Prompt: *"Act as an app store optimization specialist. Here is the English listing for SpeakSignal, a $4.99 offline pronunciation app. Write the Brazilian Portuguese listing and screenshot captions within Apple's limits. Match the tone of the English."* Output: complete pt-BR copy in minutes. I'm Brazilian, so I fixed a few phrases that were too literal and kept the rest. *Course 2 connection:* this is pure prompting strategy. Giving a role, the source text as an example, and hard constraints (character limits, tone) is what made the copy usable.

**Task 2: Weekly security audit.** Old way: click through GitHub and Heroku once a week checking dependency alerts, two-factor status, stale tokens, config vars. Prompt: *"Write a scheduled job that audits my GitHub repos and Heroku apps weekly for these items and emails me a summary."* Output: a working automation that now runs every week. *Course 2 connection:* this is tool use. The model wrote code that calls external APIs on my behalf. The value is not the text it produced but the system it wired up.

**Task 3: Texas sales tax on a web subscription.** Old way: read state guidance, maybe call an accountant. Prompt, with search enabled: *"Think step by step. Is a monthly web subscription to a language-practice app taxable in Texas? Cite the rule and show your reasoning."* Output: a reasoned walkthrough that landed on "taxable as data processing, on 80% of the price," with the citation. I verified it against the state's own page and then confirmed the 80% base with a test charge before going live. *Course 2 connection:* chain-of-thought plus retrieval. Asking for the reasoning made the answer checkable; search gave me something to check it against.

A fourth concept, memory, shows up across all three. Claude Code keeps a memory file of facts about my projects, so I don't re-explain my setup every session. I haven't fine-tuned anything; a good system prompt and a few skill files cover the specialization I need.

**AI in my field can reliably do drafting (copy, translations, docs), code and automation scaffolding, and reasoned first-pass research when I ask it to show its work and give it sources.**

---

## 3. Boundaries and Limitations

**Confident advice from nothing.** I once asked whether to raise an ad budget without pasting the campaign data. I got a clear, structured, confident recommendation based on nothing. *Alignment and reward optimization:* models are trained to be helpful and fluent, and "I need the numbers first" is less rewarded than a nice answer. Now AI only advises from real account data and shows me tradeoffs.

**Silent tool failures.** A grading pipeline I built was quietly using a different model than I'd specified because of a wrong model name. Nothing errored. I noticed when quality dropped. *Tool reliability and benchmark limitations:* the model scored great on paper, but the system around it failed silently, and benchmarks don't measure that.

**Assumed permissions.** The audit job assumed API scopes it didn't have and one check would have printed a secret into the summary email. Code that runs is not code that's safe.

**Stale memory.** The memory file is useful until a fact goes stale. It once still "knew" a campaign setting I had changed. *Memory persistence risk:* persistence helps continuity but makes old facts look current.

**Lost context.** In long sessions the conversation gets summarized and details fall out. *Context window constraints:* long doesn't mean infinite.

These matter because in my business there's no second reviewer. A leaked secret, a wrong tax setting, or a budget change based on invented numbers hits production and money directly.

**AI struggles with silent errors and with decisions that need data it doesn't have, and anything touching money, tax, stores, or user data requires human judgment.**

---

## 4. Ethical and Professional Considerations

Three that are live for me. **Accountability:** when the tax filing is wrong, the state doesn't care that an AI suggested the rate. I sign it. **Privacy:** SpeakSignal processes speech. I built it to run offline specifically so user audio never leaves the phone and never reaches a model provider. **Skill atrophy:** when most of my code is AI-written, I'm at risk of losing the ability to read it critically, which is the one skill I most need.

Responsible use in a solo shop: review everything executable, financial, or public before it ships; keep user data out of AI pipelines unless there's a clear reason; and keep writing some code by hand so I can still judge the rest.

---

## 5. Positioning Myself Alongside AI

**Automated:** boilerplate code, store copy, translations, documentation, routine ops checks. **Augmented:** feature work, debugging, data analysis, learning unfamiliar domains fast. **More valuable:** deciding what to build, reading real numbers honestly, and verifying output as fast as it's produced.

I used to be limited by how much I could build. Now I'm limited by how much I can review. The founder who reviews faster and more reliably is the one who ships more. That's where I differentiate: I'm the accountable reviewer with the domain context, not the fastest typist.

**AI can draft, build, and audit. It cannot reliably catch its own silent errors or make decisions that depend on data and context it doesn't have. Therefore, my value lies in deciding what's worth building and being the accountable reviewer of everything AI produces.**

---

## 6. What I Learned

AI is the best junior staff I've ever had and the worst at telling me when it's wrong. Its strength is speed and structure: I get complete drafts, working automations, and reasoned research in minutes. Its limitation is that nothing in it signals when it has drifted, assumed, or invented. My role is shifting from builder to reviewer and decision-maker. 

If this is what AI can do today, tomorrow I should be preparing for systems that run longer with less input from me. That means building review gates now, keeping a log of what I had to correct, and staying sharp enough in the fundamentals to still know what good looks like.
