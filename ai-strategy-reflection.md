# The Bottleneck Moved: Working With AI as a One-Person Company

*Course 3 project. Draft 3, 2026-10-28.*

---

## 1. My Field

I'm a solo technical founder. That's the polite way of saying I'm a one-person software company. I pick what to build, I build it, I get it through Apple and Google review, I keep it running, and I deal with the ads, the payments, the sales tax, and the support emails. There's no one to hand anything to.

Claude's description of the role was the usual list: product decisions, building features, shipping releases, running production, getting users, running the business, and learning new stuff fast. What it can't tell me is which of those matter on a given day. A store rejection at 9am changes the whole plan. The thing I'm actually short of is not ideas or code. It's attention, and knowing where to put it.

Four things define the job:

- **Deciding what to build** (and what to kill, and what to charge).
- **Building and shipping it** across iOS, Android, web, and backend.
- **Keeping it running** safely: monitoring, security, store and tax compliance.
- **Getting users and reading the numbers** honestly before I change anything.

---

## 2. What AI Can Actually Do Today

I use Claude every day, so here are three things I actually did with it.

**Store localization.** Old way: write the App Store copy in English, translate it myself, resize screenshots, check character limits. I gave Claude the English listing for SpeakSignal and asked: *"Act as an app store optimization specialist. Write the Brazilian Portuguese listing copy and screenshot captions for this $4.99 offline pronunciation app, within Apple's limits."* I had usable pt-BR copy in a couple of minutes. I'm Brazilian, so I caught a few too-literal phrases. The rest I kept.

**Weekly security audit.** Old way: once a week, click through GitHub and Heroku checking alerts, two-factor status, old tokens, config vars. Prompt: *"Write a scheduled job that audits my GitHub repos and Heroku apps weekly for these items and sends me a summary."* I got a working automation and it's running now. But I had to check every API permission it assumed, and I killed one check that would have printed a secret into the summary email. Fast, but not trustworthy without a read.

**Ad budget.** Old way: open Apple Ads, compare cost per install in the US and Brazil, decide. I asked: *"Here's my campaign data. Should I raise the budget?"* When I pasted the actual numbers, the answer was clear and well-organized. Without the numbers, I got equally confident advice based on nothing. That one changed how I work: AI only advises from real account data, and it shows me the tradeoffs instead of a verdict.

So where does it help? Speed, obviously. Structure: complete drafts and checklists instead of a blank page. Clarity, when it explains a tax rule or an API error. Where it fails: silent errors. A pipeline of mine was quietly using a different model than I thought, and I only noticed when quality dropped. Assumed permissions. Advice without data. Everything gets a review before it touches production or money.

**AI in my field can reliably do drafting (copy, translations, docs), code and automation scaffolding, and a first-pass analysis of data I give it.**

---

## 3. Where I'm Vulnerable and Where I Have Leverage

I ran the futurist prompt as written, with "solo technical founder."

In a year: boilerplate code, store copy, translations, docs, and routine ops checks are commodity. In two years, whole features from a spec, dependency upgrades, first-line support, routine filings. In five, most of "build and ship" is an agent pipeline I supervise. Leverage: treat agents like junior staff now. Spec, sandbox, review gate. The human premium is taste, accountability, and knowing when a confident answer is wrong. The 30-day habit: log every AI output I had to fix, and why.

I agree with most of it. The build-and-ship part isn't a prediction for me, it's already how I work. Where I disagree: it put "running production" in the two-year automation bucket. In my experience ops work is cheap to generate and expensive to get wrong. A leaked config value or a wrong tax setting is not a bug I can patch next release. The limit there is risk, not capability.

My own take: I'm not competing with AI for a job. What changed is the bottleneck. I used to be limited by how much I could build. Now I'm limited by how much I can review. The founder who reviews faster and more reliably ships more.

**The most vulnerable parts of my work are building/shipping and routine operations. The highest-value human skills are product judgment and being the person who actually verifies.**

---

## 4. Where Responsibility Stays With Me

- **Augment:** feature coding, translations, docs, analysis of data I supply, draft support replies.
- **Automate with controls:** security audits and dependency updates (read-only runs unattended, changes need my approval), release builds and store submission behind a checklist, sales tax calculation verified with a test transaction.
- **Keep human-led:** what to build and what to charge, any change to money (pricing, tax registration, ad budget), store policy decisions, anything touching user data, and the final "ship it."

My rule of thumb: if the output is executable, financial, or public, it gets verified. If the right answer depends on context the model doesn't have (my users, my market, my own capacity this month), it's my call. If a mistake is irreversible or has legal consequences, I don't delegate it.

**AI can assist with building features and writing or translating content. It can automate ops checks and release pipelines with proper controls. But product and pricing decisions, and anything touching money, tax, or user data, stay on me.**

---

## 5. My Human-in-the-Loop Map

| | |
|---|---|
| **Augment** | Feature design and coding, store copy and localization, data analysis, learning new domains fast |
| **Automate with controls** | Weekly security audits, dependency updates, release builds, tax test transactions, support triage |
| **Keep human-led** | Product and pricing decisions, ad budgets, tax and compliance filings, user data, production sign-off |

What's getting automated: scaffolding, drafts, routine checks. What's being augmented: feature work, analysis, learning. What's worth more now: choosing well, reviewing fast, owning the result.

**My positioning: AI can draft, build, and audit. It introduces risk through silent errors and confident advice without data. So my value is deciding what's worth building and being the accountable reviewer of everything AI produces.**

---

## 6. My Plan

**This year.** Start the corrections log. Write an explicit review gate for every pipeline I automate. Keep the rule that AI shows me the tradeoff before anything touches money, tax, or store settings.

**Two years.** Move from one agent per task to supervised pipelines: spec, build, test, review. Success is simple: my review time per release goes down and the incident count doesn't go up.

**Five years.** Run a small portfolio of products with agent teams doing most of the building. What sets me apart then is taste in what to build and a record of operating safely. I stay in the loop, on purpose, for every decision about money, tax, stores, or user data. That part I don't plan to automate.
