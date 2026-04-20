# Deep Research Brief: Nemotron `equation` Symbolic Solver

## Goal

Improve the `equation` family in the Kaggle `NVIDIA Nemotron Model Reasoning Challenge`, specifically the **symbolic** half, from the current local holdout of about `15.1%` family accuracy to something materially higher.

Current overall local holdout is about `78.24%`, driven by:

- `bit`: `83.5%`
- `cipher`: `100%`
- `roman`: `100%`
- `gravity`: `85.6%`
- `unit`: `84.0%`
- `equation`: `15.1%`

The numeric half of `equation` is now partially improved. The remaining hard blocker is the symbolic half.

## Problem Shape

Symbolic `equation` prompts look like this:

- examples of `lhs = rhs`
- `lhs` is a 5-character string, usually of the form:
  - two-symbol left operand
  - one operator symbol
  - two-symbol right operand
- `rhs` is length `1` to `4`
- the target asks for one more `lhs`

Important dataset facts from local analysis:

- symbolic training rows: `823`
- symbolic eval rows on the current holdout: `174`
- target operators are dominated by:
  - `-`
  - `*`
  - `+`
- many symbolic prompts have **no example with the same target operator**
- most answer characters are visible somewhere in the prompt:
  - `722 / 823` train rows have answer charset contained in visible prompt chars

## What Already Works

The current solver is in:

- [solver.py](/Users/marcelosilveira/iCloud%20Drive%20%28Archive%29/Documents/Playground/nemotron_reasoning_challenge/solver.py)

The current `equation` logic:

- numeric:
  - candidate function library over two 2-digit numbers
  - prompt-local candidate intersection when target operator is seen in examples
  - learned target-operator priors when target operator is missing or ambiguous
- symbolic:
  - exact same-operator subsequence templates only

That symbolic solver currently gets only:

- `14 / 174` correct symbolic eval rows

## Local Findings

### 1. Same-op exact subsequence templates are high precision but tiny coverage

They explain a few rows cleanly, but are nowhere near enough.

### 2. Prompt-global all-pairs symbolic inference did not help

Using all example pairs rather than same-operator pairs reduced precision and did not improve holdout.

### 3. Simple operand-sequence rule libraries did not help enough

Tried raw rules like:

- concat
- reversed concat
- unique union
- symmetric difference
- left-only / right-only
- intersection

Both direct versions and operator-prior versions were too weak.

### 4. Operator-global template plus prompt-local substitution is the most plausible unsolved direction

There is evidence that symbolic operators may correspond to a **global structural template**, while each prompt defines a **local symbol remapping**.

But current attempts did not yet solve it:

- operator-global template with one global substitution map: failed
- operator-global template with prompt-local char map: failed
- operator-global pattern templates with prompt-local map: failed
- joint search over templates for `-`, `*`, `+` using prompt-local map: failed

### 5. Numeric side is no longer the main blocker

Numeric `equation` improved from about:

- `27 / 137` to `33 / 137`

So future research should focus primarily on symbolic.

## What To Research

Please focus on **symbolic program induction / transducer inference** for synthetic symbol-manipulation tasks with:

- small input strings
- tiny output strings
- operator-conditioned transformations
- local per-instance symbol remappings

The most promising research targets are:

1. **Operator-global structural program + prompt-local cipher**
   - output built from a fixed operator-specific source template
   - then transformed by a prompt-local substitution / transducer

2. **Prompt-local graph matching**
   - infer a consistent symbol mapping from all example equations jointly
   - then apply an operator prior to unseen target operators

3. **Finite-state / transducer search**
   - low-arity symbolic transducers for 5-character inputs to 1-4 character outputs
   - conditioned on the center operator

4. **Analogy / meta-solver retrieval**
   - retrieve structurally similar train prompts
   - transfer answers through a learned prompt-level symbol alignment

5. **Neural or ML-style meta-models**
   - if there is a practical small-model or feature-engineered approach that works well on synthetic symbolic IO tasks like this, especially without requiring a heavy training stack

## Specific Questions

1. Is there a known compact program induction approach for tasks of the form:
   - `xyOpzw -> short symbolic output`
   - where `Op` has global meaning
   - but actual symbols are remapped per prompt?

2. Is there a practical way to infer:
   - operator templates
   - prompt-local symbol maps
   - jointly, with beam search or EM-like alternating optimization?

3. Are there robust feature-engineered baselines for this kind of task, such as:
   - typed edit transducers
   - weighted finite-state transducers
   - SMT-style program search
   - graph isomorphism plus answer transfer?

4. If you had to design a solver to get symbolic accuracy above `50%`, what would the architecture be?

## Constraints

- We want practical code that can live in this repo.
- Prefer Python.
- Avoid solutions that require large external training pipelines unless the gain is clearly worth it.
- Any proposal should be compatible with iterative local validation through:
  - [evaluate.py](/Users/marcelosilveira/iCloud%20Drive%20%28Archive%29/Documents/Playground/nemotron_reasoning_challenge/evaluate.py)

## Current Best Local Artifact

- current solver: [solver.py](/Users/marcelosilveira/iCloud%20Drive%20%28Archive%29/Documents/Playground/nemotron_reasoning_challenge/solver.py)
- current notes: [README.md](/Users/marcelosilveira/iCloud%20Drive%20%28Archive%29/Documents/Playground/nemotron_reasoning_challenge/README.md)
- current submission file: [nemotron_submission.csv](/Users/marcelosilveira/iCloud%20Drive%20%28Archive%29/Documents/Playground/output/nemotron_submission.csv)
