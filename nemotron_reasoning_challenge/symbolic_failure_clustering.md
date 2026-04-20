# Symbolic Failure Clustering

Current baseline on the fixed family-wise `80/20` holdout:

- overall: `78.56%`
- equation: `17.04%`

This note summarizes the **remaining 155 symbolic misses** under that baseline.

## Main clusters

1. `one_shot_visible_but_unresolved` - `40` rows
   - Exactly one same-operator example
   - Uniform RHS length
   - All answer characters are already visible somewhere in the prompt
   - Current solver still returns `None`
   - Read: the operator is partially exposed, but one example is not enough for the current copy-only symbolic prior to generalize

2. `zero_shot_copyish_fallback` - `30` rows
   - No same-operator examples at all
   - Solver returns a non-`None` answer, usually target-copyish
   - Most of these are over-copy errors, not total failures
   - Read: the global fallback prior is too weakly conditioned and tends to emit structurally plausible but wrong visible-symbol copies

3. `multi_example_uniform_dsl_gap` - `24` rows
   - Two or more same-operator examples
   - Uniform RHS length
   - Solver still returns `None`
   - Read: even with enough local evidence, the current symbolic DSL still cannot express the transformation

4. `multi_example_mixed_length_gap` - `18` rows
   - Two or more same-operator examples
   - Mixed RHS lengths
   - Solver returns `None`
   - Read: the current exact symbolic miner assumes a fixed answer length per prompt slice, so these rows mostly bypass exact fitting entirely

5. `one_shot_novel_char_gap` - `7` rows
   - Exactly one same-operator example
   - Uniform RHS length
   - Gold answer contains characters not visible anywhere in the prompt
   - Read: these likely require prompt-local remapping or operator-conditioned generation, not pure visible-symbol extraction

6. `wrong_prior_or_partial_copy` - smaller tail
   - Non-`None` prediction, but neither the no-same-op over-copy bucket nor an exact local solve
   - Read: mixed bag of poor prior ranking and partial visible-symbol programs

## Strongest quantitative signal

- Misses with `0` same-operator examples: `36`
- Misses with `1` same-operator example: `68`
- Misses with `2+` same-operator examples: `51`

So the dominant remaining problem is still **insufficient same-operator support**, not just DSL expressivity.

Another useful split:

- Misses where all gold characters are visible somewhere in the prompt: `130`
- Misses with novel gold characters: `25`

That means the next improvement should still primarily target **better structured reuse of visible prompt symbols**, not open generation.

## Operator concentration

The remaining misses are still concentrated in:

- `-`: `53`
- `*`: `34`
- `+`: `23`

So the challenge is not mostly rare punctuation operators. The common operator buckets are still underperforming.

## What this implies

The next solver iteration should probably focus on one of these, in order:

1. **Better zero-shot / one-shot symbolic priors**
   - condition symbolic priors on more than just `target_op`
   - target equality signature or simple prompt-structure signature is the obvious next conditioning axis

2. **Mixed-length same-op handling**
   - current exact symbolic miner becomes much less useful when same-op RHS lengths vary
   - a length-conditioned symbolic prior or per-length local fit would directly address `18` misses

3. **Prompt-local remapping for the visible-symbol majority**
   - most misses still use only visible symbols, but not in a way the current flat atom set can recover
   - this points toward better structure-conditioned priors before another DSL expansion

## What looks less promising

- More tiny DSL atom additions by themselves
  - we already tested two small symbolic DSL expansions
  - gains were marginal or attributable to better candidate filtering, not the new atoms

- Equality-guard branching
  - tested and rejected

- Broad retrieval / exact-canonical matching
  - tested and rejected earlier
