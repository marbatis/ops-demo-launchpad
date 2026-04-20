# NVIDIA Nemotron Model Reasoning Challenge

First-pass local workspace for the Kaggle competition:

- Competition: `NVIDIA Nemotron Model Reasoning Challenge`
- Kaggle URL: `https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge`
- Deadline: `2026-06-15 23:59:00 UTC`
- Current public files: `train.csv`, `test.csv`
- Kaggle description: `Advance reasoning techniques using NVIDIA Nemotron open models on a novel benchmark`

## Dataset shape

The released training set has `9,500` labeled examples. The public `test.csv` has only `3` rows, which suggests the real scoring likely happens on a hidden test at submission time.

There are exactly six prompt families:

1. `bit` - 8-bit binary transformations
2. `gravity` - infer a hidden gravitational constant and compute distance
3. `unit` - infer a hidden unit-conversion ratio
4. `cipher` - monoalphabetic substitution over a small English vocabulary
5. `roman` - convert integers to Roman numerals
6. `equation` - symbolic transformation over punctuation-like tokens

## First prototype

The initial solver intentionally focuses on the most structured families:

- exact solver for `roman`
- exact solver for `gravity`
- exact solver for `unit`
- vocabulary-constrained substitution solver for `cipher`
- hybrid beam-guided synthesis solver for `bit`
  - wide beam search over rotated/shifted bases, boolean compositions, and `maj` / `ch`
  - legacy shallow exact fallback preserved for cases the scored beam misses
- heuristic subfamily solver for `equation`
  - numeric `DD?DD` prompts via learned operator priors over a candidate function library
    - prompt-local candidate intersection when examples exist for the target operator
    - global target-operator priors when the prompt underdetermines the target operator
  - symbolic prompts via:
    - exact same-operator subsequence templates when the operator behavior is directly exposed
    - tiny copy-only symbolic programs with prompt-local role fitting as a fallback
    - operator-conditioned and global program priors over a small atom set (`slot`, `other`, shared / left-only / right-only, `role`)
    - a candidate-filtered exact symbolic program miner that learns these priors without brute-force full Cartesian enumeration

## Current local holdout

With a family-wise shuffled `80/20` split over `train.csv`, the current baseline reaches about `78.61%` overall accuracy.

Latest measured family scores:

- `roman`: `100%`
- `cipher`: `100%`
- `bit`: `83.5%`
- `gravity`: `85.6%`
- `unit`: `84.0%`
- `equation`: `17.4%`

That makes the current bottlenecks very clear:

1. improve `equation`, especially the symbolic half
2. tighten `gravity` and `unit` edge rounding cases
3. keep pushing `bit` only if the extra search depth pays for itself on the real Kaggle runtime budget

## Current read on `equation`

The numeric half now has a usable heuristic baseline, but the symbolic half is still the main blocker. The current symbolic path now does slightly more than raw subsequence templates:

- exact same-operator subsequences first
- then fallback to tiny exact symbolic programs learned from training support
- exact prompt-local role unification for those fallback programs
- candidate filtering by output column before symbolic program mining, which improved symbolic coverage without broadening the DSL
- operator-specific priors first, then a global prior backstop
- a very narrow symbolic retrieval fallback that only shortens obvious over-copied predictions when a high-similarity same-operator neighbor supports the shorter prefix

I also tested several richer symbolic abstractions locally:

- exact subsequence templates
- position templates plus prompt-local substitutions
- operand-level sequence / set rules like concat, reverse, unique-union, and symmetric difference
- operator-level subsequence priors when a prompt lacks same-operator examples

None of those generalized well enough to improve the holdout reliably. So the current repo keeps the symbolic solver conservative instead of adding a large rule library that looks clever but does not move measured accuracy.

## Files

- `solver.py` - routed solver implementation
- `evaluate.py` - local holdout evaluation harness
- `predict.py` - full-train prediction script for `test.csv`
- `train_lora_adapter.py` - CLI fine-tuning entrypoint that trains a PEFT / LoRA adapter and can emit `submission.zip`
- `package_adapter_submission.py` - validate and zip a PEFT adapter artifact as `submission.zip`
- `analyze_symbolic_failures.py` - failure clustering for the current symbolic equation baseline
- `iteration_loop.py` - repeatable experiment runner that evaluates current code against a saved stable baseline
- `baseline_metrics.json` - current stable baseline snapshot used by the loop runner

## Reproducible Environment (Recommended)

Use this flow to avoid fragile Kaggle notebook setup/debug loops.

1. Bootstrap a dedicated virtual environment:

```bash
bash '/Users/marcelosilveira/iCloud Drive (Archive)/Documents/Playground/scripts/nemotron_bootstrap_env.sh'
source '/Users/marcelosilveira/iCloud Drive (Archive)/Documents/Playground/.venv_nemotron/bin/activate'
```

2. Preflight Kaggle auth + competition access:

```bash
python '/Users/marcelosilveira/iCloud Drive (Archive)/Documents/Playground/scripts/nemotron_pipeline.py' preflight
```

3. Train + package + submit (single command):

```bash
python '/Users/marcelosilveira/iCloud Drive (Archive)/Documents/Playground/scripts/nemotron_pipeline.py' run-all \
  --mode smoke \
  --base-model '/kaggle/input/models/ashok205/nvidia-nemotron-3-nano-30b/pytorch/pytorch-v1/2' \
  --train-csv '/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge/train.csv' \
  --message 'nemotron smoke run'
```

Notes:

- If you only want to train/package without submitting, add `--no-submit`.
- The pipeline validates `submission.zip` and fails fast if `adapter_config.json` or non-trivial adapter weights are missing.
- For local/Colab workflows, set `NEMOTRON_BASE_MODEL` and `NEMOTRON_TRAIN_CSV` once in env variables to avoid passing them every run.

4. Optional: sync local adapter artifacts to a private Kaggle dataset:

```bash
python '/Users/marcelosilveira/iCloud Drive (Archive)/Documents/Playground/scripts/nemotron_sync_adapter_dataset.py' \
  --adapter-dir '/Users/marcelosilveira/iCloud Drive (Archive)/Documents/Playground/output/nemotron_adapter' \
  --dataset-id 'marcsilveira/nemotron-adapter-artifacts' \
  --title 'Nemotron Adapter Artifacts' \
  --message 'adapter update'
```

## Kaggle Submission Format

This competition is **not** scored by uploading a plain `submission.csv`.

The Kaggle error page for the failed upload showed:

- `No adapter_config.json found in submission`
- the uploaded archive only contained `nemotron_submission.csv`

That means the competition expects a **model adapter artifact**, not a prediction file. The public error is consistent with a Hugging Face PEFT / LoRA-style submission package.

Minimum expected files at the root of `submission.zip`:

- `adapter_config.json`
- `adapter_model.safetensors` or `adapter_model.bin`

The helper script below validates an adapter directory and creates the correctly named archive:

```bash
python3 -m nemotron_reasoning_challenge.package_adapter_submission \
  --adapter-dir /path/to/adapter_dir \
  --output-zip '/Users/marcelosilveira/iCloud Drive (Archive)/Documents/Playground/output/submission.zip'
```

If training code eventually produces a PEFT adapter via `save_pretrained()`, this script is the last packaging step before Kaggle upload.

Starter notebook artifact:

- [nemotron-kaggle-lora-starter.ipynb](/Users/marcelosilveira/iCloud%20Drive%20%28Archive%29/Documents/Playground/output/jupyter-notebook/nemotron-kaggle-lora-starter.ipynb)

Recommended training flow:

1. Run the trainer or notebook in `smoke` mode first.
2. Verify the adapter saves cleanly and `submission.zip` contains `adapter_config.json` at the zip root.
3. Switch to `full` mode only after the smoke run succeeds.

Recommended CLI entrypoint:

```bash
python3 -m nemotron_reasoning_challenge.train_lora_adapter \
  --mode smoke \
  --train-csv '/kaggle/input/nvidia-nemotron-model-reasoning-challenge/train.csv' \
  --base-model '/kaggle/input/ashok205-nvidia-nemotron-3-nano-30b/pytorch/default/2' \
  --output-dir '/kaggle/working/nemotron_adapter' \
  --submission-zip '/kaggle/working/submission.zip'
```

Current recommended training defaults:

- answer-only supervision: prompt tokens are masked out of the loss, so training pressure is spent on the target answer instead of prompt copying
- family-weighted sampling: default oversampling is `bit=4,equation=4,gravity=2,unit=2,cipher=1,roman=1,unknown=1`
- smoke mode keeps the same objective but reduces scope and weights for quick validation
- default LoRA target modules now use the broader demo-style regex `.*\\.(in_proj|out_proj|up_proj|down_proj)$`

Useful override:

```bash
--family-oversample 'bit=6,equation=6,gravity=2,unit=2,cipher=1,roman=1,unknown=1'
```

## Experiment Loop

Use the loop runner to evaluate any new iteration against the fixed holdout and the current stable baseline:

```bash
python3 -m nemotron_reasoning_challenge.iteration_loop \
  --train-csv '/Users/marcelosilveira/iCloud Drive (Archive)/Documents/Playground/tmp/nemotron/train.csv' \
  --eval-fraction 0.2 \
  --seed 7 \
  --label my-next-try
```

That writes:

- a JSON run summary with keep / revert / neutral deltas
- a symbolic failure CSV for the exact same run

under `output/nemotron_loop_runs`.
