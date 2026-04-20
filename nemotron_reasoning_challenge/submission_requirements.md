# Nemotron Kaggle Submission Requirements

## What failed

The direct upload of `submission.zip` failed on Kaggle because the archive only contained:

- `nemotron_submission.csv`

Kaggle's submission details panel reported:

- `No adapter_config.json found in submission`

This is strong evidence that the competition expects a **PEFT adapter artifact**, not a plain prediction CSV.

## Practical implication

The local reasoning solver and `nemotron_submission.csv` are still useful for rapid iteration and analysis, but they are **not** directly uploadable to this competition.

To submit successfully, we need to produce an adapter directory from a real fine-tuning run, then package it into `submission.zip`.

## Likely required root files

Based on the Kaggle error and the standard Hugging Face PEFT checkpoint format:

- `adapter_config.json`
- `adapter_model.safetensors` or `adapter_model.bin`

Optional files may also be safe to include:

- `README.md`
- tokenizer metadata files

## Packaging step

Use the local helper once a real adapter directory exists:

```bash
python3 -m nemotron_reasoning_challenge.package_adapter_submission \
  --adapter-dir /path/to/adapter_dir \
  --output-zip '/Users/marcelosilveira/iCloud Drive (Archive)/Documents/Playground/output/submission.zip'
```

## Open item

We still need the **training path** that creates a compatible adapter for the competition base model. The current repo does not yet have that fine-tuning workflow.
