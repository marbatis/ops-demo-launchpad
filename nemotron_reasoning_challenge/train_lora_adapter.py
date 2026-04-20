from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path
from typing import Any, Iterable, List

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from nemotron_reasoning_challenge.package_adapter_submission import (
        build_submission_zip,
        validate_adapter_dir,
    )
else:
    from .package_adapter_submission import build_submission_zip, validate_adapter_dir

DEFAULT_TARGET_MODULES = r".*\.(in_proj|out_proj|up_proj|down_proj)$"
DEFAULT_FAMILY_OVERSAMPLE = "bit=4,equation=4,gravity=2,unit=2,cipher=1,roman=1,unknown=1"

KAGGLE_PLACEHOLDER_BASE_MODEL = "/kaggle/input/REPLACE_ME_WITH_BASE_MODEL_PATH"
KAGGLE_PLACEHOLDER_TRAIN_CSV = "/kaggle/input/nvidia-nemotron-model-reasoning-challenge/train.csv"


def render_prompt(prompt: str) -> str:
    return (
        "You are solving one reasoning task from the NVIDIA Nemotron benchmark.\n"
        "Read the full prompt carefully and answer with the final answer only.\n\n"
        f"Prompt:\n{prompt}\n\n"
        "Final answer:\n"
    )


def detect_prompt_family(prompt: str) -> str:
    prompt_lower = str(prompt).lower()
    if "bit manipulation rule transforms 8-bit binary numbers" in prompt_lower:
        return "bit"
    if "gravitational constant has been secretly changed" in prompt_lower:
        return "gravity"
    if "secret unit conversion is applied to measurements" in prompt_lower:
        return "unit"
    if "secret encryption rules are used on text" in prompt_lower:
        return "cipher"
    if "different numeral system" in prompt_lower:
        return "roman"
    if "secret set of transformation rules is applied to equations" in prompt_lower:
        return "equation"
    return "unknown"


def parse_family_oversample(raw: str) -> dict[str, int]:
    weights: dict[str, int] = {}
    if not raw.strip():
        return weights
    for chunk in raw.split(","):
        family, _, value = chunk.partition("=")
        family = family.strip()
        value = value.strip()
        if not family or not value:
            continue
        weights[family] = max(1, int(value))
    return weights


def load_training_rows(
    train_csv: Path,
    subset_size: int | None,
    seed: int,
    family_oversample: dict[str, int],
) -> list[dict[str, str]]:
    df = pd.read_csv(train_csv)
    df["family"] = df["prompt"].map(detect_prompt_family)
    if subset_size is not None and subset_size < len(df):
        df = df.sample(n=subset_size, random_state=seed).reset_index(drop=True)

    rows: list[dict[str, str]] = []
    for row in df.itertuples(index=False):
        prompt = str(row.prompt)
        answer = str(row.answer)
        family = str(row.family)
        repeat = family_oversample.get(family, 1)
        for _ in range(repeat):
            rows.append({"prompt": prompt, "answer": answer, "family": family})
    return rows


def build_tokenized_dataset(
    rows: Iterable[dict[str, str]],
    tokenizer: Any,
    max_length: int,
) -> Any:
    from datasets import Dataset

    eos_token = tokenizer.eos_token or ""
    encoded_rows: list[dict[str, list[int] | str]] = []
    for row in rows:
        prompt_text = render_prompt(row["prompt"])
        answer_text = f"{row['answer']}{eos_token}"
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        answer_ids = tokenizer(answer_text, add_special_tokens=False).input_ids

        if not answer_ids:
            continue
        if len(answer_ids) >= max_length:
            answer_ids = answer_ids[:max_length]
        prompt_budget = max(0, max_length - len(answer_ids))
        prompt_ids = prompt_ids[:prompt_budget]

        input_ids = prompt_ids + answer_ids
        encoded_rows.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": ([-100] * len(prompt_ids)) + answer_ids,
                "family": row["family"],
            }
        )

    return Dataset.from_list(encoded_rows)


def parse_target_modules(raw: str) -> List[str] | str:
    if "," in raw:
        return [module.strip() for module in raw.split(",") if module.strip()]
    return raw.strip()


def detect_compute_dtype() -> Any:
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def find_local_model_candidates(root: Path = Path("/kaggle/input")) -> List[Path]:
    if not root.exists():
        return []

    candidates: set[Path] = set()
    for marker in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        for path in root.rglob(marker):
            candidates.add(path.parent)

    scored = []
    for candidate in candidates:
        score = int((candidate / "config.json").exists()) + int(
            (candidate / "tokenizer_config.json").exists() or (candidate / "tokenizer.json").exists()
        )
        if score >= 2:
            scored.append(candidate)

    return sorted(scored)


def list_attached_input_dirs(root: Path = Path("/kaggle/input")) -> List[str]:
    if not root.exists():
        return []
    return sorted(str(path) for path in root.iterdir() if path.is_dir())


def find_train_csv_candidates(root: Path = Path("/kaggle/input")) -> List[Path]:
    if not root.exists():
        return []

    candidates = []
    for path in root.rglob("train.csv"):
        try:
            df = pd.read_csv(path, nrows=1)
        except Exception:
            continue
        columns = {str(column) for column in df.columns}
        if {"prompt", "answer"}.issubset(columns):
            candidates.append(path)
    return sorted(candidates)


def resolve_base_model_path(raw: str) -> str:
    candidate = Path(raw)
    if candidate.exists():
        return str(candidate)

    if raw == KAGGLE_PLACEHOLDER_BASE_MODEL or raw.startswith("/kaggle/input/"):
        discovered = find_local_model_candidates()
        if len(discovered) == 1:
            return str(discovered[0])

        attached = list_attached_input_dirs()
        discovered_text = "\n".join(f"- {path}" for path in discovered[:10]) or "- none"
        attached_text = "\n".join(f"- {path}" for path in attached[:20]) or "- none"
        raise FileNotFoundError(
            "Base model path is not configured.\n"
            f"Current value: {raw}\n"
            "Attach the Nemotron base model to the Kaggle notebook and set BASE_MODEL_PATH to that folder.\n"
            "Attached input directories:\n"
            f"{attached_text}\n"
            "Detected local model candidates:\n"
            f"{discovered_text}"
        )

    return raw


def resolve_train_csv_path(raw: Path) -> Path:
    if raw.exists():
        return raw

    raw_text = str(raw)
    if raw_text == KAGGLE_PLACEHOLDER_TRAIN_CSV or raw_text.startswith("/kaggle/input/"):
        discovered = find_train_csv_candidates()
        if len(discovered) == 1:
            return discovered[0]

        attached = list_attached_input_dirs()
        discovered_text = "\n".join(f"- {path}" for path in discovered[:10]) or "- none"
        attached_text = "\n".join(f"- {path}" for path in attached[:20]) or "- none"
        raise FileNotFoundError(
            "Training CSV path is not configured.\n"
            f"Current value: {raw}\n"
            "Attach the competition dataset and set TRAIN_CSV to the actual train.csv path if needed.\n"
            "Attached input directories:\n"
            f"{attached_text}\n"
            "Detected train.csv candidates:\n"
            f"{discovered_text}"
        )

    return raw


def check_runtime_compatibility() -> None:
    import torch

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        arch_list = set(torch.cuda.get_arch_list())
        requested_arch = f"sm_{major}{minor}"
        if requested_arch not in arch_list:
            supported = ", ".join(sorted(arch_list))
            device_name = torch.cuda.get_device_name(0)
            raise RuntimeError(
                "This Kaggle runtime cannot run the current PyTorch CUDA build on the selected GPU.\n"
                f"GPU: {device_name} (compute capability {major}.{minor})\n"
                f"PyTorch supported CUDA architectures: {supported}\n"
                "Use a T4, L4, A100, or another GPU with compute capability >= 7.0, "
                "or switch to a PyTorch build that supports this older GPU."
            )


def maybe_raise_missing_mamba_dependency(exc: Exception) -> None:
    message = str(exc)
    if "mamba-ssm" in message or "mamba_ssm" in message:
        installed = importlib.util.find_spec("mamba_ssm") is not None
        install_hint = (
            "mamba-ssm is not installed in this environment. "
            "Try `%pip install -q \"mamba-ssm[causal-conv1d]\" --no-build-isolation` "
            "in the notebook before loading the model."
            if not installed
            else "mamba-ssm still failed to import. Check that `causal-conv1d` and the CUDA build match the runtime."
        )
        raise RuntimeError(
            "The selected Nemotron base model is a hybrid Mamba-Transformer model and needs the `mamba-ssm` package.\n"
            f"{install_hint}"
        ) from exc


def apply_mode_defaults(args: argparse.Namespace) -> None:
    if args.mode == "smoke":
        if args.subset_size is None:
            args.subset_size = 256
        if args.max_steps is None:
            args.max_steps = 20
        if args.max_length == 1024:
            args.max_length = 512
        if args.gradient_accumulation_steps == 16:
            args.gradient_accumulation_steps = 8
        if args.logging_steps == 10:
            args.logging_steps = 1
        if args.save_steps == 100:
            args.save_steps = 20
        if args.lora_r == 16:
            args.lora_r = 8
        if args.lora_alpha == 32:
            args.lora_alpha = 16
        if args.family_oversample == DEFAULT_FAMILY_OVERSAMPLE:
            args.family_oversample = "bit=2,equation=2,gravity=1,unit=1,cipher=1,roman=1,unknown=1"
    else:
        if args.max_steps is None:
            args.max_steps = -1


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a Nemotron PEFT adapter on the local train.csv benchmark.")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--submission-zip", type=Path, default=None)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument(
        "--target-modules",
        type=str,
        default=DEFAULT_TARGET_MODULES,
        help="Comma-separated module list or a PEFT regex for LoRA injection.",
    )
    parser.add_argument(
        "--family-oversample",
        type=str,
        default=DEFAULT_FAMILY_OVERSAMPLE,
        help="Comma-separated family multipliers like `bit=4,equation=4,gravity=2`.",
    )
    parser.add_argument(
        "--load-mode",
        choices=("auto", "qlora", "lora"),
        default="auto",
        help="Model loading mode. `auto` tries 4-bit QLoRA first and falls back to non-quantized LoRA.",
    )
    args = parser.parse_args()
    apply_mode_defaults(args)

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    check_runtime_compatibility()
    resolved_base_model = resolve_base_model_path(args.base_model)
    resolved_train_csv = resolve_train_csv_path(args.train_csv)
    compute_dtype = detect_compute_dtype()
    tokenizer = AutoTokenizer.from_pretrained(resolved_base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    family_oversample = parse_family_oversample(args.family_oversample)
    training_rows = load_training_rows(
        resolved_train_csv,
        subset_size=args.subset_size,
        seed=args.seed,
        family_oversample=family_oversample,
    )
    tokenized_dataset = build_tokenized_dataset(rows=training_rows, tokenizer=tokenizer, max_length=args.max_length)

    def load_quantized_model() -> Any:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        return AutoModelForCausalLM.from_pretrained(
            resolved_base_model,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )

    def load_dense_model() -> Any:
        return AutoModelForCausalLM.from_pretrained(
            resolved_base_model,
            torch_dtype=compute_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    load_mode = args.load_mode
    load_error: str | None = None
    try:
        if args.load_mode == "qlora":
            model = load_quantized_model()
            load_mode = "qlora"
        elif args.load_mode == "lora":
            model = load_dense_model()
            load_mode = "lora"
        else:
            try:
                model = load_quantized_model()
                load_mode = "qlora"
            except Exception as exc:
                if isinstance(exc, ImportError):
                    maybe_raise_missing_mamba_dependency(exc)
                load_error = repr(exc)
                print(f"4-bit load failed, falling back to bf16/fp16 LoRA: {load_error}")
                model = load_dense_model()
                load_mode = "lora"
    except ImportError as exc:
        maybe_raise_missing_mamba_dependency(exc)
        raise
    model.config.use_cache = False
    if load_mode == "qlora":
        model = prepare_model_for_kbit_training(model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=parse_target_modules(args.target_modules),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "trainer_state"),
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=(compute_dtype == torch.bfloat16),
        fp16=(compute_dtype == torch.float16),
        report_to="none",
        optim="paged_adamw_8bit" if load_mode == "qlora" else "adamw_torch",
        save_total_limit=1,
        remove_unused_columns=False,
        group_by_length=True,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
        ),
    )
    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    config, weight_path = validate_adapter_dir(args.output_dir, max_rank=32)
    print(f"saved_adapter_dir={args.output_dir}")
    print(f"mode={args.mode}")
    print(f"load_mode={load_mode}")
    print(f"load_error={load_error}")
    print(f"train_rows={len(training_rows)}")
    print(f"family_oversample={family_oversample}")
    print(f"dataset_family_counts={tokenized_dataset.to_pandas()['family'].value_counts().to_dict()}")
    print(f"resolved_base_model={resolved_base_model}")
    print(f"resolved_train_csv={resolved_train_csv}")
    print(f"peft_type={config.get('peft_type')}")
    print(f"base_model_name_or_path={config.get('base_model_name_or_path')}")
    print(f"weight_file={weight_path.name}")

    if args.submission_zip is not None:
        files = build_submission_zip(args.output_dir, args.submission_zip)
        print(f"submission_zip={args.submission_zip}")
        print(f"submission_files={files}")


if __name__ == "__main__":
    main()
