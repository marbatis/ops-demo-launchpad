from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import zipfile


REQUIRED_CONFIG = "adapter_config.json"
WEIGHT_CANDIDATES = ("adapter_model.safetensors", "adapter_model.bin")
OPTIONAL_FILES = (
    "README.md",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def detect_weight_file(adapter_dir: Path) -> Path:
    for name in WEIGHT_CANDIDATES:
        candidate = adapter_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing adapter weights in {adapter_dir}. Expected one of: {', '.join(WEIGHT_CANDIDATES)}"
    )


def validate_adapter_dir(adapter_dir: Path, max_rank: int | None) -> tuple[dict, Path]:
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
    if not adapter_dir.is_dir():
        raise NotADirectoryError(f"Adapter path is not a directory: {adapter_dir}")

    config_path = adapter_dir / REQUIRED_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing {REQUIRED_CONFIG} in {adapter_dir}. This competition expects a PEFT adapter artifact, not a CSV."
        )

    config = load_json(config_path)
    weight_path = detect_weight_file(adapter_dir)

    peft_type = config.get("peft_type")
    if peft_type is None:
        raise ValueError(f"{config_path} is missing 'peft_type'")

    rank = config.get("r")
    if max_rank is not None and rank is not None and rank > max_rank:
        raise ValueError(f"Adapter rank {rank} exceeds max supported rank {max_rank}")

    base_model = config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(f"{config_path} is missing 'base_model_name_or_path'")

    return config, weight_path


def build_submission_zip(adapter_dir: Path, output_zip: Path) -> list[str]:
    files_to_include = [REQUIRED_CONFIG]
    for name in WEIGHT_CANDIDATES:
        if (adapter_dir / name).exists():
            files_to_include.append(name)
            break
    for name in OPTIONAL_FILES:
        if (adapter_dir / name).exists():
            files_to_include.append(name)

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in files_to_include:
            zf.write(adapter_dir / name, arcname=name)
    return files_to_include


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and package a Kaggle Nemotron adapter submission."
    )
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument(
        "--output-zip",
        type=Path,
        default=Path("/Users/marcelosilveira/iCloud Drive (Archive)/Documents/Playground/output/submission.zip"),
    )
    parser.add_argument(
        "--max-rank",
        type=int,
        default=32,
        help="Fail validation if adapter_config.json declares a larger LoRA rank.",
    )
    args = parser.parse_args()

    config, weight_path = validate_adapter_dir(args.adapter_dir, max_rank=args.max_rank)
    files = build_submission_zip(args.adapter_dir, args.output_zip)

    print(f"adapter_dir={args.adapter_dir}")
    print(f"output_zip={args.output_zip}")
    print(f"peft_type={config.get('peft_type')}")
    print(f"base_model_name_or_path={config.get('base_model_name_or_path')}")
    print(f"rank={config.get('r')}")
    print(f"weights={weight_path.name}")
    print(f"zip_files={files}")


if __name__ == "__main__":
    main()
