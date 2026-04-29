#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print({"cmd": cmd, "cwd": str(cwd) if cwd else None})
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def load_dataset_status(dataset_id: str) -> bool:
    result = subprocess.run(
        ["kaggle", "datasets", "status", dataset_id],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        return False
    return "Ready" in result.stdout


def write_metadata(path: Path, dataset_id: str, title: str) -> None:
    metadata = {
        "title": title,
        "id": dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (path / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create/update a Kaggle dataset from a local adapter directory."
    )
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True, help="Format: username/dataset-slug")
    parser.add_argument("--title", default="Nemotron Adapter Artifact")
    parser.add_argument("--message", default="Update adapter artifact")
    args = parser.parse_args()

    adapter_dir = args.adapter_dir.expanduser().resolve()
    if not adapter_dir.exists() or not adapter_dir.is_dir():
        raise FileNotFoundError(f"adapter directory not found: {adapter_dir}")

    required = adapter_dir / "adapter_config.json"
    if not required.exists():
        raise RuntimeError(f"missing required file: {required}")

    with tempfile.TemporaryDirectory(prefix="nemotron_dataset_") as tmp:
        stage = Path(tmp)
        for entry in adapter_dir.iterdir():
            if entry.is_file():
                shutil.copy2(entry, stage / entry.name)
        write_metadata(stage, args.dataset_id, args.title)

        exists = load_dataset_status(args.dataset_id)
        if exists:
            run(["kaggle", "datasets", "version", "-p", str(stage), "-m", args.message])
        else:
            run(["kaggle", "datasets", "create", "-p", str(stage)])


if __name__ == "__main__":
    main()
