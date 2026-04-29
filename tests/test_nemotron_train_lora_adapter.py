from pathlib import Path

import pytest

from nemotron_reasoning_challenge.train_lora_adapter import (
    resolve_resume_checkpoint,
    resolve_warmup_steps,
    validate_sync_adapter_request,
)


def test_resolve_warmup_steps_prefers_explicit_value() -> None:
    assert (
        resolve_warmup_steps(
            explicit_warmup_steps=12,
            warmup_ratio=0.03,
            train_size=1000,
            micro_batch_size=1,
            gradient_accumulation_steps=16,
            max_steps=20,
            num_train_epochs=1.0,
        )
        == 12
    )


def test_resolve_warmup_steps_uses_fixed_max_steps() -> None:
    assert (
        resolve_warmup_steps(
            explicit_warmup_steps=None,
            warmup_ratio=0.03,
            train_size=1000,
            micro_batch_size=1,
            gradient_accumulation_steps=16,
            max_steps=20,
            num_train_epochs=1.0,
        )
        == 1
    )


def test_resolve_resume_checkpoint_auto_uses_latest_checkpoint(tmp_path: Path) -> None:
    trainer_state_dir = tmp_path / "trainer_state"
    (trainer_state_dir / "checkpoint-2").mkdir(parents=True)
    latest = trainer_state_dir / "checkpoint-20"
    latest.mkdir()
    (trainer_state_dir / "checkpoint-final").mkdir()

    assert resolve_resume_checkpoint(tmp_path, "auto") == str(latest)


def test_validate_sync_adapter_request_requires_packaging_first(tmp_path: Path) -> None:
    sync_script = tmp_path / "sync.py"
    sync_script.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="requires --submission-zip"):
        validate_sync_adapter_request(None, "owner/dataset-slug", sync_script)


def test_validate_sync_adapter_request_returns_normalized_target(tmp_path: Path) -> None:
    sync_script = tmp_path / "sync.py"
    sync_script.write_text("", encoding="utf-8")

    assert (
        validate_sync_adapter_request(tmp_path / "submission.zip", " owner/dataset-slug ", sync_script)
        == "owner/dataset-slug"
    )


def test_validate_sync_adapter_request_rejects_non_dataset_target(tmp_path: Path) -> None:
    sync_script = tmp_path / "sync.py"
    sync_script.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="dataset-style target"):
        validate_sync_adapter_request(tmp_path / "submission.zip", "owner", sync_script)
