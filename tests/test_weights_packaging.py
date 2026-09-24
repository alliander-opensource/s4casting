# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0
"""Tests for safetensors conversion, format-aware weight loading and release packaging."""

import io
import json
import pathlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import tomlkit
import torch

from s4casting.core.checkpoint import Checkpointer, SavedCheckpoint
from s4casting.core.config import Configuration, TransformerConfiguration
from s4casting.core.hooks import CommonHooks
from s4casting.data.files.loader import FileAccess
from s4casting.inference.onnx_export import build_model, load_checkpoint_weights
from s4casting.inference.weights import (
    checkpoint_to_safetensors,
    read_safetensors,
    sha256_file,
    verify_checksums,
    write_checksums,
)
from scripts.package_weights import main as package_main
from tests.utils import load_config


def _without_none(value):
    """Drop None values so a dumped configuration is valid TOML.

    Args:
        value: Nested dict or list from ``model_dump``.

    Returns:
        The same structure without None entries.
    """
    if isinstance(value, dict):
        return {k: _without_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_without_none(v) for v in value]
    return value


@pytest.fixture()
def small_config() -> Configuration:
    """Build a small transformer configuration that converts and exports quickly.

    Returns:
        Configuration: A transformer configuration with a GMM head.
    """
    cfg = load_config()
    cfg.model.model = "transformer"
    cfg.model.transformer = TransformerConfiguration(latent_dim=32, n_heads=2, n_layers=1, mlp_layers=2)
    cfg.model.loss.loss = "nll"
    cfg.model.output_head.arch = "gmm"
    cfg.model.output_head.n_gaussians = 2
    return cfg


@pytest.fixture()
def checkpoint(small_config: Configuration, tmp_path: pathlib.Path) -> tuple[pathlib.Path, dict[str, torch.Tensor]]:
    """Write a training checkpoint container for a randomly initialised small model.

    Returns:
        tuple[pathlib.Path, dict[str, torch.Tensor]]: The checkpoint path and the weights it holds.
    """
    torch.manual_seed(0)
    model = build_model(small_config)
    # DDP-style names, to check the prefix is stripped on conversion.
    state_dict = {f"module.{key}": value for key, value in model.state_dict().items()}
    model_buffer, optimizer_buffer = io.BytesIO(), io.BytesIO()
    torch.save(state_dict, model_buffer)
    torch.save({}, optimizer_buffer)
    saved = SavedCheckpoint(
        torch_model=model_buffer.getvalue(),
        torch_optimizer=optimizer_buffer.getvalue(),
        iteration=1234,
        eval_metrics={"crps": 0.5},
        benchmark_metrics={},
        loss=-0.25,
    )
    path = tmp_path / "checkpoint_1234.pt"
    FileAccess(str(path)).save_pydantic(saved)
    return path, model.state_dict()


def test_checkpoint_converts_to_safetensors_losslessly(checkpoint, tmp_path: pathlib.Path) -> None:
    """Conversion keeps every tensor bit-exact, drops the DDP prefix and embeds provenance."""
    path, expected = checkpoint
    out = tmp_path / "model.safetensors"
    metadata = checkpoint_to_safetensors(str(path), out, {"s4casting.code_tag": "v9.9.9"})

    state_dict, read_back = read_safetensors(str(out))
    assert set(state_dict) == set(expected)
    assert all(torch.equal(state_dict[key], expected[key]) for key in expected)
    assert metadata["s4casting.checkpoint_iteration"] == "1234"
    assert read_back["s4casting.checkpoint_loss"] == "-0.25"
    assert read_back["s4casting.code_tag"] == "v9.9.9"
    assert read_back["s4casting.n_parameters"] == str(sum(v.numel() for v in expected.values()))
    assert read_back["s4casting.weights_sha256"] == sha256_file(out)


def test_loader_accepts_both_formats(small_config: Configuration, checkpoint, tmp_path: pathlib.Path) -> None:
    """The export loader reads a checkpoint container and a safetensors file identically."""
    path, expected = checkpoint
    safe = tmp_path / "model.safetensors"
    checkpoint_to_safetensors(str(path), safe)

    from_pt = build_model(small_config)
    meta_pt = load_checkpoint_weights(from_pt, str(path))
    from_safe = build_model(small_config)
    meta_safe = load_checkpoint_weights(from_safe, str(safe))

    for key, value in expected.items():
        assert torch.equal(from_pt.state_dict()[key], value)
        assert torch.equal(from_safe.state_dict()[key], value)
    assert meta_pt["s4casting.checkpoint_iteration"] == meta_safe["s4casting.checkpoint_iteration"] == "1234"
    assert meta_safe["s4casting.weights_sha256"] == sha256_file(safe)


def test_checksums_detect_tampering(tmp_path: pathlib.Path) -> None:
    """A checksum file verifies untouched files and flags a modified one."""
    good = tmp_path / "a.bin"
    good.write_bytes(b"weights")
    other = tmp_path / "b.bin"
    other.write_bytes(b"onnx")
    checksums = write_checksums([good, other], tmp_path / "checksums.sha256")

    assert verify_checksums(checksums) == {"a.bin": True, "b.bin": True}
    other.write_bytes(b"onnx, but different")
    assert verify_checksums(checksums) == {"a.bin": True, "b.bin": False}


def test_package_builds_a_complete_verifiable_release(
    small_config: Configuration, checkpoint, tmp_path: pathlib.Path
) -> None:
    """The packaging script produces every release asset, all covered by matching checksums."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    path, _ = checkpoint
    config_path = tmp_path / "config.toml"
    config_path.write_text(tomlkit.dumps(_without_none(small_config.model_dump(mode="json"))), encoding="utf-8")
    card = tmp_path / "MODEL_CARD.md"
    card.write_text("# tiny\n", encoding="utf-8")

    folder = package_main([
        "--config-path",
        str(config_path),
        "--checkpoint",
        str(path),
        "--name",
        "tiny",
        "--out-dir",
        str(tmp_path / "release"),
        "--model-card",
        str(card),
        "--code-tag",
        "v0.0.0-test",
        "--allow-dirty",
        "--no-verify",
    ])

    names = {p.name for p in folder.iterdir()}
    assert "LICENSE" in names
    assert {
        "tiny.safetensors",
        "tiny.onnx",
        "training_config.toml",
        "MODEL_CARD.md",
        "README.md",
        "manifest.json",
        "checksums.sha256",
    } <= names
    assert all(verify_checksums(folder / "checksums.sha256").values())
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["code"]["tag"] == "v0.0.0-test"
    assert manifest["checkpoint"]["iteration"] == "1234"
    assert manifest["files"]["tiny.safetensors"]["sha256"] == sha256_file(folder / "tiny.safetensors")
    _, metadata = read_safetensors(str(folder / "tiny.safetensors"))
    assert metadata["s4casting.code_tag"] == "v0.0.0-test"


def test_checkpointer_warm_starts_from_safetensors(
    small_config: Configuration, checkpoint, tmp_path: pathlib.Path
) -> None:
    """A safetensors path in the load slot initialises the model and leaves the optimizer alone."""
    path, expected = checkpoint
    safe = tmp_path / "released.safetensors"
    checkpoint_to_safetensors(str(path), safe, {"s4casting.code_tag": "v0.0.0-test"})

    model = build_model(small_config)
    optimizer = Mock()
    context = SimpleNamespace(
        model_container=SimpleNamespace(model=model),
        machine=SimpleNamespace(torch_device="cpu", ddp=False),
        optimizer=optimizer,
    )
    Checkpointer(CommonHooks(), load=str(safe)).load(context)  # type: ignore[arg-type]

    for key, value in expected.items():
        assert torch.equal(model.state_dict()[key], value)
    optimizer.load_state_dict.assert_not_called()


def test_conversion_is_byte_reproducible(checkpoint, tmp_path: pathlib.Path) -> None:
    """Converting the same checkpoint with the same provenance twice yields identical bytes."""
    path, _ = checkpoint
    provenance = {"s4casting.code_tag": "v1", "s4casting.code_commit": "abc", "s4casting.model_name": "tiny"}
    checkpoint_to_safetensors(str(path), tmp_path / "one.safetensors", provenance)
    checkpoint_to_safetensors(str(path), tmp_path / "two.safetensors", provenance)
    assert sha256_file(tmp_path / "one.safetensors") == sha256_file(tmp_path / "two.safetensors")
