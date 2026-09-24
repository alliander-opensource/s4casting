# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0
"""Model weight files: safetensors conversion, format-aware loading and integrity checks.

Training writes a pickled checkpoint container (``.pt``) holding the model and optimizer
state. Pickle can execute code on load, so published weights are distributed as
safetensors instead: a flat, code-free tensor format with a small string-only metadata
header. This module converts between the two, loads either into a model, and produces
the SHA-256 checksums that accompany a weights release.
"""

import hashlib
import io
import json
import pathlib

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from s4casting.data.files.loader import FileAccess

SAFETENSORS_SUFFIX = ".safetensors"
CHECKSUMS_FILENAME = "checksums.sha256"
MODULE_PREFIX = "module."
# All provenance goes under one header key as canonical JSON. safetensors serialises the
# metadata through a hash map whose key order changes per process, so several keys would
# make the same weights hash differently on every run; one key keeps the file reproducible.
METADATA_KEY = "s4casting"


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove the ``module.`` prefix DistributedDataParallel adds to parameter names.

    Args:
        state_dict (dict[str, torch.Tensor]): State dict as saved by training.

    Returns:
        dict[str, torch.Tensor]: The same tensors under plain module names.
    """
    return {key.removeprefix(MODULE_PREFIX): value for key, value in state_dict.items()}


def sha256_file(path: str | pathlib.Path, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA-256 digest of a file without reading it into memory at once.

    Args:
        path (str | pathlib.Path): File to hash.
        chunk_size (int): Bytes read per iteration.

    Returns:
        str: Lower-case hex digest.
    """
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_checkpoint(checkpoint_path: str) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Read model weights and provenance from a training checkpoint container.

    Only the model weights are read; the optimizer state the container also carries is
    ignored. The container itself is pickle, so only open checkpoints you trust.

    Args:
        checkpoint_path (str): Local or remote path to the ``.pt`` container.

    Returns:
        tuple[dict[str, torch.Tensor], dict[str, str]]: Weights and provenance metadata.
    """
    checkpoint = FileAccess(checkpoint_path).load_pydantic()
    # weights_only restricts unpickling of the inner archive to tensors and primitive containers.
    state_dict = torch.load(io.BytesIO(checkpoint["torch_model"]), map_location="cpu", weights_only=True)
    metadata = {
        "s4casting.checkpoint": checkpoint_path,
        "s4casting.checkpoint_iteration": str(checkpoint["iteration"]),
        "s4casting.checkpoint_loss": str(checkpoint["loss"]),
    }
    return strip_module_prefix(state_dict), metadata


def read_safetensors(path: str) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Read model weights and their embedded metadata from a safetensors file.

    Args:
        path (str): Local or remote path to the ``.safetensors`` file.

    Returns:
        tuple[dict[str, torch.Tensor], dict[str, str]]: Weights and the file's metadata,
            extended with the file's SHA-256 under ``s4casting.weights_sha256``.
    """
    local = FileAccess(path).as_local_path()
    with safe_open(str(local), framework="pt", device="cpu") as handle:
        metadata = unpack_metadata(handle.metadata() or {})
    state_dict = load_file(str(local), device="cpu")
    metadata.setdefault("s4casting.checkpoint", path)
    metadata["s4casting.weights_sha256"] = sha256_file(local)
    return strip_module_prefix(state_dict), metadata


def read_weights(path: str) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Read model weights from either a safetensors file or a training checkpoint.

    Args:
        path (str): Path ending in ``.safetensors`` for the safe format, anything else is
            treated as a training checkpoint container.

    Returns:
        tuple[dict[str, torch.Tensor], dict[str, str]]: Weights and provenance metadata.
    """
    if path.endswith(SAFETENSORS_SUFFIX):
        return read_safetensors(path)
    return read_checkpoint(path)


def write_safetensors(
    state_dict: dict[str, torch.Tensor], path: str | pathlib.Path, metadata: dict[str, str] | None = None
) -> pathlib.Path:
    """Write a state dict as a safetensors file.

    Tensors are copied to contiguous CPU memory first: safetensors refuses views and
    tensors sharing storage, and the copies are the only representation that is
    reproducible byte for byte.

    Args:
        state_dict (dict[str, torch.Tensor]): Weights to write.
        path (str | pathlib.Path): Destination ``.safetensors`` path.
        metadata (dict[str, str] | None): String-only header metadata.

    Returns:
        pathlib.Path: The path written.
    """
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tensors = {key: value.detach().cpu().contiguous().clone() for key, value in state_dict.items()}
    save_file(tensors, str(destination), metadata=pack_metadata(metadata or {}))
    return destination


def pack_metadata(metadata: dict[str, str]) -> dict[str, str]:
    """Serialise provenance into the single, canonically ordered header entry.

    Args:
        metadata (dict[str, str]): Provenance as flat key/value pairs.

    Returns:
        dict[str, str]: A one-entry mapping safetensors can store reproducibly.
    """
    return {METADATA_KEY: json.dumps({k: str(v) for k, v in sorted(metadata.items())}, sort_keys=True)}


def unpack_metadata(header: dict[str, str]) -> dict[str, str]:
    """Read provenance back from a safetensors header.

    Accepts both the single-entry form written here and plain flat keys.

    Args:
        header (dict[str, str]): The file's metadata as safetensors returns it.

    Returns:
        dict[str, str]: Flat provenance key/value pairs.
    """
    packed = header.get(METADATA_KEY)
    if packed is None:
        return dict(header)
    metadata = {k: v for k, v in header.items() if k != METADATA_KEY}
    metadata.update(json.loads(packed))
    return metadata


def checkpoint_to_safetensors(
    checkpoint_path: str, output_path: str | pathlib.Path, extra_metadata: dict[str, str] | None = None
) -> dict[str, str]:
    """Convert a training checkpoint into a safetensors weights file.

    Args:
        checkpoint_path (str): The ``.pt`` container written by the Checkpointer.
        output_path (str | pathlib.Path): Destination ``.safetensors`` path.
        extra_metadata (dict[str, str] | None): Additional provenance to embed, for
            example the code tag and commit the weights are released with.

    Returns:
        dict[str, str]: The metadata embedded in the written file.
    """
    state_dict, metadata = read_checkpoint(checkpoint_path)
    metadata.update(extra_metadata or {})
    metadata["s4casting.n_parameters"] = str(sum(value.numel() for value in state_dict.values()))
    write_safetensors(state_dict, output_path, metadata)
    return metadata


def write_checksums(paths: list[pathlib.Path], output_path: pathlib.Path) -> pathlib.Path:
    """Write SHA-256 checksums in the format ``sha256sum -c`` and ``shasum -a 256 -c`` read.

    Args:
        paths (list[pathlib.Path]): Files to hash; recorded by base name, so they must sit
            next to the checksum file.
        output_path (pathlib.Path): Destination, conventionally ``checksums.sha256``.

    Returns:
        pathlib.Path: The path written.
    """
    lines = [f"{sha256_file(path)}  {path.name}" for path in sorted(paths, key=lambda p: p.name)]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def verify_checksums(checksum_path: pathlib.Path) -> dict[str, bool]:
    """Check every file listed in a checksum file against its recorded digest.

    Args:
        checksum_path (pathlib.Path): A file written by :func:`write_checksums`.

    Returns:
        dict[str, bool]: Whether each listed file exists and matches.
    """
    results: dict[str, bool] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        target = checksum_path.parent / name
        results[name] = target.is_file() and sha256_file(target) == digest
    return results
