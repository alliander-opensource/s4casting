# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0
"""Package a trained model for an open-weights release.

Turns a training checkpoint into the complete, verifiable asset set of a weights release:

- ``<name>.safetensors``   model weights only, no optimizer state, no pickle
- ``<name>.onnx``          exported from that safetensors file, verified against PyTorch
- ``training_config.toml`` the exact configuration the model was trained with
- ``MODEL_CARD.md`` and ``README.md``  the model card, the latter for the Hugging Face Hub
- ``LICENSE``              the licence the weights are released under (MPL-2.0)
- ``manifest.json``        the code tag and commit, hashes and training provenance
- ``checksums.sha256``     SHA-256 of every file above, for ``shasum -a 256 -c``

Every release is tied to a tagged commit of this repository: the script refuses to
package from an untagged or dirty checkout unless told otherwise, and records the tag and
commit in the safetensors header, the ONNX metadata and the manifest.
"""

import argparse
import datetime
import importlib.metadata
import json
import pathlib
import shutil
import subprocess
import sys
import warnings

from s4casting.inference.onnx_export import export_onnx
from s4casting.inference.weights import (
    CHECKSUMS_FILENAME,
    SAFETENSORS_SUFFIX,
    checkpoint_to_safetensors,
    read_safetensors,
    sha256_file,
    write_checksums,
    write_safetensors,
)

warnings.filterwarnings("ignore")


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command line arguments for the packaging script.

    Args:
        argv: List of command line arguments. If None, uses sys.argv.

    Returns:
        Parsed arguments namespace.
    """
    ap = argparse.ArgumentParser(description="Package a trained s4casting model for an open-weights release.")
    ap.add_argument("--config-path", required=True, help="Exact training configuration TOML of the model.")
    ap.add_argument("--checkpoint", required=True, help="Training checkpoint (.pt) or existing .safetensors file.")
    ap.add_argument("--name", required=True, help="Release name, used for the asset file names.")
    ap.add_argument("--out-dir", default="out/weights", help="Directory that receives <out-dir>/<name>/.")
    ap.add_argument("--model-card", help="MODEL_CARD.md to include; uploaded unchanged as README.md.")
    ap.add_argument("--license-file", default="LICENSE", help="Licence text shipped with the weights.")
    ap.add_argument("--code-tag", help="Code tag the release is built from; defaults to the tag at HEAD.")
    ap.add_argument("--allow-untagged", action="store_true", help="Package although HEAD carries no tag.")
    ap.add_argument("--allow-dirty", action="store_true", help="Package although the checkout has changes.")
    ap.add_argument("--no-verify", action="store_true", help="Skip comparing the ONNX graph against PyTorch.")
    ap.add_argument("--push", metavar="REPO_ID", help="Upload the folder to this Hugging Face model repo.")
    return ap.parse_args(argv)


def git_output(*args: str) -> str:
    """Run a git command and return its stripped output, or an empty string on failure.

    Args:
        *args (str): Arguments passed to git.

    Returns:
        str: Command output.
    """
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def code_provenance(args: argparse.Namespace) -> dict[str, str]:
    """Determine the code tag and commit the release is built from, enforcing cleanliness.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Raises:
        SystemExit: If HEAD is untagged or the tree is dirty and no override was given.

    Returns:
        dict[str, str]: Tag, commit and package version.
    """
    commit = git_output("rev-parse", "HEAD")
    tag = args.code_tag or git_output("describe", "--tags", "--exact-match")
    dirty = bool(git_output("status", "--porcelain", "--untracked-files=no"))
    if not tag and not args.allow_untagged:
        raise SystemExit("HEAD carries no tag: tag the release commit first, or pass --allow-untagged.")
    if dirty and not args.allow_dirty:
        raise SystemExit("The checkout has uncommitted changes: commit them first, or pass --allow-dirty.")
    return {
        "s4casting.code_tag": tag or "untagged",
        "s4casting.code_commit": commit or "unknown",
        "s4casting.code_dirty": str(dirty).lower(),
        "s4casting.version": importlib.metadata.version("s4casting"),
    }


def package(args: argparse.Namespace) -> pathlib.Path:
    """Build the release folder.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        pathlib.Path: The release folder.
    """
    out = pathlib.Path(args.out_dir) / args.name
    out.mkdir(parents=True, exist_ok=True)
    provenance = code_provenance(args)

    config_src = pathlib.Path(args.config_path)
    config_out = out / "training_config.toml"
    shutil.copyfile(config_src, config_out)
    provenance["s4casting.training_config_sha256"] = sha256_file(config_out)
    provenance["s4casting.model_name"] = args.name

    weights_out = out / f"{args.name}{SAFETENSORS_SUFFIX}"
    if args.checkpoint.endswith(SAFETENSORS_SUFFIX):
        state_dict, metadata = read_safetensors(args.checkpoint)
        metadata.pop("s4casting.weights_sha256", None)
        metadata.update(provenance)
        write_safetensors(state_dict, weights_out, metadata)
    else:
        metadata = checkpoint_to_safetensors(args.checkpoint, weights_out, provenance)

    onnx_out, deviations = export_onnx(
        str(config_out),
        str(out / f"{args.name}.onnx"),
        checkpoint_path=str(weights_out),
        with_quantiles=True,
        verify=not args.no_verify,
    )

    if args.model_card:
        shutil.copyfile(args.model_card, out / "MODEL_CARD.md")
        shutil.copyfile(args.model_card, out / "README.md")
    if pathlib.Path(args.license_file).is_file():
        shutil.copyfile(args.license_file, out / "LICENSE")

    assets = [p for p in out.iterdir() if p.is_file() and p.name not in {CHECKSUMS_FILENAME, "manifest.json"}]
    manifest = {
        "name": args.name,
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "code": {
            k.removeprefix("s4casting.code_"): v for k, v in provenance.items() if k.startswith("s4casting.code_")
        },
        "s4casting_version": provenance["s4casting.version"],
        "checkpoint": {
            "source": args.checkpoint,
            "iteration": metadata.get("s4casting.checkpoint_iteration"),
            "loss": metadata.get("s4casting.checkpoint_loss"),
            "n_parameters": metadata.get("s4casting.n_parameters"),
        },
        "onnx": {"file": onnx_out.name, "max_abs_deviation_from_pytorch": deviations},
        "files": {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in sorted(assets)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_checksums([*assets, out / "manifest.json"], out / CHECKSUMS_FILENAME)
    return out


def push(folder: pathlib.Path, repo_id: str, message: str) -> str:
    """Upload the release folder to a Hugging Face model repository.

    Args:
        folder (pathlib.Path): Release folder.
        repo_id (str): Target repository, for example ``org/model-name``.
        message (str): Commit message on the Hub.

    Returns:
        str: URL of the resulting Hub commit.
    """
    from huggingface_hub import HfApi  # noqa: PLC0415

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    info = api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type="model", commit_message=message)
    return str(info.commit_url)


def main(argv=None) -> pathlib.Path:
    """Package the release and report what was written.

    Args:
        argv: List of command line arguments. If None, uses sys.argv.

    Returns:
        pathlib.Path: The release folder.
    """
    args = parse_args(argv)
    folder = package(args)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    print(f"Packaged {args.name} from code {manifest['code']['tag']} ({manifest['code']['commit'][:12]}) into {folder}")  # noqa: T201
    for name, entry in manifest["files"].items():
        print(f"  {entry['sha256']}  {name}  ({entry['bytes'] / 1e6:.1f} MB)")  # noqa: T201
    if args.push:
        url = push(folder, args.push, f"{args.name} built from s4casting {manifest['code']['tag']}")
        print(f"Uploaded to {url}")  # noqa: T201
    return folder


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
