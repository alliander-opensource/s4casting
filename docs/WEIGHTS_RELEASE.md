<!--
SPDX-FileCopyrightText: Contributors to the s4casting project

SPDX-License-Identifier: MPL-2.0
-->

# WEIGHTS RELEASE

How trained models are published as open weights, and how a user verifies what they
downloaded. Weights are hosted on the Hugging Face Hub; this repository holds the model
card, the exact training configuration, and the tooling that ties a weights release to a
tagged commit of the code.

## What a release consists of

| File | Content |
|---|---|
| `<name>.safetensors` | Model weights only. No optimizer state, no pickle: safetensors cannot execute code on load. The header records the code tag, commit, checkpoint iteration and loss, and the SHA-256 of the training configuration. |
| `<name>.onnx` | Exported from that safetensors file by the released code, with a second `quantiles` output and a dynamic time axis, and verified against PyTorch before packaging. |
| `training_config.toml` | The exact configuration the model was trained with, validated against the code release. |
| `README.md` | The model card, published unchanged from `model_cards/<name>/MODEL_CARD.md`. A model repository's `README.md` is its model card, so only that copy ships. |
| `LICENSE` | The licence the weights are released under: MPL-2.0, the same as the code. |
| `manifest.json` | Name, code tag and commit, s4casting version, checkpoint provenance, ONNX deviation from PyTorch and the sequence length it was measured at, and the SHA-256 and size of every file. |
| `checksums.sha256` | SHA-256 of every file above, in `sha256sum` format. |

## Where things live

- **Hugging Face Hub**: all files above, one model repository per released model. Each
  upload is a Hub commit, so a release is pinned by its revision hash.
- **This repository**: `model_cards/<name>/MODEL_CARD.md` and
  `model_cards/<name>/training_config.toml`. The card is edited and reviewed here only,
  never on the Hub, and uploaded unchanged. New cards start from
  `model_cards/_template/`, which holds the section structure and the metadata header
  with placeholders.
- **A git tag on the code**: every weights release records the tag and commit it was
  packaged from. The tag is what a user checks out to load the weights with the code that
  produced them.

## Publishing a release

1. Copy `model_cards/_template/` to `model_cards/<name>/`, fill in the card and add
   the exact training configuration as `training_config.toml`, then merge them.
2. Tag the commit the weights are released against, for example:

   ```bash
   git tag -a v0.1.0 -m "Code release paired with <name>"
   git push origin v0.1.0
   ```

3. Package from that tagged, clean checkout. The script refuses to run from an untagged
   or dirty tree unless told otherwise, so the recorded provenance is trustworthy:

   ```bash
   uv run python scripts/package_weights.py \
       --config-path model_cards/<name>/training_config.toml \
       --checkpoint out/checkpoint_<iteration>.pt \
       --name <name> \
       --model-card model_cards/<name>/MODEL_CARD.md \
       --out-dir out/weights
   ```

   Run this from the repository root: the output directory must lie inside the working
   directory. This writes `out/weights/<name>/` with every file
   listed above and prints their checksums. The ONNX graph is exported with a dynamic time
   axis and compared against PyTorch as part of this step, on the Liander2024 geometry
   (3072 steps at 15 minutes, the last 192 forecast) rather than the length it was traced
   at; a deviation above 1e-4 aborts the packaging. `--verify-steps` and `--verify-horizon`
   change that geometry, and `--fixed-time` freezes the time axis instead.

4. Upload. Either add `--push <org>/<name>` to the command above, with a Hub token in the
   environment, or upload the folder by hand. The upload is a Hub commit and its hash
   is the release's Hub revision. It is not written into the model card: the card is
   part of that upload and covered by its checksums, so it cannot contain the hash of
   its own commit. The Hub's commit history holds it; note it in the release notes of
   the code tag if a record in this repository is wanted.

The safetensors file is reproducible: packaging the same checkpoint on the same tag
yields byte-identical weights, so anyone can re-package from the tag and compare hashes.
Provenance is stored under a single `s4casting` header key as canonical JSON, because
safetensors would otherwise write several keys in a process-dependent order. The ONNX file
is not guaranteed byte-identical across exporter versions; its checksum identifies the
published file.

## Verifying a download

```bash
shasum -a 256 -c checksums.sha256          # every line must say OK
```

Then load the weights with the code release named in `manifest.json`:

```bash
git checkout v0.1.0
uv sync
```

```python
from s4casting.inference.onnx_export import build_model, load_configuration, load_checkpoint_weights

config = load_configuration("training_config.toml")
model = build_model(config)
metadata = load_checkpoint_weights(model, "<name>.safetensors", device="cpu")
```

`metadata` carries the provenance from the safetensors header, including the SHA-256 of
the file just loaded, which must equal the one in `checksums.sha256`. The loaders accept
`.safetensors` everywhere a training checkpoint is accepted; the checkpoint container
remains supported for local work but is a pickle and should not be distributed.

## Fine-tuning from released weights

Released weights carry no optimizer state, so a run cannot be resumed as if it never
stopped, but it can be warm-started. Point the training configuration at the safetensors
file and train as usual:

```toml
[io]
load_checkpoint = "<name>.safetensors"
```

The checkpointer loads the weights into the model and leaves the optimizer, scheduler and
iteration counter fresh, and logs the source iteration and code tag from the file's
header. Training checkpoints written by that run use the container format again. If the
data differs from what the released model was trained on, for example the 4 public
weather variables instead of the 17 used in training, treat the result as a new model
and describe it in its own model card.

## Why safetensors and checksums

A training checkpoint is a pickle, and unpickling runs whatever code the file contains.
Safetensors is a flat tensor format with a JSON header: loading it cannot execute code,
and the weights are exactly the tensors listed. The SHA-256 checksums let a user confirm
that what they downloaded is what was published, independently of the hosting platform.
The code tag closes the loop: the configuration schema and model behaviour evolve, and
weights only mean what they meant with the code that trained and exported them.
