<!--
SPDX-FileCopyrightText: Contributors to the s4casting project

SPDX-License-Identifier: MPL-2.0
-->

# ONNX EXPORT

Export a trained model to a single `.onnx` file that runs without PyTorch, this
repository, or the training data. Useful for handing a forecaster to a serving stack, a
colleague, or a runtime in another language.

## Installation

ONNX is an optional dependency group:

```bash
uv sync --extra onnx
```

This adds `onnx`, `onnxruntime` and `onnxscript`. Only `onnxscript` is needed to export;
`onnxruntime` is used to verify the result and to run it afterwards.

## Quick start

You do **not** need a checkpoint to start. The graph structure is fully determined by the
TOML config, so you can export with randomly initialised weights and confirm the whole
pipeline works before a trained model exists:

```bash
# Validate the export with random weights
uv run python3 scripts/export_onnx.py --config-path configs/transformer.toml --output out/model.onnx

# The same command once weights are available
uv run python3 scripts/export_onnx.py --config-path configs/transformer.toml \
    --checkpoint out/checkpoint_750000.pt --output out/model.onnx
```

A checkpoint changes parameter *values* only. Nothing about the graph changes, so an
export that works with random weights works with trained ones.

Output:

```
[OK] Exported -> out/model.onnx
[OK] batch=1 max abs deviation vs PyTorch: 2.265e-06
[OK] batch=3 max abs deviation vs PyTorch: 2.742e-06
```

## How ONNX works

ONNX stores a **frozen computation graph**, not a Python program. The exporter runs the
model once on example tensors, records the operations that actually executed, and writes
that operation list plus the weights into a protobuf file. A runtime then replays it.

Three consequences shape everything below:

- **Only the traced path exists.** `if y is not None:` is resolved at export time, not
  kept as a branch. The loss half of `forward` is simply absent from the graph.
- **No optional tensor arguments and no Python ints.** The export wraps the model so it
  takes exactly `(x, xm)`, with the sampling intervals pinned to their configured values.
  They only ever fed the loss, so predictions are unaffected.
- **Shapes become constants unless declared dynamic.** See
  [Fixed and dynamic shapes](#fixed-and-dynamic-shapes).

## What you get

For `configs/transformer.toml` (32-day context at 15-minute resolution, 21 input
features, GMM head with 4 components):

| Property | Value |
| --- | --- |
| Inputs | `x`, `xm`, both `float32` of shape `[batch, 3072, 21]` |
| Output | `prediction`, shape `[batch, 3072, 1, 4, 3]` |
| File | one self-contained `.onnx`, 125 MB for a 29.4M-parameter model |
| Export time | roughly 25 seconds on CPU |

`x` is the input series and `xm` its mask. The mask marks which values are real: zero the
target channel across the forecast horizon to mark the steps you want predicted. This is
the same convention `DataFrameInferenceRunner` uses.

### Reading the output

The output is the raw output-head tensor, and its meaning depends on the configured head:

- **GMM head**: `[batch, time, n_out_features, n_gaussians, 3]`, where the last axis is
  `(logpi, sigma, mu)`.
- **Quantile head**: `[batch, time, n_out_features, n_quantiles]`.

For a GMM head, `--quantiles` adds a **second output** so the graph emits both the raw
mixture parameters and ready-made quantiles. See
[Emitting quantiles directly](#emitting-quantiles-directly).

Without that flag the graph emits mixture parameters only, and you convert them yourself
with `s4casting.core.distributions.gmm_to_quantiles`.

Take only the forecast horizon from the output. The model returns a prediction for every
time step, but only the last `horizon_length` steps are the forecast.

### Embedded metadata and provenance

The artefact is self-describing. It carries the **entire configuration** it was exported
with, plus the checkpoint it came from, so how a model was built and trained stays
recoverable without this repository, the original TOML, or the checkpoint:

```python
from s4casting.inference.onnx_export import read_metadata

meta = read_metadata("out/model.onnx")

meta["output_head"]                             # 'gmm'
meta["horizon_length"]                           # '192'
meta["checkpoint_iteration"]                     # '750000'
meta["checkpoint_loss"]                          # '0.2317'
meta["config"]["model"]["transformer"]           # {'latent_dim': 512, 'n_heads': 8, ...}
meta["config"]["optimizer"]["learning_rate"]     # 3e-05
meta["config"]["run"]["seed"]                    # 42069
meta["outputs"]                                  # 'prediction,quantiles'
```

`config` is the full `Configuration` as a nested dict, about 3.5 KB, covering `model`,
`io`, `training`, `optimizer`, `scheduler`, `run`, `benchmarking`, `validation`,
`machine` and `metrics`. The checkpoint keys appear only when `--checkpoint` was used.

The flat convenience keys are also there for consumers that would rather not walk the
config: `model`, `output_head`, `n_gaussians`, `quantile_values`, `n_out_features`,
`n_input_features`, `feature_order`, `patch_size`, `sequence_length`, `horizon_length`,
`base_sample_interval_minutes`.

Without this repository, read them with `onnx` directly:

```python
import json, onnx

props = {p.key: p.value for p in onnx.load("out/model.onnx").metadata_props}
config = json.loads(props["s4casting.config"])
```

> **The `authentication` section is excluded.** Pydantic already masks its `SecretStr`
> fields, but an exported model is meant to be handed to other people, so the section is
> dropped outright rather than relying on masking. A test asserts no credential reaches
> the file.

## Running the exported model

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("out/model.onnx", providers=["CPUExecutionProvider"])

x = ...                                  # float32 [batch, 3072, 8]
xm = np.ones_like(x)
xm[:, -192:, 0] = 0                      # mark the forecast horizon on the target channel

prediction = session.run(None, {"x": x, "xm": xm})[0]
logpi, sigma, mu = prediction[:, -192:, 0].transpose(3, 0, 1, 2)
```

No PyTorch and no s4casting import required.

## Options

| Flag | Effect |
| --- | --- |
| `--checkpoint` | Load trained weights. Omit to export random weights. |
| `--output` | Destination path, default `out/model.onnx`. |
| `--quantiles` | For a GMM head, add a second `quantiles` output. |
| `--dynamic-time` | Keep the time axis symbolic. See below. |
| `--external-data` | Write weights to a sidecar `.onnx.data` file instead of one file. |
| `--opset` | ONNX opset to target, default 18. |
| `--no-verify` | Skip the comparison against PyTorch. Not recommended. |

### Emitting quantiles directly

`--quantiles` gives a GMM model a second output, so a consumer needs no post-processing
and no s4casting code:

```bash
uv run python3 scripts/export_onnx.py --config-path configs/transformer.toml \
    --checkpoint out/checkpoint_750000.pt --output out/model.onnx --quantiles
```

```
outputs: prediction  ['batch', 3072, 1, 4, 3]     # logpi, sigma, mu
         quantiles   ['batch', 3072, 1, 11]       # the configured quantile_values
```

The quantiles follow `model.output_head.quantile_values` from the config, in that order,
and the list is recorded in `s4casting.quantile_values`.

`gmm_to_quantiles` itself cannot be traced: it calls `.item()`, builds
`torch.distributions` objects and loops in Python. The graph instead inverts the mixture
CDF directly. A normal CDF is an `erf`, which ONNX supports, so the mixture CDF is a
weighted sum of `erf` terms, and inverting it is a bisection with a fixed number of steps
— static control flow that traces cleanly. All quantiles are solved simultaneously rather
than in a loop, to keep the graph small.

This is **more** accurate than `gmm_to_quantiles`, which snaps its answer to a 1000-point
grid. Evaluating the mixture CDF at the values the graph returns reproduces the requested
quantiles to within `2.4e-07`. The bisection converges at 25 steps in float32; the default
is 30.

It is not free. On a small model the quantile layer added 384 nodes and took inference
from 2.5 ms to 38 ms, because it evaluates `erf` over
`batch x time x features x quantiles x components` at every step. The relative cost falls
as the model grows, but leave the flag off unless a consumer actually wants quantiles in
the graph. Computing them outside is cheaper and just as correct.

The flag is ignored with a warning for a quantile head, whose output already is quantiles.

### Fixed and dynamic shapes

The batch axis is **always** dynamic, so one artefact serves any batch size.

The time axis is fixed by default: a file exported from `configs/transformer.toml`
accepts exactly 3072 time steps and rejects anything else. `--dynamic-time` makes it
symbolic, declared as a multiple of the patch size:

```
without:  x  ['batch', 3072, 8]
with:     x  ['batch', '8*patches', 8]
```

The multiple-of-8 constraint is not cosmetic. The patch encoder reshapes the series into
whole patches and cannot handle a remainder, so the constraint is written into the graph
signature and the runtime rejects a bad length up front rather than producing garbage.

Prefer fixed shapes unless you genuinely serve varying context lengths: the runtime can
preallocate buffers and specialise kernels, and a frozen shape is one less thing that can
go wrong in production.

Verified across 528, 1056, 2112 and 4088 steps from a single artefact; a length that is
not a whole number of patches is rejected by the runtime rather than silently reshaped.

### One file or two

By default the weights are embedded, giving a single file that is easy to ship. Protobuf
caps a single file at 2 GB, which the 29.4M-parameter transformer is comfortably under at
125 MB. A model roughly fifteen times larger would need `--external-data`. That writes a sidecar `.onnx.data`
next to the `.onnx`; **both files must travel together** or the model will not load.

## Supported architectures

Every row below was checked by attempting the export, not assumed.

| Config | Status |
| --- | --- |
| `model = "transformer"` | Supported |
| `model = "ssm"`, `kernel = "s6"` | Supported, with a caveat (below) |
| `model = "ssm"`, `kernel = "s4"` | Not supported |
| `model = "ssm"`, `kernel = "gru"` | Not supported |
| `model = "chronos"` | Not supported |

Unsupported configurations are rejected before tracing starts, with the reason:

```
ONNX export is not implemented for model='ssm' with kernel='s4': SSMKernelDPLR.forward
branches on self.l_kernel.item() and grows its kernel in a while loop, so tracing fails
with GuardOnDataDependentSymNode. The kernel setup has to move out of the traced forward
pass first. Use kernel='s6', or model='transformer'.
```

Notes on the individual cases:

- **`s4`** traces into data-dependent control flow: `SSMKernelDPLR.forward` branches on
  `self.l_kernel.item()` and grows the kernel in a `while` loop. FFT is *not* the
  obstacle here; `torch.fft.rfft`/`irfft` export cleanly at opset 18. Making `s4` work
  means hoisting the kernel setup out of `forward`, after which the kernel is sized for
  one length and `--dynamic-time` would not be available for it.
- **`gru`** is broken independently of ONNX: `GruBlock.forward` does not accept the
  `rate` argument `SequenceResidualBlock` passes it, so it raises `TypeError` in plain
  PyTorch too.
- **`s6`** exports and verifies, but the export always builds on CPU, which selects the
  naive implementation in `s4casting.model.mambacpu`, while GPU training runs the CUDA
  kernels in `s4casting.model.mamba`. The two share parameter names and shapes, so a
  checkpoint loads either way, but they have **not** been compared numerically. A
  `UserWarning` says so at export time. Check an s6 artefact against the PyTorch model on
  GPU before relying on it.

## Verification

Every export is checked against the PyTorch model it came from unless you pass
`--no-verify`. Both are run on identical inputs at several batch sizes and the maximum
absolute deviation is reported; anything above `1e-4` fails the export.

This is not ceremony. An export can produce a loadable, plausible-looking file that is
silently wrong, because tracing can bake a shape in as a constant or drop a branch. Two
real defects were caught this way during development, including a graph that declared a
dynamic batch axis but had frozen it at 1.

Expect deviations around `1e-6`; that is ordinary float32 reordering, not an error.

## Gotchas

**The config must match the checkpoint.** The exporter reads architecture and feature
counts from the TOML. Export with a different config than the model was trained with and
you get a model that is wrong rather than one that errors. In particular, commenting a
feature block such as `[io.features.weather]` out of a config changes the input width.

**Tracing with a batch of 1 freezes the batch axis.** `torch.export` treats any axis whose
example length is 0 or 1 as a constant, so a batch-1 export yields a graph that *claims*
to be dynamic and then fails at batch 5. The exporter always traces with batch 2 for this
reason; keep it that way if you touch `example_inputs`.

**Data files are not required.** The top-level config validator opens every dataset under
`[io.features]`, which is usually absent on a machine that only exports. The exporter
falls back to validating just the sections that determine the graph.

## Code layout

| Path | Contents |
| --- | --- |
| `src/s4casting/inference/onnx_export.py` | `export_onnx`, `verify_onnx`, `InferenceWrapper`, config and checkpoint loading |
| `scripts/export_onnx.py` | Command line entry point |
| `tests/test_onnx_export.py` | Numerical equivalence, dynamic batch axis, metadata, architecture guards |

To export from Python instead of the command line:

```python
from s4casting.inference.onnx_export import export_onnx

path, deviations = export_onnx(
    config="configs/transformer.toml",
    output_path="out/model.onnx",
    checkpoint_path="out/checkpoint_750000.pt",
)
```
