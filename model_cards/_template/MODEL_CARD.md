---
# Model card metadata. This file is the single source of truth for the card: edit it
# here, in the S4Casting repository, never on the hosting platform. The packaging
# script publishes it unchanged as the model repository's README.md.
#
# TEMPLATE. Copy this directory to model_cards/<name>/ for a new weights release and
# replace every <PLACEHOLDER>. Guidance for each section is in HTML comments; delete
# them once the section is written. The metadata below follows the Hub's model card
# schema: https://huggingface.co/docs/hub/model-cards#model-card-metadata
language: en
license: mpl-2.0
library_name: s4casting
pipeline_tag: time-series-forecasting
tags:
  - time-series
  - load-forecasting
  - power-grid
  - probabilistic-forecasting
  - <ARCHITECTURE_TAG>  # transformer, s4 or mamba
  - onnx
  - safetensors
datasets:
  - <DATASET_ID>  # one entry per training dataset, Hub dataset ids where they exist
model-index:
  - name: <MODEL_NAME>
    results:
      - task:
          type: time-series-forecasting
        dataset:
          name: <EVALUATION_DATASET_AND_SPLIT>
          type: <EVALUATION_DATASET_ID>
        metrics:
          - type: crps
            value: "<CRPS>"
            name: CRPS (pipeline scale, <N_QUANTILES> quantiles)
          - type: nll
            value: "<NLL>"
            name: Negative log-likelihood
---

<!--
SPDX-FileCopyrightText: Contributors to the s4casting project

SPDX-License-Identifier: MPL-2.0
-->

# <MODEL_NAME>

<!-- One paragraph: what the model forecasts, for which kind of grid, at which
resolution and horizon, what it outputs (mixture parameters, quantiles), and which
datasets it was trained on, with links. Name the model by its own name; S4Casting is
the open-source project and forecasting toolkit the model was developed and trained
with, not the model. Pattern: "A probabilistic ... model ..., developed and trained
using the [S4Casting open-source project and forecasting
toolkit](https://github.com/alliander-opensource/s4casting) on the [<dataset>](<url>)." -->

<SUMMARY_PARAGRAPH>

| | |
|---|---|
| Model family | s4casting `<MODEL_FAMILY>` <!-- transformer, ssm --> |
| Parameters | <N_PARAMETERS> (float32) <!-- manifest.json: checkpoint.n_parameters --> |
| Checkpoint | training step <CHECKPOINT_ITERATION> of a <SCHEDULE_STEPS>-step schedule, run started <RUN_START_DATE> |
| Code release | s4casting tag `<CODE_TAG>`, commit `<CODE_COMMIT>` <!-- manifest.json: code.tag, code.commit --> |
| Training configuration | [`training_config.toml`](training_config.toml), the exact configuration of the run, validated against the code release |
| Weights | `<MODEL_NAME>.safetensors` (model weights only, no optimizer state) |
| ONNX | `<MODEL_NAME>.onnx`, exported from the safetensors file by the code release, opset 18 |
| Integrity | SHA-256 of every file in `checksums.sha256`, verify with `shasum -a 256 -c checksums.sha256` |
| Model weights licence | MPL-2.0 (`LICENSE` in the release) |
| Source code licence | MPL-2.0 (S4Casting repository) |
| Training dataset licence | <DATASET_LICENCES> <!-- one entry per dataset, e.g. LianderPower CC BY 4.0 --> |

## Intended use

<!-- Horizon, resolution, grid level and the concrete decisions the forecasts
support. Name the benchmark role if the model is meant as a reference. -->

<INTENDED_USE>

## Non-intended use

<!-- Keep the items that apply, adapt the horizon and grid level, add any that are
specific to this model's training data. -->

- Safety-critical real-time grid control without human oversight.
- Forecasting of individual customer consumption. The model was trained on
  aggregated grid measurements only and has never seen customer-level data.
- Use on grids or horizons substantially different from the training data without
  validation, for example other countries, other voltage levels, or horizons
  beyond <MAX_HORIZON>.
- Any use that treats a single quantile as a guarantee. The outputs are
  probabilistic and should be consumed as such.

## Training data description

<!-- One subsection per dataset. For each: provenance and licence, what it contains
and what it deliberately does not (coordinates, topology, customer data), the number
of series and the period covered, the native and training resolution. Then the
covariates, the time and location channels, the validation split and the
preprocessing. State explicitly which inputs the public release of each dataset
provides and which it does not, because that gap is a known limitation below. -->

**Datasets.** <DATASETS>

**Measurements.** <MEASUREMENTS>

<!-- Covariates: list every variable the model was trained on and say which of them the
public dataset release ships. Where the public release ships fewer variables than the
model uses (LianderPower publishes 4 of the 17 Open-Meteo variables), say so, name the
source and the retrieval code (src/s4casting/data/preparation/weather.py for
Open-Meteo) and its licence, and note that the full covariate set used in training will
be released alongside the dataset, at the same location as the public release. -->

**Covariates.** <COVARIATES>

<!-- Time and location: say whether the model uses coordinates at all. When it does,
state that the latitude and longitude are those of anonymised locations, never exact
asset coordinates: LianderPower applies spatial anonymisation (random jittering and
aggregation of locations) before publication. -->

**Time and location features.** <TIME_AND_LOCATION_FEATURES>

**Split.** <VALIDATION_SPLIT>

**Preprocessing.** <PREPROCESSING>

## Architecture overview

The s4casting toolkit implements S4 and selective state-space (S6, Mamba-style)
models and Transformer variants, see `src/s4casting/model/`. This release is the
<MODEL_FAMILY> variant with the following configuration:

<!-- Every value here must match training_config.toml. Cover: input channels and
resolution(s), context window(s) and horizon(s), masking; patch encoder; backbone
(blocks, width, heads, attention variants, causality); patch decoder; output head
and the quantile levels; training objective, optimiser, schedule, batch size, seed,
and the pre-training task (prediction, masking, cpm) if it is not plain prediction. -->

- **Input.** <INPUT>
- **Patch encoder.** <PATCH_ENCODER>
- **Backbone.** <BACKBONE>
- **Patch decoder.** <PATCH_DECODER>
- **Output head.** <OUTPUT_HEAD>
- **Training objective.** <TRAINING_OBJECTIVE>

## Evaluation results

<!-- Say exactly where each number comes from (validation split, step, metric
implementation and scale) so a reader can reproduce it. Report benchmark results
against baselines when they exist, and say so when they do not. -->

<EVALUATION_INTRO>

| Metric | Value | Where |
|---|---|---|
| CRPS | <CRPS> | validation split, step <CHECKPOINT_ITERATION> |
| Negative log-likelihood | <NLL> | validation split, step <CHECKPOINT_ITERATION> |

<BENCHMARK_RESULTS>

**Implementation parity.** The ONNX shipped with this release agrees with the
PyTorch model loaded with these weights to within <ONNX_MAX_DEVIATION> on identical
inputs <!-- manifest.json: onnx.max_abs_deviation_from_pytorch -->, compared at
<VERIFIED_SEQUENCE_LENGTH> steps <!-- manifest.json: onnx.verified_sequence_length -->.
The time axis of the ONNX graph is dynamic; parity was not measured at other lengths.

## Known limitations

<!-- Distribution shift, inputs the public datasets do not provide, fixed or
flexible geometry (resolutions, window lengths, horizons), location features and
extrapolation, calendar effects, anything the evaluation did not cover. -->

- <LIMITATION>

## Bias and fairness considerations

<!-- The model predicts aggregated loads, not individuals, but representativeness
matters: which regions, feeder types, load mixes and solar penetration the
training data covers, and what that means when forecasts feed capacity decisions.
Adapt the paragraph below to the datasets actually used. -->

The model predicts aggregated grid loads and does not process data about
individuals. The training data is anonymised and aggregated at substation and
feeder level. Representativeness still matters: <REPRESENTATIVENESS>. Forecast
accuracy may differ systematically between such groups, which can matter when
forecasts feed capacity decisions that affect which customers get connected first.
Users should evaluate accuracy per region and feeder type before using the
forecasts in decisions with distributional effects.

## Responsible AI considerations

- **Human oversight.** The model is a decision-support tool. Operational use should
  keep a person in the loop and fall back to established procedures when forecasts
  are implausible.
- **Uncertainty.** Use the full quantile set rather than the median alone, and
  monitor calibration on live data; the reported metrics are validation figures
  from one period.
- **Monitoring and re-training.** Track accuracy over time and re-train when the
  grid or its usage patterns change.
- **Provenance and integrity.** The weights are distributed as safetensors, which
  cannot execute code on load, together with SHA-256 checksums, the exact training
  configuration and the code release they were built with. Verify the checksums
  before use and load the weights with the referenced code release.
- **Data protection.** The model was trained on anonymised and aggregated grid
  measurements rather than individual customer consumption data. The training
  datasets do not contain customer data or data attributable to individual
  connections. <!-- adapt if a dataset in this release is different -->
- **Reporting.** Security issues should be reported according to the S4Casting
  project's [`SECURITY.md`](https://github.com/alliander-opensource/s4casting/blob/main/docs/SECURITY.md);
  other support requests and issues should follow the project's
  [`SUPPORT.md`](https://github.com/alliander-opensource/s4casting/blob/main/docs/SUPPORT.md).

## How to use

```python
from s4casting.inference.onnx_export import build_model, load_configuration, load_checkpoint_weights

config = load_configuration("training_config.toml")
model = build_model(config)
load_checkpoint_weights(model, "<MODEL_NAME>.safetensors", device="cpu")
```

For inference without PyTorch, run `<MODEL_NAME>.onnx` with onnxruntime: inputs
`x` and `xm` of shape `(batch, <SEQUENCE_LENGTH>, <N_INPUT_CHANNELS>)`, outputs
`prediction` of shape `<PREDICTION_SHAPE>` and `quantiles` of shape
`<QUANTILES_SHAPE>`. See `docs/ONNX_EXPORT.md` in the code release.
<!-- The shapes are in the ONNX metadata (s4casting.sequence_length and friends);
for a model with a dynamic time axis, describe the constraint instead of one length. -->

## Contributing

For contributions to S4Casting, including model improvements and new model releases,
see the S4Casting project's
[`CONTRIBUTING.md`](https://github.com/alliander-opensource/s4casting/blob/main/docs/CONTRIBUTING.md).

## Citation

See the `Citation` section of the s4casting repository README.
