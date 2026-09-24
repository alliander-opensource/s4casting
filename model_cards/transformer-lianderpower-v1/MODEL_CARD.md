---
# Hugging Face model card metadata. This file is the single source of truth for the
# card: edit it here, in the s4casting repository, never on the Hub. The packaging
# script uploads it unchanged as the model repository's README.md.
language: en
license: mpl-2.0
library_name: s4casting
pipeline_tag: time-series-forecasting
tags:
  - time-series
  - load-forecasting
  - power-grid
  - probabilistic-forecasting
  - transformer
  - onnx
  - safetensors
datasets:
  - LianderPower
model-index:
  - name: transformer-lianderpower-v1
    results:
      - task:
          type: time-series-forecasting
        dataset:
          name: LianderPower (validation split, 2024 onwards)
          type: LianderPower
        metrics:
          - type: crps
            value: 0.311
            name: CRPS (pipeline scale, 11 quantiles)
          - type: nll
            value: -0.624
            name: Negative log-likelihood
---

<!--
SPDX-FileCopyrightText: Contributors to the s4casting project

SPDX-License-Identifier: MPL-2.0
-->

# transformer-lianderpower-v1

A probabilistic short-term load forecasting model for medium-voltage power grids,
trained with the [s4casting](https://github.com/alliander-opensource/s4casting)
open-source toolkit on the open [LianderPower](https://www.liander.nl/over-ons/open-data/)
dataset. It forecasts up to two days ahead at 15-minute resolution and returns a
Gaussian mixture per time step, from which quantiles are derived.

| | |
|---|---|
| Model family | s4casting `transformer` |
| Parameters | 29,445,152 (float32) |
| Checkpoint | training step 440,000 of a 750,000-step schedule, run started 2026-09-22 |
| Code release | s4casting tag `v0.1.0` <!-- TODO: set once the tag exists -->, commit `TBD` |
| Training configuration | [`training_config.toml`](training_config.toml), recovered from the ONNX metadata and validated against the code release |
| Weights | `transformer-lianderpower-v1.safetensors` (model weights only, no optimizer state) |
| ONNX | `transformer-lianderpower-v1.onnx`, exported from the safetensors file by the code release, opset 18 |
| Integrity | SHA-256 of every file in `checksums.sha256`, verify with `shasum -a 256 -c checksums.sha256` |
| Licence | Weights and code MPL-2.0 (`LICENSE` in the release), dataset CC-BY-4.0 |

## Intended use

Short-term (up to 2 days ahead) forecasting of medium-voltage power, for research
and grid capacity management. Typical uses are day-ahead and intraday load
forecasts for substations and feeders, studies of forecast uncertainty using the
predicted quantiles, and as a reference model for benchmarking other forecasters
on the LianderPower dataset.

## Non-intended use

- Safety-critical real-time grid control without human oversight.
- Forecasting of individual customer consumption. The model was trained on
  aggregated grid measurements only and has never seen customer-level data.
- Use on grids or horizons substantially different from the training data without
  validation, for example other countries, low-voltage networks, or horizons
  beyond two days.
- Any use that treats a single quantile as a guarantee. The outputs are
  probabilistic and should be consumed as such.

## Training data description

**Dataset.** LianderPower, an anonymised power-grid time-series dataset derived
from operational SCADA telemetry of Liander, a Dutch distribution system operator,
published under CC-BY-4.0. The public release contains aggregate measurement time
series and Open-Meteo weather covariates. It contains no exact asset coordinates,
topology, switching states, feeder identifiers, customer information or raw
single-sensor measurements.

**Measurements.** 1,459 measurement locations, September 2013 to December 2024,
about 9,000 location-years of 5-minute data, resampled to 15 minutes by the
training pipeline. Series are labelled as native or pseudo aggregates and carry
either power or current as the sensor unit.

**Weather covariates.** Open-Meteo variables matched to each measurement location
by nearest neighbour over 202 weather grid points. The model was trained on 17
variables: temperature, relative humidity, dew point, precipitation, rain, snow
depth, sea-level and surface pressure, cloud cover, reference evapotranspiration,
vapour pressure deficit, wind speed at 100 m, and shortwave, direct, diffuse,
direct-normal and terrestrial radiation. The public LianderPower release ships 4 of
these (temperature, wind speed at 100 m, shortwave radiation and direct-normal
irradiance); see Known limitations.

**Time and location features.** Three additional input channels: the timestamp,
and the latitude and longitude of the measurement location.

**Split.** Time-based: validation samples are drawn from 2024 onwards
(`validation.split_type = "time"`, `start_year = 2024`, `percentage = 5`).

**Preprocessing.** Each training window is normalised per sample inside the model
(`norm_clamp = 10`, `norm_eps = 1e-4`); no global scaling is applied to the data.
Covariate dropout was enabled during training so the model tolerates missing
weather inputs.

## Architecture overview

The s4casting toolkit implements S4 and selective state-space (S6, Mamba-style)
models and Transformer variants, see `src/s4casting/model/`. This release is the
Transformer variant with the following configuration:

- **Input.** 21 channels at 15-minute resolution: the target series, 17 weather
  covariates and 3 time/location features. A window covers 32 days
  (3,072 steps), of which the last 2 days (192 steps) are the forecast horizon.
  Each channel carries a mask so missing values are handled explicitly.
- **Patch encoder.** Linear, 4 layers, patch size 8 (2 hours), giving 384 tokens
  per channel.
- **Backbone.** 6 Transformer blocks, latent width 512, 8 heads, attention bias
  enabled, no dropout. Each block applies temporal attention over the tokens of a
  channel and group attention across channels at the same time step, followed by a
  2-layer MLP with a 4x hidden width. Temporal attention is bidirectional
  (`transformer.causal = false`).
- **Patch decoder.** Linear, 8 layers, back to 15-minute resolution.
- **Output head.** Gaussian mixture with 4 components per time step, emitting
  log-weights, standard deviations and means. Quantiles at the levels 0.01, 0.05,
  0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95 and 0.99 are derived from the mixture; the
  ONNX export returns both the mixture parameters and these quantiles.
- **Training objective.** Negative log-likelihood of the mixture on masked targets
  (`alpha_clip = 0.1`), Adam with learning rate 3e-5, betas 0.9/0.99, no weight
  decay, gradient clipping at 1, reduce-on-plateau schedule (factor 0.3,
  patience 2, minimum learning rate 5e-5), batch size 8, seed 42069.

## Evaluation results

Metrics logged by the training pipeline for this checkpoint on the validation
split. CRPS is computed with `scoringrules.crps_quantile` from the 11 derived
quantiles on the pipeline's input scale, averaged over validation samples; it is
not in physical units.

| Metric | Value | Where |
|---|---|---|
| CRPS | 0.311 | validation split, step 440,000 |
| Negative log-likelihood | -0.624 | validation split, step 440,000 |
| Negative log-likelihood | -0.736 | training batch, step 440,000 |

Benchmark results against operational baselines were not recorded in this
checkpoint and are not reported here.

**Implementation parity.** The open-source implementation reproduces the reference
export of this model: on identical inputs, the mixture parameters and quantiles
from the s4casting PyTorch model loaded with these weights agree with the reference
ONNX to within 5e-6, and the ONNX shipped with this release agrees with the
PyTorch model to within 7e-6 at batch sizes 1 and 3.

## Known limitations

- Performance may degrade under distribution shift, for example new connections,
  grid reconfiguration, extreme weather, or rapid growth of solar generation and
  electrification beyond what the 2013 to 2024 history contains.
- **Weather inputs.** The model was trained on 17 weather variables. Running it
  with only the 4 variables in the public LianderPower release is outside its
  training distribution; the loaders accept it because the architecture is
  channel-count independent, but forecast quality in that setting has not been
  evaluated. Weather forecast error is not modelled either: if forecasts rather
  than observations are supplied for the horizon, their error propagates into the
  load forecast.
- **Fixed geometry.** Inputs must be 15-minute series, a full 32-day window is
  expected, and the horizon is 2 days. Other resolutions or horizons require
  re-training.
- **Location features.** Latitude and longitude are inputs, so the model can learn
  location-specific behaviour. Forecasts for locations far from the training area
  rely on extrapolation.
- **Calendar effects.** The only calendar input is the raw timestamp; holidays and
  other special days are not encoded explicitly.

## Bias and fairness considerations

The model predicts aggregated grid loads and does not process data about
individuals. The training data is anonymised and aggregated at substation and
feeder level. Representativeness still matters: all 1,459 locations lie within
Liander's service area in the Netherlands, and the mix of urban and rural areas,
feeder types, industrial versus residential load, and solar penetration reflects
that area in 2013 to 2024. Forecast accuracy may differ systematically between
such groups, which can matter when forecasts feed capacity decisions that affect
which customers get connected first. Users should evaluate accuracy per region and
feeder type before using the forecasts in decisions with distributional effects.

## Responsible AI considerations

- **Human oversight.** The model is a decision-support tool. Operational use should
  keep a person in the loop and fall back to established procedures when forecasts
  are implausible.
- **Uncertainty.** Use the full quantile set rather than the median alone, and
  monitor calibration on live data; the reported CRPS is a single validation
  figure from one period.
- **Monitoring and re-training.** Track accuracy over time and re-train when the
  grid or its usage patterns change.
- **Provenance and integrity.** The weights are distributed as safetensors, which
  cannot execute code on load, together with SHA-256 checksums, the exact training
  configuration and the code release they were built with. Verify the checksums
  before use and load the weights with the referenced code release.
- **Data protection.** No personal data was used for training and the model cannot
  reveal individual consumption.
- **Reporting.** Security issues follow the repository's `SECURITY.md`; other
  problems go through `SUPPORT.md`.

## How to use

```python
from s4casting.inference.onnx_export import build_model, load_configuration, load_checkpoint_weights

config = load_configuration("training_config.toml")
model = build_model(config)
load_checkpoint_weights(model, "transformer-lianderpower-v1.safetensors", device="cpu")
```

For inference without PyTorch, run `transformer-lianderpower-v1.onnx` with
onnxruntime: inputs `x` and `xm` of shape `(batch, 3072, 21)`, outputs
`prediction` of shape `(batch, 3072, 1, 4, 3)` and `quantiles` of shape
`(batch, 3072, 1, 11)`. See `docs/ONNX_EXPORT.md` in the code release.

## Citation

See the `Citation` section of the s4casting repository README.
