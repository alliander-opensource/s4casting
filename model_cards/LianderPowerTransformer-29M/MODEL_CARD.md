---
# Model card metadata. This file is the single source of truth for the card: edit it
# here, in the S4Casting repository, never on the hosting platform. The packaging
# script publishes it unchanged as the model repository's README.md.
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
  - name: LianderPowerTransformer-29M
    results:
      - task:
          type: time-series-forecasting
        dataset:
          name: LianderPower (validation split, 2024 onwards)
          type: LianderPower
        metrics:
          - type: crps
            value: "2.550"
            name: CRPS (pipeline scale, 13 quantiles)
          - type: nll
            value: "-0.598"
            name: Negative log-likelihood
---

<!--
SPDX-FileCopyrightText: Contributors to the s4casting project

SPDX-License-Identifier: MPL-2.0
-->

# LianderPowerTransformer-29M

LianderPowerTransformer-29M (LPT) is a probabilistic forecasting model for aggregated
power measurements in medium-voltage distribution grids, developed and trained using the
[S4Casting open-source project and forecasting toolkit](https://github.com/alliander-opensource/s4casting)
on the [open LianderPower dataset](https://www.liander.nl/over-ons/open-data/). It is
the companion baseline of the LianderPower dataset release. The model takes a history of
measurements and weather covariates at any of six sampling rates from 10 minutes to one
week, with context windows from 21 to 512 days, and forecasts the masked tail of the
window. For every time step it returns a four-component Gaussian mixture, from which 13
quantiles are derived.

| | |
|---|---|
| Model family | s4casting `transformer` |
| Parameters | 29,443,616 (float32) |
| Checkpoint | training step 490,000 of a 2,000,000-step schedule, run started 8 November 2026 |
| Code release | s4casting tag `v0.1.0`, commit `6b642d87fc5e5b11902c4eb8dfba3e3cad72c140` |
| Training configuration | [`training_config.toml`](training_config.toml), the configuration of the run with the model, optimiser and training settings unchanged; data paths, the time-feature dataset reference and the benchmark model name are adapted to the public data layout. Validated against the code release |
| Weights | `LianderPowerTransformer-29M.safetensors` (model weights only, no optimizer state) |
| ONNX | `LianderPowerTransformer-29M.onnx`, exported from the safetensors file by the code release, opset 18, dynamic batch and time axes |
| Integrity | SHA-256 of every file in `checksums.sha256`, verify with `shasum -a 256 -c checksums.sha256` |
| Model weights licence | MPL-2.0 (`LICENSE` in the release) |
| Source code licence | MPL-2.0 (S4Casting repository) |
| Training dataset licence | LianderPower CC BY 4.0; Open-Meteo weather data CC BY 4.0|

## Intended use

Short- to medium-term forecasting of aggregated medium-voltage power, for research and
grid capacity management. Typical uses are day-ahead and multi-day load forecasts for
substations and feeders at 10-minute to hourly resolution, longer-range forecasts at
daily and weekly resolution, studies of forecast uncertainty using the predicted
quantiles, and as the reference model for benchmarking other forecasters on the
LianderPower dataset. The forecast horizon is set at inference time through the input
mask; the model was trained with horizons between 5% and 30% of the context window.

## Non-intended use

- Safety-critical real-time grid control without human oversight.
- Forecasting of individual customer consumption. The model was trained on
  aggregated grid measurements only and has never seen customer-level data.
- Use on grids or horizons substantially different from the training data without
  validation, for example other countries, low-voltage networks, sampling rates other
  than the six it was trained on, or horizons beyond 30% of the supplied context.
- Any use that treats a single quantile as a guarantee. The outputs are
  probabilistic and should be consumed as such.

## Training data description

**Dataset.** LianderPower, an anonymised power-grid time-series dataset derived from
operational SCADA telemetry of Liander, a Dutch distribution system operator, published
under CC BY 4.0. The public release contains aggregate measurement time series and
Open-Meteo weather covariates. It contains no exact asset coordinates, topology,
switching states, feeder identifiers, customer information or raw single-sensor
measurements. Locations are spatially anonymised before publication by random jittering
and aggregation of nearby measurement points.

**Measurements.** 1,459 measurement locations, September 2013 to December 2024, about
9,000 location-years of 5-minute data. The training pipeline resamples each series to
the sampling rate drawn for the sample (10, 15, 30, 60, 1440 or 10080 minutes). Series
are labelled as native or pseudo aggregates and carry either power or current as the
sensor unit.

**Covariates.** 17 Open-Meteo variables, matched to each measurement location by nearest
neighbour over the weather grid: temperature, relative humidity, dew point,
precipitation, rain, snow depth, sea-level and surface pressure, cloud cover, reference
evapotranspiration, vapour pressure deficit, wind speed at 100 m, and shortwave, direct,
diffuse, direct-normal and terrestrial radiation. They were retrieved from the
[Open-Meteo](https://open-meteo.com/) historical weather API with
`src/s4casting/data/preparation/weather.py` in the S4Casting repository, and are
licensed CC BY 4.0 by Open-Meteo. The public LianderPower release currently ships 4 of
the 17 variables (temperature, wind speed at 100 m, shortwave radiation and direct-normal
irradiance); the full set used in training will be published at the same location as the
dataset. Until then, the remaining 13 variables can be regenerated with the script above.
See Known limitations.

**Time and location features.** One additional input channel, the unix timestamp, from
which the model derives time-of-day, day-of-week and day-of-year embeddings. No location
coordinates are used: the model is location-agnostic and the same weights apply to any
series.

**Split.** Time-based: validation samples are drawn from 2024 onwards
(`validation.split_type = "time"`, `start_year = 2024`, `percentage = 5`).

**Preprocessing.** Each training window is normalised per sample inside the model
(`norm_clamp = 10`, `norm_eps = 1e-4`); no global scaling is applied to the data. Every
sample draws a sampling rate and a context window within the per-rate limits of
`[io.interval_context_limits]`. Covariate dropout was enabled: a random subset of the
weather channels is hidden per sample, and with probability 0.3 a kept covariate is also
hidden over all or part of the forecast window, so the model tolerates missing weather
inputs and weather forecasts that end before the horizon does.

## Architecture overview

The S4Casting toolkit implements S4 and selective state-space (S6, Mamba-style) models
and Transformer variants, see `src/s4casting/model/`. This release is the Transformer
variant with the following configuration:

- **Input.** 19 channels: the target series, 17 weather covariates and the unix
  timestamp, at one of six sampling rates (10, 15, 30 and 60 minutes, daily, weekly).
  Context windows of 21, 32, 64, 128, 256, 364, 448 or 512 days at the base rate of 15
  minutes, bounded per rate. Each channel carries a mask so missing values are handled
  explicitly; the forecast horizon is the masked tail of the target channel.
- **Pre-training task.** Contiguous Patch Masking (CPM, after TiRex): on top of the
  forecast window, which covers 5% to 30% of the sample, blocks of target patches inside
  the context are masked with a per-sample probability of up to 0.25, in blocks of up to
  2 days, leaving the first 2 days of history untouched. Masked blocks are hidden from
  the input and included in the loss. Covariates and time features are never masked.
- **Patch encoder.** Linear, 4 layers, patch size 8 (2 hours at 15 minutes); value and
  mask of each patch are embedded together.
- **Backbone.** 6 Transformer blocks, latent width 512, 8 heads, attention bias
  enabled, no dropout. Each block applies causal temporal attention over the tokens of a
  channel (`transformer.causal = true`, tokens attend to the past only) and
  bidirectional group attention across channels at the same time step, followed by a
  2-layer MLP.
- **Patch decoder.** Linear, patch size 8, back to the sample's resolution.
- **Output head.** Gaussian mixture with 4 components per time step, emitting
  log-weights, standard deviations and means. Quantiles at the levels 0.01, 0.05, 0.1,
  0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95 and 0.99 are derived from the mixture; the
  ONNX export returns both the mixture parameters and these quantiles.
- **Training objective.** Negative log-likelihood of the mixture on masked targets
  (`alpha_clip = 0.1`, no sigma regularisation), RAdam with learning rate 3e-5, betas
  0.9/0.99, no weight decay, gradient clipping at 1, reduce-on-plateau schedule (factor
  0.3, patience 2), batch size 16, bf16 autocast on CUDA with fp32 master weights,
  seed 42069. The released checkpoint is step 490,000 of a 2,000,000-step schedule.

## Evaluation results

Metrics logged by the training pipeline for this checkpoint on the validation split.
CRPS is computed with `scoringrules.crps_quantile` from the 13 derived quantiles on the
pipeline's input scale, averaged over validation samples; it is not in physical units.

| Metric | Value | Where |
|---|---|---|
| CRPS | 2.550 | validation split, step 490,000 |
| Negative log-likelihood | -0.598 | validation split, step 490,000 |
| Negative log-likelihood | -0.572 | training batch, step 490,000 |

**fev-bench (energy tasks).** Zero-shot results on Energy-domain tasks of
[fev-bench](https://arxiv.org/abs/2509.26468), produced with
`scripts/run_fevbench.py` from this checkpoint. The skill score is relative to Seasonal
Naive (higher is better). Mean SQL is the scaled quantile loss averaged over the ENTSO-E,
KDD Cup 2022 and solar-with-weather tasks (lower is better).

| Model | Skill score | Mean SQL | Notes |
|---|---|---|---|
| LianderPowerTransformer-29M | 0.376 | 0.501 | zero-shot |
| Chronos-2 | 0.412 | 0.546 | |

**Liander2024 benchmark.** Day-ahead forecasts on the
[Liander2024 benchmark](https://huggingface.co/datasets/OpenSTEF/liander2024-energy-forecasting-benchmark)
through the StefBeam evaluator (32-day context, 2-day horizon, 15-minute resolution),
averaged over the benchmark targets. rCRPS and rMAE are computed by openstef-beam
4.7.0; rCRPS values from openstef-beam 4.1.0 onwards are about 20% higher than those from
earlier versions for the same forecasts and must not be compared across that boundary.

| Model | Mean rMAE | Mean rCRPS |
|---|---|---|
| LianderPowerTransformer-29M | 0.070 | 0.050 |

**Implementation parity.** The ONNX shipped with this release agrees with the PyTorch
model loaded with these weights to within 1e-4 on identical inputs, the tolerance the
packaging step enforces; the measured figure for this release is in `manifest.json`.
The comparison uses the geometry of the Liander2024 benchmark: 3072 steps at 15 minutes,
30 days of context followed by a 2-day forecast. The time axis of the ONNX graph is
dynamic, so it runs at other lengths too, but parity was not measured at those.

## Known limitations

- Performance may degrade under distribution shift, for example new connections, grid
  reconfiguration, extreme weather, or rapid growth of solar generation and
  electrification beyond what the 2013 to 2024 history contains.
- **Weather inputs.** The model was trained on 17 weather variables. Running it with
  only the 4 variables in the current public LianderPower release is within what the
  covariate dropout exposed it to, but forecast quality in that setting has not been
  evaluated separately. Weather forecast error is not modelled either: if forecasts
  rather than observations are supplied for the horizon, their error propagates into
  the load forecast.
- **Geometry.** Inputs must be at one of the six trained sampling rates, and the
  sequence length must be a multiple of the patch size (8 steps). The ONNX graph
  accepts any such length from 16 to 32,768 steps and rejects others; accepting a length
  does not mean the model was trained for it. Horizons longer than 30% of the context
  were not seen in training.
- **Location.** The model has no location input, so it cannot learn location-specific
  behaviour beyond what the series itself reveals; this is also what makes it transfer
  to series without coordinates.
- **Calendar effects.** The only calendar input is the timestamp; holidays and other
  special days are not encoded explicitly.
- **Evaluation coverage.** The fev-bench and Liander2024 results above cover energy
  series of the kinds in those benchmarks; other domains were not evaluated.

## Bias and fairness considerations

The model predicts aggregated grid loads and does not process data about individuals.
The training data is anonymised and aggregated at substation and feeder level.
Representativeness still matters: all measurement locations lie within Liander's service
area in the Netherlands, and the mix of urban and rural areas, feeder types, industrial
versus residential load, and solar penetration reflects that area in 2013 to 2024.
Forecast accuracy may differ systematically between such groups, which can matter when
forecasts feed capacity decisions that affect which customers get connected first. Users
should evaluate accuracy per region and feeder type before using the forecasts in
decisions with distributional effects.

## Responsible AI considerations

- **Human oversight.** The model is a decision-support tool. Operational use should
  keep a person in the loop and fall back to established procedures when forecasts
  are implausible.
- **Uncertainty.** Use the full quantile set rather than the median alone, and
  monitor calibration on live data; the reported metrics are validation and benchmark
  figures from fixed periods.
- **Monitoring and re-training.** Track accuracy over time and re-train when the
  grid or its usage patterns change.
- **Provenance and integrity.** The weights are distributed as safetensors, which
  cannot execute code on load, together with SHA-256 checksums, the training
  configuration and the code release they were built with. Verify the checksums
  before use and load the weights with the referenced code release.
- **Data protection.** The model was trained on anonymised and aggregated grid
  measurements rather than individual customer consumption data. The LianderPower
  dataset does not contain customer data or data attributable to individual
  connections.
- **Reporting.** Security issues should be reported according to the S4Casting
  project's [`SECURITY.md`](https://github.com/alliander-opensource/s4casting/blob/main/docs/SECURITY.md);
  other support requests and issues should follow the project's
  [`SUPPORT.md`](https://github.com/alliander-opensource/s4casting/blob/main/docs/SUPPORT.md).

## How to use

```python
from s4casting.inference.onnx_export import build_model, load_configuration, load_checkpoint_weights

config = load_configuration("training_config.toml")
model = build_model(config)
load_checkpoint_weights(model, "LianderPowerTransformer-29M.safetensors", device="cpu")
```

For inference without PyTorch, run `LianderPowerTransformer-29M.onnx` with onnxruntime:
inputs `x` and `xm` of shape `(batch, steps, 19)`, outputs `prediction` of shape
`(batch, steps, 1, 4, 3)` and `quantiles` of shape `(batch, steps, 1, 13)`, where `steps`
is any multiple of 8. Set the target channel of `xm` to 0 over the steps to forecast.
See `docs/ONNX_EXPORT.md` in the code release.

## Contributing

For contributions to S4Casting, including model improvements and new model releases,
see the S4Casting project's
[`CONTRIBUTING.md`](https://github.com/alliander-opensource/s4casting/blob/main/docs/CONTRIBUTING.md).

## Citation

See the `Citation` section of the s4casting repository README.
