<!--
SPDX-FileCopyrightText: Contributors to the s4casting project

SPDX-License-Identifier: MPL-2.0
-->

# Evaluation

The evaluators in the S4Casting project are responsible for assessing the performance of the model during training, benchmarking, and inference. They provide metrics, visualizations, and reports to help understand how well the model is performing on the given tasks.

## Overview of Evaluators

### **GMMHeadEvaluator**
The `GMMHeadEvaluator` is a specialized evaluator for models with Gaussian Mixture Model (GMM) heads. It evaluates the model's predictions and generates detailed reports. 

### **QuantileHeadEvaluator**
The `QuantileHeadEvaluator` is designed for models with quantile-based output heads. It evaluates the model's predictions and generates reports specific to quantile forecasting.

## Choosing an Output Head

- GMM head: provides a full continuous predictive distribution with higher expressiveness.
- Quantile head: simpler and produces calibrated prediction intervals at fixed quantile levels.

Current status: on primary benchmarks, neither approach has shown a consistent performance advantage so far.

## Forecast Evaluation Metrics

This section evaluates the probabilistic forecasting performance of the trained model using the `Metrics` class from `s4casting.eval.metric`.
It summarizes both **distribution-level accuracy** and **quantile-specific behavior**.


### 1. Probabilistic Metrics

| Metric                                         | Description                                                                            | Interpretation                                                                   |
| ---------------------------------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **CRPS** (Continuous Ranked Probability Score) | Measures how closely the predicted cumulative distribution matches the actual outcome. | Lower scores indicate sharper and better-calibrated forecasts.                   |
| **NLL** (Negative Log-Likelihood)              | Measures how much probability mass the model assigns to the true observation.          | More negative values indicate a more confident and accurate probabilistic model. |

### 2. Quantile-Based Metrics

These metrics evaluate each quantile level (0.05, 0.1, 0.3, …) as a thresholding task relative to a congestion threshold.

| Metric        | Description                                                            | Interpretation                                                                                                                 |
| ------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Precision** | Fraction of predicted exceedances that were correct. | High precision means fewer false alarms. Low precision means the model over-predicts events.                                   |
| **Recall**    | Fraction of actual exceedances that were captured by the forecast.     | High recall means the model successfully flags most real events. Low recall indicates missed events.                           |
| **Fβ** (β=10) | Recall-weighted harmonic mean of precision and recall.                 | Highlights the model’s ability to detect events, placing strong emphasis on recall.                                            |
| **MAE**       | Mean absolute error between each predicted quantile and the observation.| Lower = quantile closer to truth; unrelated to event classification. |

### Plots created

| Plot Name                          | Purpose                                                                 |
|-----------------------------------|-------------------------------------------------------------------------|
| **Training Data Plot**            | Visualize predictions on training data and compare with ground truth.  |
| **Benchmarking Forecast Plot**    | Evaluate predictions on benchmarking data and analyze quantiles.       |   |
| **Medium-Term Training Quantiles Plot** | Evaluate quantile predictions for medium-term forecasting.     |

## fev-bench

`FevBenchBenchmark` scores the model zero-shot on [fev-bench](https://arxiv.org/abs/2509.26468) tasks with the
`fev` library, using fev's own protocol: rolling evaluation windows per task, Scaled Quantile Loss (SQL) at the
quantile levels 0.1 to 0.9 as the primary metric, and MASE, WQL and WAPE as extra metrics. The task
definitions and the published baseline results are read from the fev repository at a pinned release, so
results stay comparable; by default only the 26 Energy-domain tasks run (ENTSO-E load, EPF prices, GEFCom,
solar, KDD Cup 2022 wind, ERCOT and ETT). `fev.leaderboard` ranks each run against the Seasonal Naive and
Chronos-2 baselines by skill score and win rate with bootstrap confidence intervals.

Each fev series is presented to the model as follows (`s4casting.eval.fev_bench.adapter`):

- the target sits in channel 0 and is masked over the forecast horizon; the point forecast is the median;
- the task's covariates fill the covariate slots in task order. Known-future covariates keep their horizon
  values, past-only covariates are masked there, and unused slots are zero and masked, exactly as covariate
  dropout presents them in training. Covariate order carries no meaning for the models;
- the unix-time channel is filled from the task timestamps;
- the transformer gets the series at its own length, capped at the longest window it trained with at that
  rate and floored to whole patches, so no masked padding is ever prepended: a long series (ERCOT) is cut to
  the longest trained window, a short one (one year of solar at weekly resolution) is used whole. The model
  trains on unpadded windows of many lengths with relative positions and timestamp features, so a length
  between two trained ones is in distribution, whereas a long masked run would not be. The SSM, which the
  training batcher zero-pads to one fixed length, is padded the same way;
- the series is handed over at its own sample rate, which must be one of the training rates: the benchmark
  never resamples. A task at a rate the model was not trained on (weekly data for an hourly-only model) is
  skipped and counted in `fev_bench/num_skipped`, and a warning says that the leaderboard then covers the
  remaining tasks only and is not comparable with results over the full task set;
- multivariate tasks (ETT) are forecast one target column at a time and recombined.

Training sources that are test data in fev-bench (GEFCom2017, SDWPF = KDD Cup 2022, OPSD) are excluded from
`configs/transformer.toml`; any remaining overlap must be declared in `trained_on_datasets` so the summaries
flag it.
