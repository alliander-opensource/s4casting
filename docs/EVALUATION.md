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