# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
from openstef_beam.evaluation.models.report import EvaluationReport, EvaluationSubsetReport


def get_external_benchmarks(csv_file, iteration: int) -> pd.DataFrame:
    """Get external benchmarks from a CSV file.

    Args:
        csv_file (str): Path to the CSV file containing benchmark data.
        iteration (int): Iteration number to filter the data.

    Returns:
        pd.DataFrame: DataFrame containing the filtered benchmark data.
    """
    df = pd.read_csv(csv_file, na_values=["inf", "-inf"])
    df = df[df.iteration == iteration]
    df.replace(np.inf, np.nan, inplace=True)
    df.replace(-np.inf, np.nan, inplace=True)
    df.rename(columns={"location": "B1B2B3"}, inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df


def filter_report(report: EvaluationReport, filtering: str) -> EvaluationSubsetReport | None:
    """Filter the evaluation report based on the given filtering criteria.

    Args:
        report (EvaluationReport): The evaluation report to filter.
        filtering (str): The filtering criteria.

    Returns:
        EvaluationSubsetReport | None: The filtered subset report or None if not found.
    """
    for sub in report.subset_reports:
        if str(sub.filtering) == filtering:
            return sub
    return None


def build_summary_from_reports(
    reports: list[tuple],
    metrics_wanted: set[str],
    filtering: str,
) -> pd.DataFrame:
    """Returns DataFrame with columns of desired metrics computed across targets.

    Args:
        reports (tuple): Tuple of (metadata, EvaluationReport) pairs.
        metrics_wanted (set[str]): Set of desired metric names.
        filtering (str): Filtering criteria to select the subset report.

    Returns:
        pd.DataFrame: DataFrame summarizing the metrics.
    """
    rows: list[dict] = []

    for meta, report in reports:
        subset = filter_report(report, filtering)  # selects pred mimicking production setup
        if subset is None:
            continue

        global_metric = subset.get_global_metric()

        if global_metric is None:
            raise ValueError("Check if your targets make sense, no global metrics are found")

        for quantile, metric in global_metric.metrics.items():
            for metric_name, value in metric.items():
                if metric_name in metrics_wanted:
                    rows.append({
                        "group": str(meta.group_name),
                        "quantile": quantile,
                        "metric": metric_name,
                        "value": float(value),
                    })

    df = pd.DataFrame(rows)
    return df.groupby(["group", "quantile", "metric"])["value"].mean().reset_index(name="mean")
