# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tqdm


class DatasetFormatter:
    """Formats a dataset from parquet/csv files, creates spans, memmap, and metadata JSON."""

    def __init__(
        self,
        folder: str,
        output_prefix: str = "external_data_wrapped",
        output_dir: str = ".",
        target_col: str = "value",
        time_col: str = "time",
        sample_interval_minutes: int = 5,
        locations_file: str = "data/locations.csv",
    ) -> None:
        """The constructor for DatasetFormatter.

        Args:
            folder: Path to the folder containing data files.
            output_prefix: Prefix for output files.
            output_dir: Directory to save output files.
            target_col: Name of the target column in the data files.
            time_col: Name of the time column in the data files.
            sample_interval_minutes: Sampling interval in minutes.
            locations_file: Path to the locations CSV file with columns "name", "lon", and "lat".
        """
        self.folder: str = folder
        self.output_prefix: str = output_prefix
        self.output_dir: str = output_dir
        self.parquet_list: list[str] = self.list_files()
        self.df_spans: pd.DataFrame = pd.DataFrame(columns=["id", "location", "datetime_start", "num_values"])
        self.total_n: int = 0
        self.data_array: np.ndarray = np.array([], dtype="float32")
        self.locations: dict[int, dict[str, Any]] = {}
        self.target_col: str = target_col
        self.time_col: str = time_col
        self.sample_interval_minutes: int = sample_interval_minutes
        try:
            self.df_locations: pd.DataFrame = pd.read_csv(locations_file)
        except FileNotFoundError:
            self.df_locations: pd.DataFrame = pd.DataFrame(columns=["name", "lon", "lat"])

    def list_files(self) -> list[str]:
        """List all parquet and csv files in the folder.

        Returns:
            List of file names.
        """
        return [
            f
            for f in os.listdir(self.folder)
            if (Path(self.folder) / f).is_file() and (f.endswith(".parquet") or f.endswith(".csv"))
        ]

    def prepare_dataframe(self, file: str, columns: list[str]) -> pd.DataFrame:
        """Load a file and select columns, converting 'time' to unixtime.

        Args:
            file: File name.
            columns: List of columns to select.

        Returns:
            DataFrame with selected columns and 'time' as unixtime.
        """
        path = Path(self.folder) / file
        if file.endswith(".parquet"):
            df = pd.read_parquet(path)
        elif file.endswith(".csv"):
            df = pd.read_csv(path, usecols=columns)
        df = df[columns]
        df[self.time_col] = pd.to_datetime(df[self.time_col]).astype("int64") // 10**9  # type: ignore
        return df

    def update_spans(self, span_rows: list[dict[str, Any]], df: pd.DataFrame, i: int) -> list[dict[str, Any]]:
        """Update the spans DataFrame with a new span.

        Args:
            span_rows: Current list of span dictionaries.
            df: DataFrame for the current location.
            i: Location index.

        Returns:
            Updated list of span dictionaries.
        """
        num_values = len(df)
        span_rows.append({
            "id": self.total_n,
            "location": int(i),
            "datetime_start": int(df.iloc[0][self.time_col]),
            "num_values": int(num_values),
        })
        self.total_n += num_values
        return span_rows

    def update_locations(self, i: int, file: str) -> None:
        """Update the locations dictionary with a new location.

        Args:
            i: Location index.
            file: File name.
        """
        name = Path(file).stem
        try:
            loc = self.df_locations[self.df_locations["name"] == name].iloc[0]
            self.locations[i] = {"name": name, "lon": loc["lon"], "lat": loc["lat"]}
        except IndexError:
            self.locations[i] = {"name": name}

    def build(self) -> None:
        """Build the dataset from files."""
        data_arrays = []
        span_rows = []
        for i, file in enumerate(tqdm.tqdm(self.parquet_list, mininterval=1.0)):
            df = self.prepare_dataframe(file, columns=[self.target_col, self.time_col])
            span_rows = self.update_spans(span_rows, df, i)
            self.update_locations(i, file)
            data_arrays.append(df[self.target_col].to_numpy(dtype="float32"))

        self.df_spans = pd.DataFrame(span_rows)
        self.data_array = np.concatenate(data_arrays)

    def save(self) -> None:
        """Save the memmap, spans parquet, and locations JSON."""
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)
        memmap_path = Path(self.output_dir) / f"{self.output_prefix}.np"
        spans_path = Path(self.output_dir) / f"{self.output_prefix}_spans.parquet"
        json_path = Path(self.output_dir) / f"{self.output_prefix}.json"

        print(f"Saving memmap to {memmap_path}")  # noqa: T201
        # Save memmap
        np_memmap = np.memmap(memmap_path, mode="w+", dtype="float32", shape=(self.total_n,))
        np_memmap[:] = self.data_array[:]
        np_memmap.flush()

        print(f"Saving spans to {spans_path}")  # noqa: T201
        # Save spans to parquet
        self.df_spans.to_parquet(spans_path, index=False)

        print(f"Saving locations to {json_path}")  # noqa: T201
        # Save locations to json
        with Path(json_path).open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "dimension": 1,
                    "sample_interval_minutes": self.sample_interval_minutes,
                    "dataset": f"{self.output_prefix}.np",
                    "spans": f"{self.output_prefix}_spans.parquet",
                    "locations": self.locations,
                },
                f,
                indent=4,
            )

    def run(self) -> None:
        """Run the full formatting process."""
        self.build()
        self.save()


if __name__ == "__main__":
    DatasetFormatter(folder="parquet", output_prefix="external_data_wrapped", output_dir="output_data").run()
