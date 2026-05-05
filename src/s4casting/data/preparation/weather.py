# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import dotenv
import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
from openmeteo_sdk.WeatherApiResponse import WeatherApiResponse
from retry_requests import retry
from s4casting.data.preparation.constants import DEFAULT_PARAMS
from tqdm import tqdm

dotenv.load_dotenv()


class WeatherDatasetFormatter:
    """Formats a weather dataset using OpenMeteo API, creates spans, memmap, and metadata JSON.

    Importanted to note is that all our data is measured for the UTC timezone.
    """

    def __init__(
        self,
        df_locations: Path,
        start_date: str,
        end_date: str,
        output_prefix: str = "weather",
        output_dir: str = "out",
        tilt: float = 48.7,
        azimuth: float = 180.0,
    ) -> None:
        """The constructor for WeatherDatasetFormatter.

        Args:
            df_locations: DataFrame with 'lon' and 'lat' columns for locations.
            start_date: Start date for weather data.
            end_date: End date for weather data.
            output_prefix: Prefix for output files.
            output_dir: Directory to save output files.
            tilt: Tilt angle for solar calculations.
            azimuth: Azimuth angle for solar calculations.
        """
        self.df_locations = pd.read_csv(df_locations).copy()
        if not {"lon", "lat"}.issubset(self.df_locations.columns):
            raise ValueError("coords_csv must contain columns: lon, lat")
        self.start_date = datetime.fromisoformat(start_date).replace(tzinfo=UTC)
        self.end_date = datetime.fromisoformat(end_date).replace(tzinfo=UTC)
        self.output_prefix = output_prefix
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)

        self.weather_params = list(DEFAULT_PARAMS)
        self.tilt = tilt
        self.azimuth = azimuth

        # Derived values
        self.samples = int((self.end_date.timestamp() - self.start_date.timestamp()) // 3600) + 24

        # Ensure location_id
        self.df_locations["location_id"] = (
            self.df_locations.groupby(["lon", "lat"], sort=False).ngroup() + 1_000_000
        ).astype(int)

        # Runtime state
        self.spans: list[tuple[int, int, int, int]] = []
        self.locations_dict: dict[int, dict[str, float]] = {}

        # Allocate memmap
        self.data_out = np.memmap(
            Path(self.output_dir) / f"{self.output_prefix}.np",
            dtype="float32",
            mode="w+",
            shape=(len(self.df_locations) * self.samples, len(self.weather_params)),
        )
        self.cache_session: requests_cache.CachedSession | None = None
        self.total_calls = 0
        self.cache_hits = 0

    def _make_client(self) -> openmeteo_requests.Client:
        """Create an OpenMeteo client with caching and retry logic.

        Returns:
            Configured OpenMeteo client.
        """
        self.cache_session = requests_cache.CachedSession("data/.cache", expire_after=-1)

        def count_hits(response, *_args, **_kwargs):
            if getattr(response, "from_cache", False):
                self.cache_hits += 1

        self.total_calls += 1

        self.cache_session.hooks["response"].append(count_hits)

        retry_session = retry(self.cache_session, retries=5, backoff_factor=0.2)
        return openmeteo_requests.Client(session=retry_session)  # type: ignore[return-value]

    def _fetch(self, lat: float, lon: float) -> WeatherApiResponse:
        """Fetch weather data for given coordinates.

        Args:
            lat: Latitude.
            lon: Longitude.

        Returns:
            WeatherApiResponse object with fetched data.
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": self.start_date.strftime("%Y-%m-%d"),
            "end_date": self.end_date.strftime("%Y-%m-%d"),
            "hourly": list(self.weather_params),
            "timezone": "UTC",
            "tilt": self.tilt,
            "azimuth": self.azimuth,
        }

        if os.getenv("OPENMETEO_API_KEY"):
            params["apikey"] = os.getenv("OPENMETEO_API_KEY")
            url = "https://customer-archive-api.open-meteo.com/v1/archive"
        else:
            url = "https://archive-api.open-meteo.com/v1/archive"
        client = self._make_client()

        try:
            responses = client.weather_api(url, params=params)  # type: ignore[attr-defined]
        except Exception as e:
            raise ValueError(f"Error fetching data for coordinates ({lat}, {lon}): {e}") from e

        if not responses:
            raise ValueError(f"Error fetching data for coordinates ({lat}, {lon})")
        return responses[0]

    def build(self):
        """Build the weather dataset."""
        row_offset = 0
        iterator = self.df_locations.itertuples(index=False)

        for row in tqdm(iterator):
            lon = float(row.lon)
            lat = float(row.lat)
            location_id = int(row.location_id)

            weather = self._fetch(lat, lon)

            variables = [weather.Hourly().Variables(i).ValuesAsNumpy() for i in range(len(self.weather_params))]  # type: ignore[attr-defined]

            for j, data in enumerate(variables):
                self.data_out[row_offset : row_offset + self.samples, j] = data

            self.spans.append((row_offset, location_id, int(self.start_date.timestamp()), self.samples))
            self.locations_dict[location_id] = {"lon": lon, "lat": lat}
            row_offset += self.samples

        self.data_out.flush()
        print(  # noqa: T201
            f"API calls: {self.total_calls} | cache hits: {self.cache_hits}"
        )

    def save(self):
        """Save the formatted weather dataset to disk."""
        spans_path = Path(self.output_dir) / f"{self.output_prefix}_spans.parquet"
        json_path = Path(self.output_dir) / f"{self.output_prefix}.json"

        pd.DataFrame(self.spans, columns=["id", "location", "datetime_start", "num_values"]).to_parquet(spans_path)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "dimension": len(self.weather_params),
                    "sample_interval_minutes": 60,
                    "dataset": f"{self.output_prefix}.np",
                    "spans": f"{self.output_prefix}_spans.parquet",
                    "feature_names": list(self.weather_params),
                    "locations": self.locations_dict,
                },
                f,
                indent=4,
            )

    def run(self) -> None:
        """Run the full formatting process."""
        self.build()
        self.save()
