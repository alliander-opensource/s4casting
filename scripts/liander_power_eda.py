# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

"""EDA script for the Liander Power croissant dataset.

Loads all measurement spans for one location within a 1-year window,
GPS-matches the nearest weather station, resamples every span independently
to 15-minute resolution (no interpolation across gaps), and plots all signals
in a single interactive Plotly figure.

Power is on the left y-axis (raw); weather features are normalised 0-1 on the
right y-axis so they can share the same scale. Each trace can be toggled via
the legend.

Usage:
    uv run python scripts/liander_power_eda.py
    uv run python scripts/liander_power_eda.py --location-id 51 --year 2022
"""

from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial import cKDTree  # type: ignore[import]

UTC = datetime.UTC
RESAMPLE = "15min"

FEAT_LABELS = {
    "temperature_2m": "Temperature 2m",
    "wind_speed_100m": "Wind speed 100m",
    "shortwave_radiation": "Shortwave radiation",
    "direct_normal_irradiance": "DNI",
}
FEAT_COLORS = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd"]
POWER_COLOR = "#1f77b4"

log = logging.getLogger(__name__)


def to_series(start_time: int, sample_interval_s: int, values: np.ndarray) -> pd.Series:
    """Convert a span's values to a pd.Series with a UTC DatetimeIndex.

    Returns:
        Series with a UTC-aware DatetimeIndex at the given sample interval.
    """
    idx = pd.date_range(
        start=pd.Timestamp(start_time, unit="s", tz=UTC),
        periods=len(values),
        freq=pd.tseries.frequencies.to_offset(datetime.timedelta(seconds=sample_interval_s)),
    )
    return pd.Series(values.astype(float), index=idx)


def resample_spans(
    rows: pd.DataFrame, t_start: pd.Timestamp, t_end: pd.Timestamp, *, interpolate: bool
) -> tuple[list, list]:
    """Resample each span to RESAMPLE independently, clip to window, concatenate with None gaps.

    Returns:
        Tuple of (x_values, y_values) lists with None sentinels marking span boundaries.
    """
    xs: list = []
    ys: list = []
    for _, row in rows.sort_values("start_time").iterrows():
        s = to_series(int(row["start_time"]), int(row["sample_interval_s"]), np.array(row["values"], dtype=np.float32))
        s = s[(s.index >= t_start) & (s.index < t_end)]
        if s.empty:
            continue
        s = s.resample(RESAMPLE).interpolate("time") if interpolate else s.resample(RESAMPLE).mean()
        if xs:
            xs.append(None)
            ys.append(None)
        xs.extend(s.index.tolist())
        ys.extend(s.tolist())
    return xs, ys


def normalise(ys: list) -> list:
    """Normalise a list of values (with possible None sentinels) to [0, 1].

    Returns:
        List with non-None values scaled to [0, 1]; None sentinels are preserved.
    """
    vals = np.array([v for v in ys if v is not None], dtype=float)
    finite = vals[np.isfinite(vals)]
    if len(finite) == 0 or finite.max() == finite.min():
        return ys
    lo, hi = finite.min(), finite.max()
    return [None if v is None else float((v - lo) / (hi - lo)) for v in ys]


def gps_match(meas_lat: float, meas_lon: float, weather_meta: pd.DataFrame) -> int:
    """Return the nearest weather_location_id to the given coordinates.

    Returns:
        The weather_location_id of the closest weather station.
    """
    unique = weather_meta.drop_duplicates("weather_location_id").set_index("weather_location_id")
    ids = unique.index.tolist()
    _, idx = cKDTree(list(zip(unique["lon"], unique["lat"]))).query([(meas_lon, meas_lat)], k=1)
    return ids[int(idx[0])]


def _build_and_save_figure(
    loc_id: int,
    meas_lat: float,
    meas_lon: float,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
    matched_wlid: int,
    dist_km: float,
    m_spans: pd.DataFrame,
    feature_names: list[str],
    weather_spans: dict[str, pd.DataFrame],
) -> Path:
    """Build the interactive Plotly EDA figure and save it to an HTML file.

    Returns:
        Path to the saved HTML file.
    """
    power_xs, power_ys = resample_spans(m_spans, t_start, t_end, interpolate=False)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=power_xs,
            y=power_ys,
            mode="lines",
            name="Power [MW]",
            line={"color": POWER_COLOR, "width": 1},
            yaxis="y1",
        )
    )
    for fi, feat in enumerate(feature_names):
        wx, wy = resample_spans(weather_spans[feat], t_start, t_end, interpolate=True)
        label = FEAT_LABELS.get(feat, feat)
        fig.add_trace(
            go.Scatter(
                x=wx,
                y=normalise(wy),
                mode="lines",
                name=f"{label} (norm.)",
                line={"color": FEAT_COLORS[fi % len(FEAT_COLORS)], "width": 1},
                yaxis="y2",
            )
        )
    fig.update_layout(
        title={
            "text": (
                f"Liander Power EDA  ·  location {loc_id} "
                f"(lat={meas_lat:.4f}, lon={meas_lon:.4f})  "
                f"{t_start.year}  ·  weather {matched_wlid} (~{dist_km:.1f} km)"
            ),
            "font_size": 13,
        },
        xaxis={
            "showspikes": True,
            "spikemode": "across",
            "spikesnap": "cursor",
            "spikecolor": "rgba(0,0,0,0.25)",
            "spikethickness": 1,
        },
        yaxis={"title": "Power [MW]", "title_font_color": POWER_COLOR, "tickfont_color": POWER_COLOR},
        yaxis2={
            "title": "Weather features (normalised 0-1)",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
            "range": [-0.05, 1.05],
        },
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        height=500,
    )
    out_path = Path(f"liander_power_eda_{loc_id}.html")
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return out_path


def main() -> None:
    """Run the Liander Power EDA: load one location's spans and plot against weather."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release-dir", type=Path, default=Path("data/LianderPower"))
    parser.add_argument("--location-id", type=int, default=None)
    parser.add_argument(
        "--year", type=int, default=None, help="Calendar year to plot (default: first year of data for chosen location)"
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Choose location
    # ------------------------------------------------------------------
    log.info("Scanning measurement metadata ...")
    m_meta = pd.read_parquet(
        args.release_dir / "measurements.parquet",
        columns=["location_id", "lat", "lon", "start_time", "sample_interval_s", "num_values"],
    )

    if args.location_id is None:
        loc_id = int(m_meta.groupby("location_id")["num_values"].sum().idxmax())
        log.info("Defaulting to location_id=%d (most total samples)", loc_id)
    else:
        loc_id = args.location_id

    loc_meta = m_meta[m_meta["location_id"] == loc_id].sort_values("start_time")
    if loc_meta.empty:
        raise SystemExit(f"location_id={loc_id} not found")

    meas_lat = float(loc_meta.iloc[0]["lat"])
    meas_lon = float(loc_meta.iloc[0]["lon"])
    log.info("Location %d: lat=%.5f, lon=%.5f, %d span(s) total", loc_id, meas_lat, meas_lon, len(loc_meta))

    # ------------------------------------------------------------------
    # 2. Define 1-year window
    # ------------------------------------------------------------------
    if args.year is not None:
        t_start = pd.Timestamp(year=args.year, month=1, day=1, tz=UTC)
    else:
        first_ts = int(loc_meta.iloc[0]["start_time"])
        t_start = pd.Timestamp(first_ts, unit="s", tz=UTC).replace(month=1, day=1, hour=0, minute=0, second=0)
    t_end = t_start + pd.DateOffset(years=1)

    t_start_s = int(t_start.timestamp())
    t_end_s = int(t_end.timestamp())
    span_end_s = loc_meta["start_time"] + loc_meta["num_values"] * loc_meta["sample_interval_s"]
    window_meta = loc_meta[(loc_meta["start_time"] < t_end_s) & (span_end_s > t_start_s)]

    log.info("Window: %s - %s,  %d span(s) overlap", t_start.date(), t_end.date(), len(window_meta))
    if window_meta.empty:
        raise SystemExit("No spans in the requested year. Try --year with a different value.")

    # ------------------------------------------------------------------
    # 3. Load measurement values for spans in window
    # ------------------------------------------------------------------
    log.info("Loading measurement spans ...")
    m_spans = pd.read_parquet(
        args.release_dir / "measurements.parquet",
        filters=[("location_id", "=", loc_id), ("start_time", "in", window_meta["start_time"].tolist())],
    )

    # ------------------------------------------------------------------
    # 4. GPS match
    # ------------------------------------------------------------------
    log.info("Loading weather metadata for GPS matching ...")
    w_meta = pd.read_parquet(args.release_dir / "weather.parquet", columns=["weather_location_id", "lat", "lon"])
    matched_wlid = gps_match(meas_lat, meas_lon, w_meta)

    w_loc_row = w_meta[w_meta["weather_location_id"] == matched_wlid].iloc[0]
    dist_km = np.sqrt((meas_lat - float(w_loc_row["lat"])) ** 2 + (meas_lon - float(w_loc_row["lon"])) ** 2) * 111.0
    log.info("Matched weather_location_id=%d (~%.1f km)", matched_wlid, dist_km)

    # ------------------------------------------------------------------
    # 5. Load weather spans that overlap the window
    # ------------------------------------------------------------------
    log.info("Loading weather spans ...")
    w_all = pd.read_parquet(args.release_dir / "weather.parquet", filters=[("weather_location_id", "=", matched_wlid)])
    feature_names: list[str] = list(dict.fromkeys(w_all["feature_name"].tolist()))

    weather_spans: dict[str, pd.DataFrame] = {}
    for feat in feature_names:
        feat_df = w_all[w_all["feature_name"] == feat].copy()
        feat_end_s = feat_df["start_time"] + feat_df["num_values"] * feat_df["sample_interval_s"]
        weather_spans[feat] = feat_df[(feat_df["start_time"] < t_end_s) & (feat_end_s > t_start_s)]

    # ------------------------------------------------------------------
    # 6. Build and save figure
    # ------------------------------------------------------------------
    out_path = _build_and_save_figure(
        loc_id, meas_lat, meas_lon, t_start, t_end, matched_wlid, dist_km, m_spans, feature_names, weather_spans
    )
    log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
