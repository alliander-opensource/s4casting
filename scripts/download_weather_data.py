# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import argparse
import sys

from s4casting.data.preparation.weather import WeatherDatasetFormatter


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: List of command line arguments.

    Returns:
        Parsed arguments.
    """
    ap = argparse.ArgumentParser(description="Build and save weather dataset from coordinates CSV.")
    ap.add_argument("--coords_csv", type=str, required=True, help="CSV with columns: lon, lat")
    ap.add_argument("--start_date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end_date", type=str, required=True, help="End date (YYYY-MM-DD)")
    ap.add_argument("--output_prefix", type=str, default="weather", help="Prefix for output files")
    ap.add_argument("--output_dir", type=str, default="out", help="Directory to save outputs")
    ap.add_argument("--tilt", type=float, default=48.7, help="PV tilt")
    ap.add_argument("--azimuth", type=float, default=180.0, help="PV azimuth")
    return ap.parse_args(argv)


def main(argv=None):
    """Main function to build weather dataset from coordinates CSV.

    Args:
        argv: List of command line arguments.
    """
    args = parse_args(argv)

    WeatherDatasetFormatter(
        df_locations=args.coords_csv,
        start_date=args.start_date,
        end_date=args.end_date,
        output_prefix=args.output_prefix,
        output_dir=args.output_dir,
        tilt=args.tilt,
        azimuth=args.azimuth,
    ).run()


if __name__ == "__main__":
    main(sys.argv[1:])
