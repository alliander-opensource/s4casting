# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import argparse
import sys

from s4casting.data.preparation.dataset_formatter import DatasetFormatter


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command line arguments for dataset formatting.

    Args:
        argv: List of command line arguments. If None, uses sys.argv.

    Returns:
        Parsed arguments namespace.
    """
    ap = argparse.ArgumentParser(description="Format dataset from parquet/csv files.")
    ap.add_argument("--folder", type=str, default="parquet", help="Folder containing input files.")
    ap.add_argument("--output_prefix", type=str, default="external_data_wrapped", help="Prefix for output files.")
    ap.add_argument("--output_dir", type=str, default="output_data", help="Directory to save output files.")
    ap.add_argument("--target_col", type=str, default="value", help="Name of the target column.")
    ap.add_argument("--time_col", type=str, default="time", help="Name of the time column.")
    ap.add_argument("--sample_interval_minutes", type=int, default=5, help="Sampling interval in minutes.")
    ap.add_argument(
        "--locations_file", type=str, default="data/locations.csv", help="CSV file containing location data."
    )
    return ap.parse_args(argv)


def main(argv=None):
    """Main function to run inference from command line."""
    args = parse_args(argv)
    DatasetFormatter(
        folder=args.folder,
        output_prefix=args.output_prefix,
        output_dir=args.output_dir,
        target_col=args.target_col,
        time_col=args.time_col,
        sample_interval_minutes=args.sample_interval_minutes,
        locations_file=args.locations_file,
    ).run()


if __name__ == "__main__":
    main(sys.argv[1:])
