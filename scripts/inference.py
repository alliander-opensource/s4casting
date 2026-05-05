# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import argparse
import sys
import warnings

from s4casting.inference.runner import run_inference

warnings.filterwarnings("ignore")


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command line arguments for inference script.

    Args:
        argv: List of command line arguments. If None, uses sys.argv.

    Returns:
        Parsed arguments namespace.
    """
    ap = argparse.ArgumentParser(description="Run simple S4Casting inference on a dataframe.")
    ap.add_argument("--config-path", required=True, help="Path to model config TOML.")
    ap.add_argument("--data", required=True, help="Path to input data (parquet or csv).")
    ap.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file.")
    ap.add_argument("--target-col", default="measurements", help="Target column name in data.")
    ap.add_argument("--time-col", default="time", help="Time column name in data.")
    ap.add_argument(
        "--save-path-predictions", help="Where to store output predictions Parquet (omit to disable saving)."
    )
    ap.add_argument("--save-path-pickle", help="Where to store output pickle (omit to disable saving).")
    ap.add_argument("--plot-path", default="out/inference_plot.html", help="Path to save plot HTML.")  # added
    ap.add_argument("--show-plots", action="store_true", help="Whether to display plots.")
    return ap.parse_args(argv)


def main(argv=None):
    """Main function to run inference from command line."""
    args = parse_args(argv)
    run_inference(
        config=args.config_path,
        data_path=args.data,
        checkpoint_path=args.checkpoint,
        target_col=args.target_col,
        time_col=args.time_col,
        save_path_predictions=args.save_path_predictions,
        save_path_pickle=args.save_path_pickle,
        plot_path=args.plot_path,
        show_plots=args.show_plots,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
