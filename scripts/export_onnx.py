# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

"""Export an s4casting model to ONNX.

The checkpoint is optional: without one the model is exported with random weights, which
is enough to validate that the graph converts and that the shapes are what a consumer
expects. Re-run with --checkpoint once trained weights are available.
"""

import argparse
import sys
import warnings

from s4casting.inference.onnx_export import export_onnx

warnings.filterwarnings("ignore")


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command line arguments for the ONNX export script.

    Args:
        argv: List of command line arguments. If None, uses sys.argv.

    Returns:
        Parsed arguments namespace.
    """
    ap = argparse.ArgumentParser(description="Export an s4casting model to ONNX.")
    ap.add_argument("--config-path", required=True, help="Path to model config TOML.")
    ap.add_argument("--output", default="out/model.onnx", help="Destination .onnx path.")
    ap.add_argument("--checkpoint", help="Path to checkpoint .pt file (omit to export random weights).")
    ap.add_argument(
        "--dynamic-time",
        action="store_true",
        help="Keep the time axis dynamic, in multiples of the patch size. Batch is always dynamic.",
    )
    ap.add_argument(
        "--external-data",
        action="store_true",
        help="Write weights to a sidecar .onnx.data file instead of one self-contained file.",
    )
    ap.add_argument("--opset", type=int, default=18, help="ONNX opset version to target.")
    ap.add_argument(
        "--quantiles",
        action="store_true",
        help="For a GMM head, add a second 'quantiles' output so consumers need no post-processing.",
    )
    ap.add_argument("--no-verify", action="store_true", help="Skip comparing the ONNX graph against PyTorch.")
    return ap.parse_args(argv)


def main(argv=None):
    """Export the model and report where it was written."""
    args = parse_args(argv)

    path, deviations = export_onnx(
        config=args.config_path,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        dynamic_time=args.dynamic_time,
        external_data=args.external_data,
        opset_version=args.opset,
        with_quantiles=args.quantiles,
        verify=not args.no_verify,
    )
    print(f"[OK] Exported -> {path}")  # noqa: T201

    for batch_size, deviation in deviations.items():
        print(f"[OK] batch={batch_size} max abs deviation vs PyTorch: {deviation:.3e}")  # noqa: T201


if __name__ == "__main__":
    main(sys.argv[1:])
