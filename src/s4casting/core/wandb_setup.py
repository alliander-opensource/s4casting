# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import logging
import os
import stat
import tempfile
from pathlib import Path


def setup_wandb(wandb_root: str):
    """Set up Weights & Biases (W&B) directories and environment variables for SageMaker."""
    if not wandb_root:
        return
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("wandb_setup")

    # 2. Create a dictionary of all required directories
    directories = {
        "root": wandb_root,
        "runs": f"{wandb_root}/runs",
        "runs_wandb": f"{wandb_root}/runs/wandb",  # This is critical - W&B creates this nested structure
        "cache": f"{wandb_root}/cache",
        "config": f"{wandb_root}/config",
        "temp": f"{wandb_root}/temp",
    }

    # 3. Create all directories with full permissions
    for name, directory in directories.items():
        try:
            Path(directory).mkdir(parents=True)
            # Set full permissions (read/write/execute for all users)
            Path(directory).chmod(stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

            # Verify write access with a test file
            test_file = Path(directory) / ".test_write"
            with Path.open(test_file, "w") as f:
                f.write("test")
            Path.unlink(test_file)
            logger.info(f"✓ Created and verified directory: {directory}")
        except FileExistsError:
            pass

    # 4. Set all environment variables
    os.environ["WANDB_DIR"] = directories["runs"]
    os.environ["WANDB_CACHE_DIR"] = directories["cache"]
    os.environ["WANDB_CONFIG_DIR"] = directories["config"]
    os.environ["WANDB_TEMP"] = directories["temp"]
    os.environ["TMPDIR"] = directories["temp"]
    os.environ["TEMP"] = directories["temp"]
    os.environ["TMP"] = directories["temp"]

    # 5. Configure Python's tempfile module
    tempfile.tempdir = directories["temp"]

    logger.info("W&B directories successfully configured!")
    logger.info(f"WANDB_DIR = {os.environ['WANDB_DIR']}")
    logger.info(f"WANDB_CONFIG_DIR = {os.environ['WANDB_CONFIG_DIR']}")
    logger.info(f"WANDB_TEMP = {os.environ['WANDB_TEMP']}")
