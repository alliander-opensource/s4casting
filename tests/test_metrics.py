# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

from unittest.mock import MagicMock

# Adjust the import path if your classes are in a different module
from s4casting.eval.evaluator_heads import EvaluatorHead


def test_head_evaluator_initialization():
    """Tests that EvaluatorHead subclasses correctly calculate their properties upon initialization."""
    # --- Configuration ---
    mock_hooks = MagicMock()

    # --- Test Both Evaluator Classes ---
    evaluator = EvaluatorHead(
        head_type="GMM",
        hookable=mock_hooks,
    )

    # --- Assertions ---
    assert evaluator.head_type == "GMM"
