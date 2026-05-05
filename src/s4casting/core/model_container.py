# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


class ModelContainer:
    """Container for the model, optionally wrapped in DDP."""

    def __init__(self, model: nn.Module, ddp: DDP | None = None) -> None:
        """Initialize the ModelContainer.

        Args:
            model (nn.Module): The model.
            ddp (DDP | None): The DDP wrapped model, if applicable.
        """
        self._model = model
        self._ddp = ddp

    @property
    def raw_model(self) -> nn.Module:
        """Get the raw model, regardless of DDP wrapping.

        Returns:
            nn.Module: The raw model.
        """
        return self._model

    @property
    def model(self) -> nn.Module | DDP:
        """Get the model, wrapped in DDP if applicable.

        Returns:
            nn.Module | DDP: The model, possibly wrapped in DDP.
        """
        return self._ddp or self._model

    @property
    def ddp(self) -> DDP | None:
        """Get the DDP wrapped model, if applicable.

        Returns:
            DDP | None: The DDP wrapped model, if applicable.
        """
        return self._ddp
