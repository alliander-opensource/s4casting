# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

from torch.nn.parallel import DistributedDataParallel as DDP

from s4casting.model.ss import SSModel


class ModelContainer:
    """Container for the model, optionally wrapped in DDP."""

    def __init__(self, model: SSModel, ddp: DDP | None = None) -> None:
        """Initialize the ModelContainer.

        Args:
            model (SSModel): The S4 model.
            ddp (DDP | None): The DDP wrapped model, if applicable.
        """
        self._model = model
        self._ddp = ddp

    @property
    def raw_model(self) -> SSModel:
        """Get the raw S4 model, regardless of DDP wrapping.

        Returns:
            SSModel: The raw S4 model.
        """
        return self._model

    @property
    def model(self) -> SSModel | DDP:
        """Get the model, wrapped in DDP if applicable.

        Returns:
            SSModel | DDP: The model, possibly wrapped in DDP.
        """
        return self._ddp or self._model

    @property
    def ddp(self) -> DDP | None:
        """Get the DDP wrapped model, if applicable.

        Returns:
            DDP | None: The DDP wrapped model, if applicable.
        """
        return self._ddp
