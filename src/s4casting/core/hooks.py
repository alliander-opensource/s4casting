# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

from collections.abc import Callable
from typing import TypeVar

from plotly.graph_objects import Figure

from s4casting.core.context import Context

T = TypeVar("T", bound=Callable)


class Hook[T: Callable]:
    """Hook system to register and call functions at specific events."""

    def __init__(self, name: str):
        """Initialize the Hook.

        Args:
            name (str): Name of the hook.
        """
        self.name = name
        self._functions: list[T] = []

    @property
    def call(self) -> T:
        """Get a callable that invokes all registered functions."""
        return self

    def __call__(self, *args, **kwargs):
        """Invoke all registered functions with the given arguments."""
        for f in self._functions:
            f(*args, **kwargs)

    def register(self, f: T):
        """Register a function to the hook.

        Args:
            f (T): Function to register.
        """
        self._functions.append(f)

    def unregister(self, f: T):
        """Unregister a function from the hook.

        Args:
            f (T): Function to unregister.
        """
        self._functions.remove(f)

    def clear(self) -> None:
        """Clear all registered functions."""
        self._functions.clear()


class CommonHooks:
    """Common hooks for the framework."""

    def __init__(self) -> None:
        """Initialize the CommonHooks."""
        self.start: Hook[Callable[[Context], None]] = Hook("start")
        self.finished: Hook[Callable[[Context], None]] = Hook("finished")


class TrainingHooks(CommonHooks):
    """Training-specific hooks for the framework."""

    def __init__(self) -> None:
        """Initialize the TrainingHooks."""
        super().__init__()
        self.step: Hook[Callable[[Context, int], None]] = Hook("step")
        self.epoch: Hook[Callable[[Context, int], None]] = Hook("epoch")
        self.evaluate: Hook[Callable[[Context, int], None]] = Hook("evaluate")
        self.checkpoint: Hook[Callable[[Context, int], None]] = Hook("checkpoint")
        self.benchmark: Hook[Callable[[Context, int], None]] = Hook("benchmark")
        self.stef_beam: Hook[Callable[[Context, int], None]] = Hook("stef_beam")
        self.benchmark_metrics: Hook[Callable[[Context, int], None]] = Hook("benchmark_metrics")
        self.eval_plot: Hook[Callable[[Context, Figure], None]] = Hook("eval_plot")
        self.benchmark_plot: Hook[Callable[[Context, Figure, str], None]] = Hook("benchmark_plot")
        self.benchmark_complete: Hook[Callable[[Context, int], None]] = Hook("benchmark_complete")
