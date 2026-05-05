"""Generic sequential pipeline of callables.

Used to compose data-cleaning quality cuts (``Table -> Table``) and image
augmentation transforms (``ndarray -> ndarray``) with the same idiom.
"""

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Compose(Generic[T]):
    """Sequentially apply a list of callables, threading the output through.

    Parameters
    ----------
    steps : list of callables
        Each callable takes a value of type ``T`` and returns one of the
        same type. Run in order on each call; ``Compose([])`` is the
        identity.
    """
    def __init__(self, steps: list[Callable[[T], T]]):
        """Store the ordered list of step callables to apply on each call."""
        self.steps = steps

    def __call__(self, value: T) -> T:
        """Apply each step in order to *value* and return the final result."""
        for step in self.steps:
            value = step(value)
        return value
