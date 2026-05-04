"""Shared loader scaffold for catalog-style data classes.

Defines the :class:`Data` abstract base that every model namespace's
loader (``GaiaData``, ``WISEData``, ``APOGEEData``, ``GalaxyZooData``)
inherits from. The base owns the ``pipeline`` field, the ``__len__``
shortcut, and the lifecycle:

    fetch → sanitize → pipeline → post-pipeline

so all four namespaces apply quality cuts and column augmentations the
same way and any future loader picks up the same shape "for free."
Subclasses provide the source-specific fetch and sanitize
implementations, may declare ``_required_stages`` to auto-prepend
source-mandated transforms (e.g., attaching derived columns the cut
classes assume), and may override the optional ``_post_pipeline`` hook.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ugdatalab.methods.compose import Compose


@dataclass
class Data(ABC):
    """Abstract loader template parameterized by a transform pipeline.

    The ``pipeline`` parameter is **keyword-only** so subclasses can add
    positional fields (e.g., ``query``, ``csv_path``) without colliding
    with the inherited default. ``data`` is populated by ``__post_init__``
    via the four-step lifecycle and is not an init argument.

    Subclasses may set the class-level ``_required_stages`` attribute to
    a tuple of source-mandated callables that are auto-prepended to the
    user-supplied pipeline. This is how loaders enforce invariants such
    as "every Gaia table has the RR Lyrae representative-period column"
    without forcing every caller to include the stage explicitly.

    Parameters
    ----------
    pipeline : Compose
        Pipeline of cuts and column augmentations applied immediately
        after fetch + sanitize. Default ``Compose([])`` — no
        transformations beyond ``_required_stages``.

    Attributes
    ----------
    data : Any
        The loaded, sanitized, augmented, and filtered dataset. Concrete
        subclasses narrow the type (e.g., ``astropy.table.Table`` for
        Gaia/WISE/APOGEE; ``pandas.DataFrame`` for Galaxy Zoo).
    """
    _required_stages: ClassVar[tuple[Callable[[Any], Any], ...]] = ()
    pipeline: Compose = field(default_factory=lambda: Compose([]), kw_only=True)
    data: Any = field(init=False, repr=False)

    def __post_init__(self):
        """Run the fetch, sanitize, pipeline, and post-pipeline lifecycle."""
        raw = self._fetch()
        self._sanitize(raw)
        effective = Compose(list(self._required_stages) + list(self.pipeline.steps))
        self.data = effective(raw)
        self._post_pipeline()

    @abstractmethod
    def _fetch(self) -> Any:
        """Return the raw, unsanitized dataset from the underlying source."""
        ...

    @abstractmethod
    def _sanitize(self, raw: Any) -> None:
        """Coerce types in place to satisfy this class's schema."""
        ...

    def _post_pipeline(self) -> None:
        """Run any secondary work after the pipeline has been applied (default: no-op)."""
        pass

    def __len__(self) -> int:
        """Return the number of rows in ``self.data``."""
        return len(self.data)
