"""Gaia/AllWISE pipeline stages: row-filter cuts and column augmentations.

Each stage is a small ``__call__``-able (dataclass when parameterized)
that takes an ``astropy.table.Table`` and returns one. Compose with
:class:`~ugdatalab.methods.compose.Compose` to build pipelines that
:class:`~ugdatalab.models.gaia.wise.WISEData` runs immediately after
fetching and sanitizing.

Shared parallax/latitude cuts (``HighParallaxSnrCut``,
``HighLatitudeCut``) live in :mod:`ugdatalab.models.gaia.pipeline`
and are reused at the call site.
"""

from dataclasses import dataclass

import numpy as np
from astropy import table


def _char_at(column, index: int) -> np.ndarray:
    """Extract the character at *index* from each string in *column*."""
    return np.array([s[index] if len(s) > index else "" for s in np.asarray(column, dtype=str)])


# ---------------------------------------------------------------------------
# WISE column augmentation
# ---------------------------------------------------------------------------


class AddW2PhotometryColumns:
    """Attach W2 absolute-magnitude and distance-modulus columns.

    Adds ``mu``, ``sigma_mu``, ``M_W2``, ``sigma_M_W2`` derived from the
    AllWISE ``w2mpro`` and Gaia ``parallax``. Mutates the table in place
    and returns it.
    """

    def __call__(self, data: table.Table) -> table.Table:
        """Attach W2 photometry columns and return the same table."""
        omega = data["parallax"]
        data["mu"] = 10 - 5 * np.log10(omega)
        data["sigma_mu"] = 5 * data["parallax_error"] / (omega * np.log(10))
        data["M_W2"] = data["w2mpro"] - data["mu"]
        data["sigma_M_W2"] = np.sqrt(data["w2mpro_error"] ** 2 + data["sigma_mu"] ** 2)
        return data


# ---------------------------------------------------------------------------
# WISE-specific row-filter cuts
# ---------------------------------------------------------------------------


class WISEFiniteCut:
    """Drop rows whose AllWISE join produced non-finite or non-positive W2 values.

    Sanity bundle ensuring that downstream WISE code sees only rows
    where the AllWISE counterpart exists (``allwise_oid`` is finite —
    the LEFT OUTER JOIN can return NaN here) and the W2 magnitude plus
    its error are finite with positive error. The WISE analogue of
    :class:`~ugdatalab.models.gaia.pipeline.GaiaFiniteCut`.
    """

    def __call__(self, data: table.Table) -> table.Table:
        """Return rows with a finite AllWISE match and finite/positive W2 photometry."""
        matched = data[np.isfinite(data["allwise_oid"])]
        w2 = matched["w2mpro"]
        w2_error = matched["w2mpro_error"]
        mask = np.isfinite(w2) & np.isfinite(w2_error) & (w2_error > 0)
        return matched[mask]


@dataclass
class MarreseOneToOneMatchCut:
    """One-to-one Gaia–AllWISE cross-match (Marrese et al. 2019, A&A 621, A144).

    Keeps rows whose AllWISE neighbourhood has exactly one Gaia
    counterpart and no rival mates. Discards ambiguous matches where a
    Gaia source has multiple AllWISE candidates (mates > 0) or the
    AllWISE source has multiple Gaia candidates (neighbours > 1) —
    either case makes the W2 photometry unsafe to associate with the
    Gaia astrometry.

    Parameters
    ----------
    max_mates : int
        Maximum allowed ``number_of_mates``. Default ``0`` — strictly
        one-to-one; raise if you accept some Gaia ambiguity in exchange
        for sample size.
    max_neighbours : int
        Maximum allowed ``number_of_neighbours``. Default ``1`` —
        strictly one-to-one on the AllWISE side as well.
    """
    max_mates: int = 0
    max_neighbours: int = 1

    def __call__(self, data: table.Table) -> table.Table:
        """Return rows passing the Marrese 2019 one-to-one match criteria."""
        number_of_mates = data["number_of_mates"]
        number_of_neighbours = data["number_of_neighbours"]
        mask = (
            np.isfinite(number_of_mates)
            & np.isfinite(number_of_neighbours)
            & (number_of_mates <= self.max_mates)
            & (number_of_neighbours <= self.max_neighbours)
        )
        return data[mask]


@dataclass
class W2PhotometryQualityCut:
    """AllWISE W2 photometric-quality flags (Cutri et al. 2013, AllWISE Explanatory Supplement).

    Bundled because all three flags come from the same AllWISE pipeline
    table and tend to be tuned together as one "stricter/looser
    photometry" knob.

    Parameters
    ----------
    allowed_ph_qual : tuple of str
        Allowed W2 ``ph_qual`` grades (the second character of the
        4-character string). Default ``("A", "B")`` — SNR > 3; add
        ``"C"`` to admit SNR > 2 sources, or restrict to ``("A",)`` for
        SNR > 10.
    max_cc_flag_w2 : str
        Maximum allowed W2 ``cc_flags`` character (lexicographic).
        Default ``"0"`` — no artifact contamination at all; raise to
        admit specific persistent / scattered-light artifacts.
    max_ext_flag : int
        Maximum allowed ``ext_flag``. Default ``1`` — point sources
        and marginally extended; raise to admit clearly extended
        sources, drop to ``0`` for strict point-source only.
    """
    allowed_ph_qual: tuple[str, ...] = ("A", "B")
    max_cc_flag_w2: str = "0"
    max_ext_flag: int = 1

    def __call__(self, data: table.Table) -> table.Table:
        """Return rows passing the three W2 AllWISE pipeline-quality flags."""
        ph_qual_w2 = _char_at(data["ph_qual"], 1)
        cc_flags_w2 = _char_at(data["cc_flags"], 1)
        ext_flg = data["ext_flag"]
        mask = (
            np.isin(ph_qual_w2, list(self.allowed_ph_qual))
            & (cc_flags_w2 <= self.max_cc_flag_w2)
            & np.isfinite(ext_flg)
            & (ext_flg <= self.max_ext_flag)
        )
        return data[mask]
