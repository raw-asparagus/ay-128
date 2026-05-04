"""Lomb-Scargle periodogram wrapper with anti-alias best-period selection."""

from dataclasses import dataclass

import numpy as np
from astropy.timeseries import LombScargle


@dataclass(frozen=True)
class PeriodogramResult:
    """Result of a Lomb-Scargle periodogram.

    Attributes
    ----------
    periods : ndarray
        Tested periods, sorted by descending power.
    power : ndarray
        Periodogram power, aligned with ``periods``.
    best_period : float
        Period selected by the anti-alias rule.
    best_power : float
        Peak power.
    fap : float
        False alarm probability of ``best_power``.
    """
    periods: np.ndarray
    power: np.ndarray
    best_period: float
    best_power: float
    fap: float


def lomb_scargle(
    times: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray,
    period_min: float | None = None,
    period_max: float | None = None,
) -> PeriodogramResult:
    """Compute a Lomb-Scargle periodogram and identify the best period.

    When multiple peaks are within 2% of the maximum power, the longest
    period among them is selected.

    Parameters
    ----------
    times, values, errors : array-like
        Observation times, measured values, and their uncertainties.
    period_min, period_max : float, optional
        Search range in the same units as *times*. ``None`` defers to
        the astropy autopower defaults.

    Returns
    -------
    PeriodogramResult
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)

    ls = LombScargle(times, values, errors)
    autopower_kwargs = {}
    if period_max is not None:
        autopower_kwargs["minimum_frequency"] = 1.0 / period_max
    if period_min is not None:
        autopower_kwargs["maximum_frequency"] = 1.0 / period_min
    freqs, power = ls.autopower(**autopower_kwargs)

    periods = 1.0 / freqs
    order = np.argsort(power)[::-1]
    periods = periods[order]
    power = power[order]

    best_power = float(power[0])
    near_max = np.where(power >= 0.98 * best_power)[0]
    best_period = float(periods[near_max[np.argmax(periods[near_max])]])

    fap = float(ls.false_alarm_probability(best_power))

    return PeriodogramResult(
        periods=periods,
        power=power,
        best_period=best_period,
        best_power=best_power,
        fap=fap,
    )
