from dataclasses import dataclass, field
from pathlib import Path

from astroquery.gaia import Gaia
from astropy import table
from astropy.io import ascii
import matplotlib.pyplot as plt
import numpy as np
from joblib import Memory

# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _cache_stable(memory: Memory, module: str = '01'):
    def deco(func):
        func.__module__ = module
        return memory.cache(func)
    return deco

_memory = Memory(".joblib-cache")

# ---------------------------------------------------------------------------
# Zero-point data (resolved relative to this file)
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent.parent
ZP_DIR    = _HERE / "GaiaEDR3_passbands_zeropoints_version2"
ZP_README = str(ZP_DIR / "ReadMe")

ZEROPT         = ascii.read(str(ZP_DIR / "zeropt.dat"), readme=ZP_README)
ZEROPT_VEGAMAG = ZEROPT[ZEROPT["System"] == "VEGAMAG"]

ZP_G     = float(ZEROPT_VEGAMAG["GZp"][0])
ZP_ERR_G = float(ZEROPT_VEGAMAG["e_GZp"][0])

# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

@_cache_stable(_memory)
def get_gaia(query):
    job     = Gaia.launch_job_async(query)
    results = job.get_results()
    return results


@_cache_stable(_memory)
def fetch_gaia_data(query):
    raw  = get_gaia(query)
    poe  = raw["parallax_over_error"]
    b    = raw["b"]
    data = raw[(poe > 5) & (np.abs(b) > 30)]

    sigma_G_meas    = (2.5 / np.log(10)) * np.abs(
        data["phot_g_mean_flux_error"] / data["phot_g_mean_flux"]
    )
    data["sigma_G"] = np.sqrt(sigma_G_meas**2 + ZP_ERR_G**2)

    omega            = data["parallax"]
    data["mu"]       = 10 - 5 * np.log10(omega)
    data["sigma_mu"] = 5 * data["parallax_error"] / (omega * np.log(10))
    data["M_G"]      = data["phot_g_mean_mag"] - data["mu"]
    data["sigma_M"]  = np.sqrt(data["sigma_G"]**2 + data["sigma_mu"]**2)

    pf   = data["pf"].filled(np.nan)
    p1_o = data["p1_o"].filled(np.nan)

    data["period"]  = np.where(np.isfinite(pf), pf, p1_o)
    data["is_rrab"] = np.isfinite(pf)  & ~np.isfinite(p1_o)
    data["is_rrc"]  = ~np.isfinite(pf) &  np.isfinite(p1_o)
    return data

# ---------------------------------------------------------------------------
# GaiaData and subclasses
# ---------------------------------------------------------------------------

@dataclass
class GaiaData:
    """Fetches and caches Gaia data, with plotting helpers."""
    query: str
    data: table.Table = field(init=False, repr=False)

    def __post_init__(self):
        self.data = fetch_gaia_data(self.query)

    @classmethod
    def from_table(cls, query: str, data: table.Table) -> "GaiaData":
        """Construct a GaiaData from an already-fetched table (no network call)."""
        obj = object.__new__(cls)
        obj.query = query
        obj.data  = data
        return obj

    def plot_mollweide(self, ax=None, title=None, **scatter_kwargs):
        """
        Plot stars in a Mollweide projection using galactic (l, b) coordinates.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            A pre-existing Mollweide axes to draw on. If None, a new figure
            and axes are created. Returning the axes allows child classes and
            callers to add overlays before calling plt.show().
        title : str, optional
            Axes title.
        show_band : bool, optional
            If True (default), shade the |b| < 30° galactic plane band.
        **scatter_kwargs
            Forwarded to ax.scatter (e.g. c=, cmap=, s=).

        Returns
        -------
        ax : matplotlib Axes
        """
        l_wrap = np.where(self.data["l"] > 180, self.data["l"] - 360, self.data["l"])
        l_rad  = np.deg2rad(l_wrap)
        b_rad  = np.deg2rad(self.data["b"])

        if ax is None:
            fig = plt.figure(figsize=(10, 5), dpi=300)
            ax  = fig.add_subplot(111, projection="mollweide")

        ax.scatter(l_rad, b_rad, **scatter_kwargs)

        longs  = np.linspace(-np.pi, np.pi, 1000)
        lat_lo = np.full_like(longs, np.deg2rad(-30))
        lat_hi = np.full_like(longs, np.deg2rad(30))
        ax.fill_between(
            longs, lat_lo, lat_hi,
            color="C3", alpha=0.1, label=r"$|b| < 30^\circ$ band",
        )

        ax.set_xlabel("Galactic longitude $l$")
        ax.set_ylabel("Galactic latitude $b$")
        if title is not None:
            ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        return ax

    def plot_mollweide_diff(self, subset: "GaiaData", ax=None, title=None, **scatter_kwargs):
        """
        Mollweide sky map showing kept vs. removed stars.

        Parameters
        ----------
        subset : GaiaData
            A subset of self (e.g. after a quality cut or outlier rejection).
            Stars in subset are plotted in blue ("Kept"); stars in self but not
            in subset are plotted in orange ("Removed").
        ax : matplotlib Axes, optional
            A pre-existing Mollweide axes to draw on. If None, a new figure
            and axes are created.
        title : str, optional
            Axes title.
        **scatter_kwargs
            Forwarded to both scatter calls (e.g. s=, alpha=, rasterized=).

        Returns
        -------
        ax : matplotlib Axes
        """
        diff_mask = ~np.isin(self.data["source_id"], subset.data["source_id"])
        n_diff    = int(diff_mask.sum())
        n_sub     = len(subset.data)

        if ax is None:
            fig = plt.figure(figsize=(10, 5), dpi=300)
            ax  = fig.add_subplot(111, projection="mollweide")

        scatter_kwargs.setdefault("s", 5)
        scatter_kwargs.setdefault("alpha", 0.6)
        scatter_kwargs.setdefault("rasterized", True)

        def to_rad(data):
            l_wrap = np.where(data["l"] > 180, data["l"] - 360, data["l"])
            return np.deg2rad(l_wrap), np.deg2rad(data["b"])

        diff_data = self.data[diff_mask]
        l_diff, b_diff = to_rad(diff_data)
        ax.scatter(l_diff, b_diff,
                   color="C1", label=f"Removed ($N$={n_diff})", zorder=1,
                   **scatter_kwargs)

        l_sub, b_sub = to_rad(subset.data)
        ax.scatter(l_sub, b_sub,
                   color="C0", label=f"Kept ($N$={n_sub})", zorder=2,
                   **scatter_kwargs)

        longs  = np.linspace(-np.pi, np.pi, 1000)
        lat_lo = np.full_like(longs, np.deg2rad(-30))
        lat_hi = np.full_like(longs, np.deg2rad(30))
        ax.fill_between(longs, lat_lo, lat_hi,
                        color="C3", alpha=0.1, label=r"$|b| < 30^\circ$ band")

        ax.set_xlabel("Galactic longitude $l$")
        ax.set_ylabel("Galactic latitude $b$")
        if title is not None:
            ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        return ax

    def plot_period_abs_mag(self, ax=None, title=None, **scatter_kwargs):
        rrc  = np.asarray(self.data["is_rrc"],  dtype=bool)
        rrab = ~rrc

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

        scatter_kwargs.setdefault("s", 5)
        scatter_kwargs.setdefault("alpha", 0.6)
        scatter_kwargs.setdefault("rasterized", True)

        M_G     = np.asarray(self.data["M_G"])
        sigma_M = np.asarray(self.data["sigma_M"])

        ax.errorbar(self.data["period"], M_G, yerr=sigma_M,
                    fmt="none", color="gray", alpha=0.3, zorder=1)
        ax.scatter(self.data["period"][rrab], M_G[rrab], label="RRab", zorder=2, **scatter_kwargs)
        ax.scatter(self.data["period"][rrc],  M_G[rrc],  label="RRc",  zorder=2, **scatter_kwargs)

        ax.set_xscale("log")
        ax.invert_yaxis()
        ax.set_xlabel(r"$P$ [days]")
        ax.set_ylabel(r"$M_G$ [mag]")
        if title is not None:
            ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_period_luminosity_diff(self, subset: "GaiaData", ax=None, title=None, **scatter_kwargs):
        """
        Plot period–luminosity diagram showing kept vs. removed stars.

        Parameters
        ----------
        subset : GaiaData
            A subset of self (e.g. after a quality cut). Stars in subset are
            plotted in blue ("Kept"); stars in self but not in subset are
            plotted in orange ("Removed").
        ax : matplotlib Axes, optional
            Pre-existing axes to draw on. If None, a new figure is created.
        title : str, optional
            Axes title.
        **scatter_kwargs
            Forwarded to both scatter calls (e.g. s, alpha, rasterized).

        Returns
        -------
        ax : matplotlib Axes
        """
        diff_mask = ~np.isin(self.data["source_id"], subset.data["source_id"])
        n_diff    = int(diff_mask.sum())
        n_sub     = len(subset.data)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

        scatter_kwargs.setdefault("s", 5)
        scatter_kwargs.setdefault("alpha", 0.6)
        scatter_kwargs.setdefault("rasterized", True)

        diff_data = self.data[diff_mask]
        ax.scatter(
            diff_data["period"], np.asarray(diff_data["M_G"]),
            color="C1", label=f"Removed ($N$={n_diff})", zorder=1,
            **scatter_kwargs,
        )
        ax.scatter(
            subset.data["period"], np.asarray(subset.data["M_G"]),
            color="C0", label=f"Kept ($N$={n_sub})", zorder=2,
            **scatter_kwargs,
        )

        ax.set_xscale("log")
        ax.invert_yaxis()
        ax.set_xlabel(r"$P$ [days]")
        ax.set_ylabel(r"$M_G$ [mag]")
        if title is not None:
            ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_hr(self, ax=None, title=None, **scatter_kwargs):
        rrc  = np.asarray(self.data["is_rrc"],  dtype=bool)
        rrab = ~rrc

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 7), dpi=300)

        scatter_kwargs.setdefault("s", 5)
        scatter_kwargs.setdefault("alpha", 0.6)
        scatter_kwargs.setdefault("rasterized", True)

        ax.scatter(self.data["bp_rp"][rrab], np.asarray(self.data["M_G"])[rrab], label="RRab", **scatter_kwargs)
        ax.scatter(self.data["bp_rp"][rrc],  np.asarray(self.data["M_G"])[rrc],  label="RRc",  **scatter_kwargs)

        ax.invert_yaxis()
        ax.set_xlabel(r"$G_\mathrm{BP} - G_\mathrm{RP}$ [mag]")
        ax.set_ylabel(r"$M_G$ [mag]")
        if title is not None:
            ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax


class Local(GaiaData):
    def __init__(self, source: GaiaData):
        self.query = source.query
        self.data  = source.data[source.data["parallax"] > 0.25]


class StrictGBPRP(GaiaData):
    """Strict BP/RP signal-to-noise cut.

    Accepts sources satisfying:
      phot_g_mean_flux_over_error > 5
      phot_bp_mean_flux_over_error > 5
      phot_rp_mean_flux_over_error > 5
    """
    def __init__(self, source: GaiaData):
        self.query = source.query
        mask       = (
            (source.data["phot_g_mean_flux_over_error"] > 5) &
            (source.data["phot_bp_mean_flux_over_error"] > 5) &
            (source.data["phot_rp_mean_flux_over_error"] > 5)
        )
        self.data  = source.data[mask]


class Cut1(GaiaData):
    """RUWE quality cut.

    Accepts sources satisfying:
      ruwe < 1.2 * max(1, exp(-0.2 * (G - 19.5)))
    """
    def __init__(self, source: GaiaData):
        self.query = source.query
        G          = np.asarray(source.data["phot_g_mean_mag"])
        u_max      = 1.2 * np.maximum(1, np.exp(-0.2 * (G - 19.5)))
        mask       = np.asarray(source.data["ruwe"]) < u_max
        self.data = source.data[mask]


class Cut2(GaiaData):
    """BP/RP flux excess factor quality cut.

    Accepts sources satisfying:
      1.0 + 0.015*(bp_rp)^2 < phot_bp_rp_excess_factor < 1.3 + 0.06*(bp_rp)^2
    """
    def __init__(self, source: GaiaData):
        self.query = source.query
        bp_rp      = np.asarray(source.data["bp_rp"])
        E          = np.asarray(source.data["phot_bp_rp_excess_factor"])
        mask       = (
            (E > 1.0 + 0.015 * bp_rp**2) &
            (E < 1.3 + 0.06  * bp_rp**2)
        )
        self.data = source.data[mask]
