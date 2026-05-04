"""Lazy downloaders for external data archives.

Datasets are returned from ``~/.cache/ugdatalab/<name>/`` if present,
otherwise downloaded from the upstream URL and cached on disk. Each
cached directory carries a ``.done`` sentinel; directories without it
are treated as partial and re-fetched.
"""

import shutil
import tarfile
from pathlib import Path

import requests
from tqdm.auto import tqdm

_GAIA_ZP_NAME = "GaiaEDR3_passbands_zeropoints_version2"
_GAIA_ZP_BASE = "https://cdsarc.cds.unistra.fr/ftp/J/A+A/649/A3/"
_GAIA_ZP_FILES = ("zeropt.dat", "passband.dat", "ReadMe")
_GAIA_ZP_PROGRESS_PREFIX = "Gaia ZP"

_MIST_NAME = "MIST_v2.5_vvcrit0.0_full_isos"
_MIST_URL = f"https://mist.science/data/tarballs_v2.5/isos/{_MIST_NAME}.txz"
_MIST_PROGRESS_DESC = "MIST v2.5 isochrones (~7.2 GB)"

# 1 MiB streaming chunk size.
_DOWNLOAD_CHUNK_BYTES = 1 << 20

# HTTP timeout for downloads of unknown size.
_DOWNLOAD_TIMEOUT_S = 600


_CACHE_ROOT = Path.home() / ".cache" / "ugdatalab"


def _cached_dir(name: str) -> Path | None:
    """Return the user-cache directory if already populated, else ``None``."""
    cached = _CACHE_ROOT / name
    if (cached / ".done").exists():
        return cached
    return None


def _stream_download(url: str, dest: Path, desc: str) -> None:
    """Stream-download *url* to *dest* with a tqdm progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT_S) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0)) or None
        with open(dest, "wb") as handle, tqdm(
            total=total, unit="B", unit_scale=True, desc=desc
        ) as bar:
            for chunk in response.iter_content(_DOWNLOAD_CHUNK_BYTES):
                handle.write(chunk)
                bar.update(len(chunk))


def _reset_dir(path: Path) -> None:
    """Remove *path* if it exists, then recreate it as an empty directory."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _fetch_files(
    base_url: str, filenames: tuple[str, ...], name: str, desc_prefix: str,
) -> Path:
    """Download *filenames* from *base_url* into a fresh cache directory for *name*."""
    target = _CACHE_ROOT / name
    _reset_dir(target)
    for fname in filenames:
        _stream_download(base_url + fname, target / fname, f"{desc_prefix}: {fname}")
    (target / ".done").touch()
    return target


def _fetch_tarball(url: str, name: str, desc: str) -> Path:
    """Download and extract the tarball at *url* into a fresh cache directory for *name*, stripping any single shared top-level prefix."""
    target = _CACHE_ROOT / name
    _reset_dir(target)

    archive = _CACHE_ROOT / f"{name}.txz"
    _stream_download(url, archive, desc)
    with tarfile.open(archive) as tf:
        members = tf.getmembers()
        first = members[0].name.split("/", 1)[0] if members else ""
        has_top_level_dir = bool(first) and all(
            m.name == first or m.name.startswith(first + "/")
            for m in members
        )
        if has_top_level_dir:
            tf.extractall(_CACHE_ROOT)
            extracted = _CACHE_ROOT / first
            if extracted.resolve() != target.resolve():
                target.rmdir()
                extracted.rename(target)
        else:
            tf.extractall(target)
    archive.unlink()
    (target / ".done").touch()
    return target


def gaia_zeropoints_dir() -> Path:
    """Return the directory containing the Gaia EDR3 passband zeropoint files.

    Looks first in the user cache (``~/.cache/ugdatalab/``); on a miss,
    downloads ``zeropt.dat`` and ``ReadMe`` from the CDS archive and
    writes them into the cache.

    Returns
    -------
    Path
        Cache directory containing the zeropoint files.

    Raises
    ------
    requests.HTTPError
        If a download fails on a cold cache.
    """
    cached = _cached_dir(_GAIA_ZP_NAME)
    if cached is not None:
        return cached
    return _fetch_files(_GAIA_ZP_BASE, _GAIA_ZP_FILES, _GAIA_ZP_NAME, _GAIA_ZP_PROGRESS_PREFIX)


def mist_isochrones_dir() -> Path:
    """Return the directory containing the MIST v2.5 vvcrit0.0 isochrone grid.

    Looks first in the user cache (``~/.cache/ugdatalab/``); on a miss,
    downloads and extracts the ~7.2 GB MIST tarball into the cache. Cold-
    cache calls block on a multi-minute network download.

    Returns
    -------
    Path
        Cache directory containing the extracted ``.iso`` files.

    Raises
    ------
    requests.HTTPError
        If the tarball download fails on a cold cache.
    """
    cached = _cached_dir(_MIST_NAME)
    if cached is not None:
        return cached
    return _fetch_tarball(_MIST_URL, _MIST_NAME, _MIST_PROGRESS_DESC)
