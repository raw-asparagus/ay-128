"""Joblib-backed disk cache decorator shared across query loaders."""

from joblib import Memory


_memory = Memory(".joblib-cache", verbose=0)


def cache_stable(*, module: str):
    """Return a decorator that wraps a function in the shared joblib disk cache.

    Parameters
    ----------
    module : str
        Value assigned to the wrapped function's ``__module__`` attribute,
        which joblib incorporates into the cache key.

    Returns
    -------
    callable
        Decorator that sets ``__module__`` on its argument and returns the
        joblib-cached wrapper.
    """
    def _deco(fn):
        fn.__module__ = module
        return _memory.cache(fn)
    return _deco
