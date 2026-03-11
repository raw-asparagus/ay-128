from joblib import Memory


_memory = Memory(".joblib-cache", verbose=0)


def _cache_stable(func=None, *, module: str | None = None):
    """Cache a function in the shared `.joblib-cache` store."""

    def deco(inner_func):
        if module is not None:
            inner_func.__module__ = module
        return _memory.cache(inner_func)

    if func is None:
        return deco
    return deco(func)
