# Project: ay-128 (ugdatalab)

## Audit: "audit ugdatalab"

When the user says "audit ugdatalab", perform the following checks on all `.py` files in `ugdatalab/` (recursively), including `ugdatalab/methods/`, `ugdatalab/models/`, and `ugdatalab/plotting.py`.

### Checklist

1. **Dead code** — functions/classes defined but never called or imported
2. **Stale imports** — importing things that don't exist or aren't used
3. **Lazy imports** — no function-level imports; all imports at top of file
4. **Return type consistency** — in-place mutation functions must return `None`; functions that produce new data must return it
5. **Defensive guards** — no unnecessary guards (e.g., `if len(x) == 0` when input is guaranteed non-empty)
6. **Hardcoded values** — no unexplained magic numbers; data-driven defaults where possible
7. **Cross-references** — all import paths resolve; no references to deleted/renamed modules
8. **ABC conformance** — every concrete class implements all abstract methods from its base
9. **Naming** — private building blocks prefixed `_`; public methods match what engines consume; functions used only within the same file must be private; internal decorators and helpers must not be exported from `__init__.py`
10. **Docstring accuracy** — no references to removed parameters, wrong types, or stale descriptions
11. **Export consistency** — `__init__.py` files export exactly what's used externally; no private symbols (`_`-prefixed) exported; no internal utilities (e.g., caching decorators) in public API
12. **Duplicate code** — flag identical logic that could be shared
13. **Convention consistency** — schemas use `type: [colnames]` format; prior scales are data-driven; style dicts used for plot styling
14. **Redundant rcParams** — no rcParams that match matplotlib defaults
15. **Redundant type casting** — no `np.asarray(col, dtype=...)` on columns already sanitized by `_sanitize_table`; no casting of return values from libraries that already return numpy arrays (e.g., `LombScargle.autopower`)

### Output format

For each file: list issues or say "clean". End with `import ugdatalab` verification.
