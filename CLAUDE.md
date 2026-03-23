# Project: ay-128 (ugdatalab)

## Audit: "audit methods and models"

When the user says "audit methods and models", perform the following checks on all `.py` files in `ugdatalab/methods/` and `ugdatalab/models/` (recursively). Also verify imports from consumer files (`ugdatalab/__init__.py`, `ugdatalab/deoutlier.py`, `ugdatalab/dust.py`, `ugdatalab/relations.py`, `ugdatalab/artifacts.py`, `ugdatalab/lab1_plotter.py`).

### Checklist

1. **Dead code** — functions/classes defined but never called or imported
2. **Stale imports** — importing things that don't exist or aren't used
3. **Return type consistency** — in-place mutation functions must return `None`; functions that produce new data must return it
4. **Defensive guards** — no unnecessary `if name in colnames` checks outside `_sanitize_table`; no empty-table guards
5. **Hardcoded values** — no unexplained magic numbers; data-driven defaults where possible
6. **Cross-references** — all import paths resolve; no references to deleted/renamed modules
7. **ABC conformance** — every concrete class implements all abstract methods from its base
8. **Naming** — private building blocks prefixed `_`; public methods match what engines consume
9. **Docstring accuracy** — no references to removed parameters, wrong types, or stale backend names
10. **Export consistency** — `__init__.py` files export exactly what's used externally
11. **Duplicate code** — flag identical logic that could be shared
12. **Convention consistency** — schemas use `type: [colnames]` format; prior scales are data-driven

### Output format

For each file: list issues or say "clean". End with `import ugdatalab` verification.
