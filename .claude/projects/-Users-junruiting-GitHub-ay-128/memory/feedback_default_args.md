---
name: Default argument policy
description: Package-level code may use documented defaults; business/lab code must not use defaults — all arguments explicit at call site
type: feedback
---

Package-level functions/factories/classes (`ugdatalab/`) may have default parameter values, but every default must be clearly documented (what, why, when to override). Defaults encoding scientific choices (e.g., `bad_bits`, `degree`, `period_min`) need the most documentation.

Business code (`labs/NN/` plotters, notebook helpers, single-use functions) must not have default parameter values unless absolutely necessary for functional reuse. The notebook call site should be fully self-documenting — you read it and know exactly what's happening without chasing defaults.

**Why:** Defaults in business code hide what each notebook is doing differently. A plotter called from 3 notebooks is still business code (single-project, not single-use) — defaults there obscure the analysis. Explicit arguments make notebooks readable as standalone records of what was computed.

**How to apply:** When writing or auditing `ugdatalab/` code, check that defaults are documented. When writing or auditing `labs/NN/` code, flag any function signature with defaults and make the argument explicit at every call site instead. Module-level structural constants (`_FIGURES_DIR`, `savefig`) are not function defaults and are fine in both layers.

Codified in: `CLAUDE.md` (audit checklist items 19 for ugdatalab, 11-12 for labs), `FRAMEWORK.md` ("Default arguments" section under Layer 2).
