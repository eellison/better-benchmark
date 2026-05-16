# Repro Quality Issues

Generated board of repro-generator defects. Separates repro defects from
compiler-performance issues.

## Current Issue Counts

| Class | Count | Shimmable? |
|-------|-------|------------|
| `undefined_device_symbol` | 28 | Yes (loader shim) |
| `random_index_input_needs_bounds` | 16 | No (generator fix) |
| `randn_bool_input` | 9 | Yes (loader shim) |
| `undefined_inf_symbol` | 8 | Yes (loader shim) |
| `unsupported_prims_fma` | 3 | No (generator fix) |
| `undefined_symbolic_size` | 1 | No (generator fix) |

## Shimmable Loader Issues

These are handled at load time by `bench.py` and `repro_harness.py`:

- **`undefined_device_symbol`**: Generated code uses bare `device(...)` without
  importing or defining it. Shim: `device = torch.device`.
- **`undefined_inf_symbol`**: Generated code references `inf` or `nan` without
  import. Shim: `inf = math.inf`, `nan = math.nan`.
- **`randn_bool_input`**: `torch.randn(..., dtype=torch.bool)` is invalid.
  Shim: replace with `torch.randint(0, 2, ..., dtype=torch.bool)`.

## Hard Blockers (Require Regeneration)

These cannot be fixed at load time and require generator-level fixes:

- **`random_index_input_needs_bounds`**: Integer inputs used by `embedding`,
  `gather`, `scatter`, `index`, `index_put`, or `index_select` need
  graph-aware bounds. Random values cause index-out-of-bounds errors.
- **`unsupported_prims_fma`**: Generated code calls
  `torch.ops.prims.fma.default` which is not available in this local PyTorch
  checkout. Needs lowering to `mul + add` or `aten.addcmul`.
- **`undefined_symbolic_size`**: `make_inputs()` references symbolic sizes that
  were never materialized to concrete values.
- **`missing_make_inputs`**: Repro file has no `make_inputs()` function.
- **`missing_class_repro`**: Repro file has no `class Repro` definition.

## Generator Fixes Needed

1. Emit `torch.device(...)` or `from torch import device` when generated code
   uses bare `device(...)`.
2. Emit `from math import inf, nan` when constants appear in the graph.
3. Use `torch.randint` or `torch.full` for bool placeholders instead of
   `torch.randn(dtype=torch.bool)`.
4. Propagate graph-aware integer bounds into index tensors, including through
   view-like paths into `embedding`, `gather`, `scatter`, `index_put`.
5. Lower stale `prims.fma` to supported arithmetic (`mul + add` or
   `aten.addcmul`).
6. Materialize concrete symbolic sizes in `make_inputs()` when dynamic symbols
   appear in generated factories.

## Triage Rule

Invalid inputs and stale ops **block** compiler-performance triage. Do not
report a kernel as "slow" if the repro itself doesn't run correctly. Regenerate
the repro first.
