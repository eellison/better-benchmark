# Standardized Oracle Format

## Motivation

Oracles establish performance floors for each repro: the fastest achievable
kernel for the exact computation the compiled repro performs. Before this
standard, oracles had ad-hoc interfaces (different CLI flags, different output
formats, different ways of loading inputs), making automated validation
unreliable and contributing to 56 bad oracles (scope mismatches, slower than
compile).

This document defines the standard oracle format that all new oracles must
follow and existing oracles should be migrated to.

## Template

Copy `scripts/oracle_template.py` into `repros/canonical/<repro_id>/oracle_<name>.py`
and fill in the marked sections.

## Required Structure

Every oracle file must have:

### 1. Docstring with Gap Diagnosis

```python
"""
Oracle for <repro_id>

Gap diagnosis:
  Classification: <CATEGORY>
  What oracle does differently: <1 sentence>
  What Inductor change would fix: <1 sentence>
"""
```

Valid classifications: `SCHEDULER_FUSION`, `COOPERATIVE_SPLIT_K`,
`SCATTER_REDUCE`, `ONLINE_SOFTMAX`, `MULTI_OUTPUT_REDUCTION`,
`SPLIT_THRESHOLD`, `ALGEBRAIC_ELIMINATION`, `DEFERRED_ASSERTIONS`,
`BANDWIDTH_BOUND` (for at-floor oracles).

### 2. Standard Constants

```python
REPRO_ID = "<repro_id>"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
```

### 3. Scope Invariant Functions

```python
def get_inputs():
    """Load inputs from the repro's make_inputs."""

def get_repro_instance():
    """Create Repro() for reference comparison."""
```

These enforce the scope invariant: the oracle uses the SAME inputs the repro
uses, and compares against the SAME eager reference.

### 4. Oracle Implementation

```python
def oracle_forward(inputs):
    """MUST accept same inputs as Repro.forward() and return same outputs."""
```

The function signature enforces scope matching. It takes the full input tuple
and returns the full output (same shapes, dtypes, count).

### 5. Standard CLI

```
python oracle_<name>.py --check       # correctness only
python oracle_<name>.py --bench       # benchmark only
python oracle_<name>.py               # both (default)
```

## Output Format

### --check output

Human-readable lines, one per output tensor:

```
Checking <repro_id>...
  output 0: PASS (shape=[8192, 262144] dtype=torch.bfloat16 max_diff=3.91e-03)
Correctness: PASS
```

Exit code: 0 = pass, 1 = fail.

### --bench output

One JSON line (machine-parseable) followed by optional human-readable context:

```json
{"repro_id": "sum_e00c7291b6ee", "oracle_us": 1523.45, "compile_us": 4892.12, "ratio": 3.212, "status": "GOOD"}
```

Fields:
- `repro_id`: string, matches the directory name
- `oracle_us`: float, oracle timing in microseconds (min-of-rep)
- `compile_us`: float, torch.compile timing in microseconds (min-of-rep)
- `ratio`: float, compile_us / oracle_us (>1.0 means oracle is faster)
- `status`: `"GOOD"` if ratio > 1.0, `"BAD_ORACLE"` if oracle is slower

Exit code: always 0 for --bench (BAD_ORACLE is a warning, not a failure).

## How validate_oracles.py Uses This Format

The validation script (`scripts/validate_oracles.py`) relies on:

1. **--check flag**: runs `python oracle_*.py --check` via subprocess.
   Expects exit code 0 for pass, non-zero for failure.

2. **Output parsing**: looks for "PASS", "FAIL", "SCOPE_MISMATCH" in stdout
   to classify results.

3. **Module loading fallback**: if --check is not supported (legacy oracles),
   loads the module and looks for `oracle_forward(inputs)` to call directly.

Oracles following this template are guaranteed to work with validate_oracles.py
because:
- They support `--check` with proper exit codes
- They print "PASS" or "FAIL" clearly
- They expose `oracle_forward` as a callable for programmatic validation
- They use `get_inputs()` which calls the repro's own `make_inputs()`

## Scope Invariant

The most critical property of a valid oracle:

> The oracle must compute the EXACT SAME function as `Repro()(*make_inputs())`.
> Same inputs in, same outputs out. No subset. No superset.

Violations that disqualify an oracle:
- Oracle only benchmarks the reduction, not the full fused kernel
- Oracle skips epilogue operations (cast, store, mask)
- Oracle uses different input shapes or dtypes
- Oracle produces fewer or more output tensors
- Oracle output has different shapes/strides than eager

If full-scope coverage is impractical, do NOT mark the oracle as
`implemented_unmeasured`. Instead, leave it as a diagnostic artifact and
note the missing scope in `next_action`.

## Migration Guide for Existing Oracles

For oracles that predate this format:

1. Add the gap diagnosis docstring
2. Replace custom input loading with `get_inputs()` / `_load_repro_module()`
3. Wrap the core computation in `oracle_forward(inputs)`
4. Add `run_check` and `run_bench` functions matching the template
5. Add the standard argparse CLI with `--check` / `--bench`
6. Verify with: `python oracle_<name>.py --check`

## Future Enhancement: Shape/Hardware Dispatch Registry

Currently oracles handle shape variation via:
- `@triton.autotune` (tile size selection)
- Manual if/else in `oracle_forward()` for algorithm selection
- `get_hardware_info()` for GPU-specific branching

If the pattern recurs enough, consider a registry:
```python
ORACLE_REGISTRY = {}

def register_oracle(shape_pred=None, hw_pred=None):
    def decorator(fn):
        ORACLE_REGISTRY[fn.__name__] = (fn, shape_pred, hw_pred)
        return fn
    return decorator

@register_oracle(shape_pred=lambda s: s[1] <= 1024)
def persistent_variant(inputs): ...

@register_oracle(shape_pred=lambda s: s[1] > 1024)  
def tiled_variant(inputs): ...
```

For now, simple if/else in oracle_forward() is sufficient.
