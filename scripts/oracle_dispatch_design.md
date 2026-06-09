# Oracle Dispatch: Explicit (Hardware, Shape, Configs) Registration — FINAL DESIGN

## Problem

All 1483 oracles were written on H100; we measure on B200. 474 are BAD_ORACLE,
largely because their launch configs (BLOCK sizes, num_warps) were tuned for
H100. The kernel bodies are usually fine — only the configs are wrong for the
new hardware. We need a way for the oracle server to add B200 configs to
existing kernels WITHOUT forking them, and to be explicit about which concrete
shape each implementation was tuned for.

## Design

### Core idea: explicit shape registration, not predicates

Each oracle was written and tuned for ONE concrete shape. Registration declares
that shape explicitly — the shape is documentation: "this implementation was
written/tuned for THIS shape."

```python
@registry.register(hardware="H100", shape=(8192, 262144))
def oracle_split_k(inputs): ...
```

Lambda/shape predicates are NOT supported (the registry raises `TypeError` if
`shape` is callable). Predicates hid what an oracle was actually tuned for;
concrete shapes don't.

`shape` accepts:
- a tuple of ints — the key (first tensor) input shape, e.g. `(8192, 262144)`
- a tuple of tuples — ALL tensor input shapes, e.g. `((8192, 262144), (262144,))`
- `None` — no shape constraint (only sensible for default fallbacks)

### Shared kernels with different configs (the critical case)

The same kernel function is often registered multiple times with different
launch configs — different hardware or shapes need different BLOCK sizes /
num_warps, but the kernel body is identical. The `configs` kwarg on `register`
is passed through to the implementation at dispatch time:

```python
def _softmax_impl(inputs, *, BLOCK=1024, num_warps=4):
    ...one kernel body...

registry.register(hardware="H100", shape=(32768, 1024),
                  configs={"BLOCK": 1024, "num_warps": 4})(_softmax_impl)
registry.register(hardware="B200", shape=(32768, 1024),
                  configs={"BLOCK": 2048, "num_warps": 8})(_softmax_impl)
registry.register(hardware="B200", shape=(8192, 262144),
                  configs={"BLOCK": 4096, "num_warps": 16})(_softmax_impl)
```

- Dispatch calls `fn(inputs, **configs)`.
- `configs=None` means the implementation is called plain: `fn(inputs)`.
- The decorator returns the function UNCHANGED so the same function can be
  registered repeatedly under different `(hardware, shape, configs)` combos.

This is exactly the mechanism that lets the oracle server fix the 474
BAD_ORACLE cases: add a B200 registration with new configs next to the
existing H100 one; the kernel body is shared.

### Dispatch tiers

Given actual inputs at runtime, `registry.dispatch(inputs)`:

| Tier | Match | Meaning |
|------|-------|---------|
| 1 | exact `(hardware, shape)` | tuned for this exact hw + shape |
| 2 | same shape, any hardware | right shape, configs from other hw |
| 3 | same hardware, nearest shape | right hw, tuned for a different shape (entries registered for this hw with `shape=None` qualify after any finite-distance shape entry) |
| 4 | any registration, nearest shape | wrong hw AND wrong shape |
| 5 | default (no constraints) | unconstrained fallback |

"Nearest shape" = smallest sum of `|log(actual_dim / registered_dim)|` across
dims. Same rank is required (distance is infinite otherwise). Distance 0.0 is
an exact match. Log-space distance means a 2x size difference in any dim
counts the same regardless of absolute size, which matches how kernel-config
suitability actually scales.

Within a tier the smallest distance wins; registration order breaks ties.

### Dispatch observability

After every `dispatch()` (or `select()`), `registry.last_dispatch_info` holds:

```python
{
    "tier": 3,
    "tier_name": "hw_nearest_shape",
    "fallback": True,            # tier > 1 — log "may be suboptimal"
    "fn_name": "_softmax_impl",
    "description": "...",
    "registered_hardware": "B200",
    "registered_shape": ((32768, 1024),),
    "configs": {"BLOCK": 2048, "num_warps": 8},
    "distance": 1.39,            # 0.0 = exact; None = no shape constraint
    "current_hardware": "B200",
    "actual_shapes": ((16384, 1024),),
}
```

Benchmarks should log a warning when `fallback` is true: the chosen
implementation was tuned for different hardware and/or shape and the measured
oracle time may understate what the oracle could achieve.

`registry.select(inputs)` returns `(entry, info)` without calling the
implementation, for tooling that wants to inspect dispatch decisions.

### Backward compatibility

Registration is OPT-IN. Oracle files that just define `oracle_forward(inputs)`
with no registry keep working with `check_oracle`, `bench_oracle`, and
`bench_oracle_all_shapes` unchanged. The registry only activates when an
oracle file instantiates one and routes `oracle_forward` through
`registry.dispatch`.

### Per-file registry

Each oracle file instantiates its own `OracleRegistry()`. Module-level
conveniences (`register_oracle` / `dispatch_oracle` / `reset_oracle_registry`)
exist for single-file use, but per-file registries are recommended.

### Hardware detection

`get_gpu_kind()` returns a cached short string ("B200", "H100", "A100", ...)
via nvidia-smi, matching `repro_harness._detect_hardware()`.

## Full example oracle file

```python
"""Oracle for softmax-family repro.

Tuned points:
  - H100 (32768, 1024): BLOCK=1024, num_warps=4   (original tuning)
  - B200 (32768, 1024): BLOCK=2048, num_warps=8   (re-tuned for B200)
  - B200 (8192, 262144): split-K, different algorithm
"""
import torch
import triton
import triton.language as tl
from oracle_harness import OracleRegistry

registry = OracleRegistry()


@triton.jit
def _softmax_kernel(X, Out, N, BLOCK: tl.constexpr):
    ...


def _softmax_impl(inputs, *, BLOCK=1024, num_warps=4):
    x = inputs[0]
    out = torch.empty_like(x)
    _softmax_kernel[(x.shape[0],)](x, out, x.shape[1],
                                   BLOCK=BLOCK, num_warps=num_warps)
    return out

# One kernel body, three tuned (hardware, shape, configs) points:
registry.register(hardware="H100", shape=(32768, 1024),
                  configs={"BLOCK": 1024, "num_warps": 4})(_softmax_impl)
registry.register(hardware="B200", shape=(32768, 1024),
                  configs={"BLOCK": 2048, "num_warps": 8})(_softmax_impl)
registry.register(hardware="B200", shape=(8192, 262144),
                  configs={"BLOCK": 4096, "num_warps": 16})(_softmax_impl)


# Different algorithm where configs alone aren't enough:
@registry.register(hardware="B200", shape=(8192, 1048576),
                   description="two-pass split-K for huge inner dim")
def _oracle_split_k(inputs):
    ...


# Unconstrained default fallback:
@registry.register(description="conservative default")
def _oracle_default(inputs):
    return torch.softmax(inputs[0], dim=-1)


def oracle_forward(inputs):
    out = registry.dispatch(inputs)
    info = registry.last_dispatch_info
    if info["fallback"]:
        print(f"NOTE: fallback dispatch tier={info['tier']} "
              f"({info['tier_name']}) -> {info['fn_name']}; may be suboptimal")
    return out
```

## Interaction with bench_oracle_all_shapes

`bench_oracle_all_shapes` iterates shape configs and calls `oracle_forward`
per shape; dispatch selects the best registration for each shape
automatically. One `oracle_forward` entry point, the RIGHT configs per shape,
and `last_dispatch_info` tells the sweep whether the timing came from a tuned
point (tier 1) or a fallback (tier > 1).

## What this does NOT do

- No automatic autotuning at dispatch time. Each registration is a pre-tuned
  point; dispatch just selects and forwards configs.
- No config interpolation between registered shapes. Nearest-shape fallback
  reuses the nearest tuned configs verbatim (and flags it as a fallback).
- No serialization of the registry. Rebuilt on import.
- No lambda/predicate shapes. Concrete tuples only.

## Implementation

Lives in `oracle_harness.py`:
- `get_gpu_kind()` — cached GPU detection
- `class OracleRegistry` — registration, tiered dispatch, `last_dispatch_info`,
  `select()`, `list_entries()`
- `register_oracle()` / `dispatch_oracle()` / `reset_oracle_registry()` —
  module-level conveniences

Self-test: `scripts/test_oracle_dispatch.py` (no GPU work; monkeypatches
`get_gpu_kind`). Template: `scripts/oracle_template.py`.
