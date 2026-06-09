# Oracle Dispatch: Registration-Based Multi-Implementation Mechanism

## Problem

Oracles today: one kernel, tuned for one shape, on one GPU. Three real needs:

1. **Reuse oracles across shapes** -- many oracles work for a range of shapes
   (e.g., "any batch size, inner dim <= 4096"). We need to express that.
2. **Clarity about what shape an oracle was written for** -- when reading an oracle
   file, it should be obvious which shapes each implementation targets.
3. **Different configs or kernels per hardware** -- a B200-tuned kernel may use
   different tile sizes, or an entirely different algorithm, than an H100 one.

## Design

### Core Idea

Each oracle file registers complete implementations keyed by:
- **hardware**: which GPU this was tuned for (or None = any GPU)
- **shape predicate**: which input shapes this applies to (or None = any shape)

The shape predicate is the key piece -- it makes explicit what shapes an oracle
covers. An oracle can cover a single shape, a range, or all shapes.

### API

```python
from oracle_harness import OracleRegistry

registry = OracleRegistry()

# Written for B200, small inner dimension. Clear about what it targets.
@registry.register(
    hardware="B200",
    shape=lambda inputs: inputs[0].shape[-1] <= 1024,
    description="persistent single-pass, B200, inner_dim <= 1024"
)
def oracle_persistent_small(inputs):
    ...

# Same kernel, different config for large shapes on B200.
@registry.register(
    hardware="B200",
    shape=lambda inputs: inputs[0].shape[-1] > 1024,
    description="split-K multi-pass, B200, inner_dim > 1024"
)
def oracle_split_k_large(inputs):
    ...

# Reusable across all shapes on H100 (hardware-specific config, any shape).
@registry.register(hardware="H100", description="H100-tuned, all shapes")
def oracle_h100(inputs):
    ...

# Default fallback: works everywhere, conservative config.
@registry.register(description="default fallback, any hw/shape")
def oracle_default(inputs):
    ...

def oracle_forward(inputs):
    return registry.dispatch(inputs)
```

### Key Properties

**1. Reuse across shapes:**
An oracle with `shape=None` (no predicate) applies to ALL shapes on its target
hardware. An oracle with a broad predicate like `lambda inputs: inputs[0].ndim == 2`
covers a wide range. The predicate makes the reuse range explicit.

**2. Clarity about targeted shapes:**
The `description` field + the shape predicate together document intent. When reading
the file, you see exactly: "this function is for B200 when inner_dim <= 1024".
No ambiguity about what shape an oracle was written for.

**3. Different configs/kernels per hardware:**
Register the same algorithmic approach with different configs per hardware, or
register entirely different algorithms. The dispatch handles routing.

```python
# Same algorithm, hardware-specific tile sizes
@registry.register(hardware="B200", description="layernorm, BLOCK=2048 for B200 shared mem")
def oracle_b200(inputs):
    return _layernorm_kernel(inputs, BLOCK=2048)

@registry.register(hardware="H100", description="layernorm, BLOCK=1024 for H100")
def oracle_h100(inputs):
    return _layernorm_kernel(inputs, BLOCK=1024)
```

### Dispatch Priority

When `registry.dispatch(inputs)` is called:

1. **Exact match** (hardware + shape): hardware matches current GPU AND shape predicate returns True
2. **Shape-only**: hardware=None AND shape predicate returns True
3. **Hardware-only**: hardware matches current GPU, no shape constraint
4. **Default**: no constraints (hardware=None, shape=None)

First match within each tier wins (registration order breaks ties within a tier).

### Per-File Registry

Each oracle file instantiates its own `OracleRegistry()`. Oracle files are
independent -- they should not interfere with each other.

### Hardware Detection

`get_gpu_kind()` returns a cached short string: "B200", "H100", "A100", etc.

### Interaction with bench_oracle_all_shapes

When benchmarking across shapes, dispatch automatically selects different
implementations per shape:

```python
# bench_oracle_all_shapes iterates over shape configs
for config in load_shape_configs(shapes_file):
    inputs = make_inputs_from_config(config)
    # oracle_forward -> registry.dispatch(inputs) -> picks best impl for THIS shape
    result = bench_oracle(oracle_forward, instance, inputs, ...)
```

This is the payoff: one `oracle_forward` entry point, but the RIGHT kernel
runs for each shape. Shapes that can reuse the same oracle do; shapes that
need a specialized kernel get one.

### Listing What Is Registered (Debugging / CI)

```python
for entry in registry.list_entries():
    print(f"  {entry['fn_name']}: hw={entry['hardware']} shape={entry['description']}")
```

Output:
```
  oracle_persistent_small: hw=B200 shape=persistent single-pass, B200, inner_dim <= 1024
  oracle_split_k_large: hw=B200 shape=split-K multi-pass, B200, inner_dim > 1024
  oracle_h100: hw=H100 shape=H100-tuned, all shapes
  oracle_default: hw=None shape=default fallback, any hw/shape
```

## Example: Full Oracle File

```python
"""Oracle for amax_sum pattern -- online softmax variants.

Shape coverage:
  - B200, inner <= 2048: persistent single-pass (1 kernel)
  - B200, inner > 2048:  split-K (2 kernels)
  - Any GPU, any shape:  conservative default
"""
import torch
import triton
import triton.language as tl
from oracle_harness import OracleRegistry

registry = OracleRegistry()


# --- B200 small: persistent, written for inner_dim <= 2048 ---
@triton.jit
def _kernel_persistent(X_ptr, Out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    ...

@registry.register(
    hardware="B200",
    shape=lambda inputs: inputs[0].shape[-1] <= 2048,
    description="persistent single-pass, inner <= 2048"
)
def oracle_b200_small(inputs):
    x = inputs[0]
    out = torch.empty_like(x)
    _kernel_persistent[(x.shape[0],)](x, out, x.shape[-1], BLOCK=2048)
    return (out,)


# --- B200 large: split-K, written for inner_dim > 2048 ---
@triton.jit
def _kernel_split_k(X_ptr, Out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    ...

@registry.register(
    hardware="B200",
    shape=lambda inputs: inputs[0].shape[-1] > 2048,
    description="split-K multi-pass, inner > 2048"
)
def oracle_b200_large(inputs):
    x = inputs[0]
    out = torch.empty_like(x)
    _kernel_split_k[(x.shape[0],)](x, out, x.shape[-1], BLOCK=1024)
    return (out,)


# --- Default: reused across ALL shapes on non-B200 hardware ---
@triton.jit
def _kernel_default(X_ptr, Out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    ...

@registry.register(description="default, reusable across all shapes")
def oracle_default(inputs):
    x = inputs[0]
    out = torch.empty_like(x)
    _kernel_default[(x.shape[0],)](x, out, x.shape[-1], BLOCK=1024)
    return (out,)


# --- Public entry point ---
def oracle_forward(inputs):
    return registry.dispatch(inputs)
```

## What This Does NOT Do

- No automatic kernel generation or autotune at dispatch time.
  Each registered function is pre-tuned; dispatch just selects.
- No config merging. Variants are completely independent.
- No serialization of the registry. Rebuilt on import.
- No dynamic registration at benchmark time. All variants defined at module level.

## Implementation Location

Prototype lives in `oracle_harness.py` alongside existing infrastructure:
- `get_gpu_kind()` -- cached GPU detection
- `class OracleRegistry` -- per-file registry + dispatch
- `register_oracle()` / `dispatch_oracle()` / `reset_oracle_registry()` -- global convenience
