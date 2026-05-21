# Repro Reuse & Multi-Shape Design

## Problem

Today, repros are stored per-model in `output/aten_repros/<model>/` with
deduplication happening at extraction time via `pattern_hash` (graph topology)
and `shape_hash` (concrete dims). This means:

- Same kernel pattern extracted from 5 models → 5 files with identical `forward()`.
- Same pattern at 3 shapes → 3 separate files, no way to benchmark across shapes
  in one invocation.
- No reverse lookup: "which models produce this kernel pattern?"
- Reshape/stride baked into the graph makes two logically-same patterns hash differently.

## Design

### Layer 1: Canonical Repro Set (flat)

```
repros/
  canonical/
    amax_sum_eb4fe3ac03e0/
      repro.py            # The Repro class + benchmark harness
      shapes.json         # All known shape configs for this pattern
      meta.json           # Pattern metadata (ops, reduction types, etc.)
    var_mean_757a32aa7b72/
      repro.py
      shapes.json
      meta.json
    ...
```

Each canonical repro is keyed by `pattern_hash` — the structural hash of ops +
dtypes, ignoring concrete shapes. One directory per unique kernel topology.

### Layer 2: Model Manifest

```
repros/
  manifest.json           # model → [pattern_hash, ...] with shape info
```

Structure:

```json
{
  "version": 1,
  "models": {
    "dynamo_AlbertForMaskedLM_inference": {
      "repros": [
        {
          "pattern_hash": "eb4fe3ac03e0",
          "shape_config": "albert_inference_bs8",
          "region_index": 0,
          "original_file": "region_000_amax_sum_eb4fe3ac03e0_d46ab65d.py"
        }
      ]
    },
    "dynamo_DistillGPT2_inference": {
      "repros": [
        {
          "pattern_hash": "eb4fe3ac03e0",
          "shape_config": "distillgpt2_inference_bs32",
          "region_index": 0,
          "original_file": "region_000_amax_sum_eb4fe3ac03e0_23f019ef.py"
        }
      ]
    }
  }
}
```

This gives you:
- Forward lookup: model → which canonical repros it uses
- Reverse lookup: pattern_hash → which models share this kernel
- Shape provenance: each shape config is tagged with where it came from

### Layer 3: Multi-Shape `shapes.json`

```json
{
  "pattern_hash": "eb4fe3ac03e0",
  "configs": {
    "albert_inference_bs8": {
      "inputs": [
        {"shape": [8, 512], "dtype": "torch.int64", "stride": null},
        {"shape": [4096, 30000], "dtype": "torch.float32", "stride": null}
      ],
      "source_models": ["dynamo_AlbertForMaskedLM_inference"],
      "shape_hash": "d46ab65d"
    },
    "distillgpt2_inference_bs32": {
      "inputs": [
        {"shape": [32, 512], "dtype": "torch.int64", "stride": null},
        {"shape": [16384, 50257], "dtype": "torch.float32", "stride": [50260, 1]}
      ],
      "source_models": ["dynamo_DistillGPT2_inference"],
      "shape_hash": "23f019ef"
    },
    "opt_inference_bs16": {
      "inputs": [
        {"shape": [16, 512], "dtype": "torch.int64", "stride": null},
        {"shape": [8192, 50272], "dtype": "torch.float32", "stride": null}
      ],
      "source_models": ["dynamo_OPTForCausalLM_inference"],
      "shape_hash": "248255b1"
    }
  }
}
```

`stride: null` means contiguous; an explicit stride list means non-contiguous
(preserving the as_strided behavior from the current generator).

### Layer 4: Canonical `repro.py` (importable + directly runnable)

```python
"""Cross-entropy amax_sum pattern — canonical repro."""
import torch
from torch import device
from math import inf, nan
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs


class Repro(torch.nn.Module):
    def forward(self, arg0: "i64[B, S]", arg1: "f32[B*S, V]"):
        # ... same FX graph as today, but with symbolic annotations in comments
        reshape_default = torch.ops.aten.reshape.default(arg0, [-1])
        ne_scalar = torch.ops.aten.ne.Scalar(reshape_default, -100)
        amax_default = torch.ops.aten.amax.default(arg1, [1], True)
        # ... rest of forward ...
        return div_tensor


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is None:
        configs = load_shape_configs(__file__)
        shape_config = next(iter(configs.values()))

    inputs = make_inputs_from_config(shape_config)
    return inputs


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
```

Key properties:
- **Directly runnable**: `python repros/canonical/amax_sum_eb4fe3ac03e0/repro.py`
  benchmarks the first (default) shape config.
- **Shape-selectable**: `python repro.py --shape distillgpt2_inference_bs32`
  benchmarks a specific config.
- **All-shapes**: `python repro.py --all-shapes` benchmarks every config and
  reports a comparison table.
- **Importable**: other scripts can `from canonical.amax_sum_eb4fe3ac03e0.repro import Repro, make_inputs`.

### Shared Harness (`repro_harness.py`)

Lives at `repros/repro_harness.py`. Provides:

```python
def load_shape_configs(repro_file: str) -> dict:
    """Load shapes.json from the same directory as repro_file."""

def make_inputs_from_config(config: dict) -> list[torch.Tensor]:
    """Build tensors from a shape config entry (handles stride/as_strided)."""

def benchmark_repro(repro_file: str, repro_cls, make_inputs_fn, args=None):
    """Full benchmark: compile, CD-tune, SOL, CUDA graph. Handles --shape/--all-shapes."""

def count_output_bytes(model, inputs) -> int:
    """Count unique output bytes (alias-aware)."""

def count_triton_kernels(model, inputs) -> int:
    """Count .run() calls from fresh inductor cache."""
```

This is the common infra you mentioned — each canonical repro imports it rather
than inlining the benchmark logic.

## Dealing with Baked-in Reshapes

The hard case: `reshape([B*S, V])` with concrete `V=30000` vs `V=50257` makes
the graph hash differ even though the topology is identical.

Approach: **two-level hashing**.

1. **Topology hash** (`pattern_hash`): hash op names + graph structure only,
   ignoring all integer constants. This groups "same kernel pattern, different
   vocab size" together.
2. **Shape hash**: hash of concrete input shapes + strides (what we have today).

The canonical directory is keyed by topology hash. Two repros that differ only
in a baked-in reshape constant (like vocab_size) land in the same canonical
directory with different shape configs. The `forward()` in `repro.py` uses the
largest/most-representative shape, and `make_inputs` parametrizes the rest.

For reshapes that depend on input shape (like `reshape([-1, V])`), the V can
become a parameter derived from the input shape config. For reshapes with
truly-baked constants that change the op structure (e.g., different number of
output elements → different downstream ops), those genuinely are different
patterns and should remain separate.

Heuristic: if two pattern_hashes differ ONLY in integer constants appearing in
`reshape`, `expand`, `full`, `arange` size args, and the op DAG is otherwise
identical, merge them into one canonical repro.

## Migration Path

1. **Phase 1**: Build `repro_harness.py` with the shared benchmark infra.
   Existing repros continue to work unchanged.

2. **Phase 2**: Add `scripts/canonicalize_repros.py` that:
   - Scans `output/aten_repros/*/index.json`
   - Groups by pattern_hash (and optionally by topology hash for reshape merging)
   - Writes `repros/canonical/` tree + `manifest.json`
   - Each canonical repro.py is generated from the first instance, with
     `make_inputs` wired to `shapes.json`

3. **Phase 3**: Update `extract_reductions.py` to emit directly into canonical
   structure on new extractions. Per-model directories become views (symlinks or
   index-only).

## Queries the Manifest Enables

```python
# Which models share this kernel pattern?
manifest["models"] | filter(has pattern_hash "eb4fe3ac03e0")

# What shapes does this pattern appear at?
shapes_json["configs"].keys()

# Which patterns appear in >3 models? (high-value optimization targets)
Counter(p for m in manifest["models"].values() for p in m["repros"]).most_common()

# Run all shapes of a pattern and find shape-dependent perf cliffs
python repro.py --all-shapes --output results.json
```

## What This Doesn't Solve (Yet)

- **Dynamic shapes within one config**: a single repro that truly accepts
  arbitrary batch/seq would need the generator to emit symbolic dims. That's a
  generator-level change (emit `torch.sym_int` or parametrize factories).
- **Reshape fusion across different vocab sizes in one forward()**: the graph
  literally has different constants, so one `forward()` can't serve both without
  making V a parameter and rewriting the reshape. Separate shape configs with
  separate compilation is the pragmatic answer.
- **Cross-pattern similarity** (e.g., "these two patterns are both LayerNorm
  but with/without bias"): needs semantic labeling beyond structural hashing.
  The manifest could gain a `family` tag for this later.
