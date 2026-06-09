# pointwise_49473bb3f273 — Grouped-KV Expand Clone

## Summary
- **Repro**: vllm_Qwen_Qwen3-0.6B_000
- **Oracle**: oracle_grouped_kv_materialize.py
- **Ratio**: 1.09x (oracle 6.72us vs compile 7.33us)
- **Classification**: NEW_PATTERN

## Root Cause

The oracle computes a grouped-KV materialization where a [B, KV, S, D] tensor (after
view/permute) is expanded along a new group dimension (GROUPS=2) and cloned to contiguous.
The oracle kernel loads each source KV-head tile once and stores it into every expanded
group slot (2 stores per load), whereas Inductor lowers this as a generic pointwise copy
kernel where each output element independently computes its source index and loads from the
input.

With GROUPS=2, the oracle does 1 load + 2 stores per source element, while Inductor does
1 load + 1 store per output element (so 2 loads + 2 stores total for the same source data).
The oracle saves ~50% of memory reads by reusing each loaded value across group stores.

## Kernel Count
- **Oracle**: 1 kernel (multi-store per source tile)
- **Inductor**: 1 kernel (per-output-element copy with computed source index)

## Config Exploration
- `coordinate_descent_tuning = True`: no meaningful effect
- `combo_kernels = True`: N/A (single kernel)
- The gap is only 1.09x which is relatively small, partly because GROUPS=2 (minimal reuse).

## Generated Code Analysis

Inductor kernel:
```python
def triton_poi_fused_clone_expand_permute_unsqueeze_view_0(in_ptr0, out_ptr0, xnumel, XBLOCK):
    x0 = (xindex % 128)       # D dimension
    x1 = ((xindex // 128) % 512)  # S dimension  
    x3 = ((xindex // 131072) % 8)  # KV head
    x4 = xindex // 1048576    # batch
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x3 + 1024*x1 + 524288*x4), ...)
    tl.store(out_ptr0 + (x5), tmp0, ...)
```

Each thread independently computes the source address. Adjacent threads in the group
dimension (stride 65536) load from the SAME source address, causing redundant global
memory reads. The oracle instead assigns threads by (batch, kv_head, tile) and writes
to multiple group offsets in a loop.

## Design Doc: Grouped Expand-Clone Optimization

**What's needed**: Detect when a clone of an expanded (zero-stride) dimension produces a
contiguous output. In such cases, the copy kernel should iterate over the expanded
dimension in the inner loop (multiple stores from one load) rather than computing the
source index per output element.

**Where to fix**:
- `torch/_inductor/codegen/triton.py` — When generating copy kernels for expand+clone
  patterns, detect zero-stride dimensions and emit a multi-store loop.
- `torch/_inductor/scheduler.py` — Could also be handled at the scheduling level by
  recognizing the expand-clone as a "broadcast copy" pattern.

**Affected files**:
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py`
- `/tmp/pytorch-work/torch/_inductor/scheduler.py`

**Note**: The gap is only 1.09x (0.61us absolute) because GROUPS=2 provides minimal reuse.
With larger group counts (e.g., GQA with 8 or 16 groups), the gap would be proportionally
larger. This is a common pattern in GQA-based LLM inference (Llama, Qwen, etc.).
