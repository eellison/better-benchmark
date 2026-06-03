# Scatter-Reduce Fusion Prototype

## Summary

Implemented a prototype FX pass that recognizes scatter-then-reduce patterns and can eliminate the scatter intermediate. The pass is at:

- **FX Pass**: `/tmp/pytorch-work/torch/_inductor/fx_passes/scatter_reduce_fusion.py`
- **Triton Kernel**: `/tmp/pytorch-work/torch/_inductor/kernel/scatter_reduce_gather.py`
- **Config**: `torch._inductor.config.scatter_reduce_fusion_enabled` (env: `TORCHINDUCTOR_SCATTER_REDUCE_FUSION=1`)
- **Integration**: Wired into `post_grad.py` after `partitioned_scatter_optimization_pass`

## Results

### Pattern Detection (Verified)

The pass correctly detects the scatter-reduce pattern in all target repros:
- 4 scatter nodes (bilinear index_put with accumulate=True)
- Intervening add nodes combining scatter results
- Optional where/mask node (BN-affine + ReLU gate)
- Final sum reduction over dims [0, 2, 3] (batch + spatial)
- Validates that all indexed dims (2, 3) are covered by reduction

### Rewrite: Simple Case (No Mask) - WORKING

For `sum(scatter_add(zeros, idx, values), dims)` patterns without an intervening mask:
- **Algebraic identity**: `sum(scatter_add(zeros, idx, src), all_scatter_dims) == sum(src, source_dims)`
- **Correctly rewrites** the FX graph to eliminate the scatter entirely
- **End-to-end verified** through `torch.compile` with matching outputs

### Rewrite: Masked Case (UNet Pattern) - DETECTED, KERNEL PROTOTYPE WORKING

For the UNet pattern with mask:
```
zeros -> 4x index_put(accumulate=True) -> add -> where(mask, 0, result) -> sum(dim=[0,2,3])
```

The pass detects the pattern and logs it. A separate gather-reduce kernel is provided:
- `scatter_reduce_gather_fast`: PyTorch vectorized implementation
- `try_scatter_reduce_gather_triton`: Triton implementation (prototype)

### Performance Benchmarks (A100)

At UNet scale (B=8, C=128, H_out=160, W_out=239, H_src=320, W_src=478):

| Implementation | Time (us) | Speedup vs Original |
|----------------|-----------|-------------------|
| Original (scatter+mask+reduce) | 113,141 | 1.0x |
| PyTorch gather-reduce | 18,492 | 6.1x |
| Triton gather-reduce (prototype) | 18,535 | 6.1x |

Note: These are eager-mode timings without Inductor fusion. The original 14.5ms
reported in the repro is from Inductor-compiled code (which fuses some but not all
of the operations). The relative improvement should translate to the compiled case.

### Correctness Verification

- Simple case (no mask): exact match (max diff: 3.81e-06, fp32 rounding)
- Masked case at scale: matches within tolerance (max diff: 3.91e-03, max rel err: 2.53e-05)
- Both verified against the original scatter-then-reduce computation

## Architecture

### FX Pass (`scatter_reduce_fusion.py`)

```
scatter_reduce_fusion_pass(graph)
  -> _find_scatter_reduce_chains(graph)
       -> For each sum node, trace backwards through where/mask/add/index_put
       -> Validate: all scatters target same zeros, reduction covers indexed dims
  -> For each chain:
       -> Simple case: _rewrite_simple_scatter_reduce (direct graph transform)
       -> Masked case: _rewrite_masked_scatter_reduce (detection + custom kernel)
```

Key design decisions:
1. Uses direct graph traversal (not pattern_matcher DSL) because the pattern has variable structure (N scatters, optional mask, variable add chains)
2. Conservative single-use checking: only rewrites if the scatter intermediate has no other consumers
3. Validates algebraic correctness: reduction must cover ALL indexed dimensions

### Gather-Reduce Kernel (`scatter_reduce_gather.py`)

The kernel eliminates the scatter intermediate by:
1. For each source position (b, c, r, s), look up its target output position via the index tensors
2. Check if that output position is masked (gather the mask into source space)
3. If not masked, accumulate the source value into the per-channel result

```python
# Core logic of scatter_reduce_gather_fast:
for each source tensor src_i with indices (row_idx_i, col_idx_i):
    gathered_mask = mask[:, :, row_idx_i, col_idx_i]  # [B, C, H_src, W_src]
    unmasked_src = src_i * (~gathered_mask).float()
    result += unmasked_src.sum(dim=[0, 2, 3])         # [C]
```

## Next Steps

### P0: Wire the masked rewrite into the FX pass
The `_emit_gather_reduce_for_bilinear` function currently returns False (detection only).
To complete the integration:
1. Register `scatter_reduce_gather_fast` as a custom op via `torch.library`
2. In the FX pass, replace the scatter+mask+reduce subgraph with a call to the custom op
3. Provide a lowering for the custom op that emits the gather-reduce Triton kernel

### P1: Optimize the Triton kernel
The current Triton kernel is 1:1 with the PyTorch version (no speedup). To reach SOL:
1. Tile over (channel, batch) with persistent programs
2. Use shared memory for row/col index caching
3. Vectorize source loads (contiguous in W dimension)
4. Use `tl.atomic_add` or warp-level reduction for channel accumulation

### P2: Handle the full repro (not just the scatter-reduce portion)
The target repros also include:
- BN backward computations that USE the masked scatter result
- The mask itself depends on the skip connection (separate computation)
- Two dependent reductions that share the intermediate

The full optimization requires fusing the mask computation, gather-reduce, and
both dependent reductions into a single kernel pass.

### P3: Extend to max_pool backward (sum_18262b26934c)
The max_pool case uses `scatter_add` instead of `index_put` with a different
index structure. The same algebraic identity applies but needs a different
pattern match.

## Files Modified

| File | Change |
|------|--------|
| `torch/_inductor/fx_passes/scatter_reduce_fusion.py` | NEW: FX pass implementation |
| `torch/_inductor/kernel/scatter_reduce_gather.py` | NEW: Gather-reduce kernel |
| `torch/_inductor/fx_passes/post_grad.py` | Added import and pass invocation |
| `torch/_inductor/config.py` | Added `scatter_reduce_fusion_enabled` config |
