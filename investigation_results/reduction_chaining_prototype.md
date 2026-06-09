# Reduction Chaining Prototype: Investigation Results

## Summary

Successfully prototyped the detection layer for "reduction chaining" in Inductor's
scheduler. The pattern is correctly identified for the RMSNorm backward case
(sum_sum_mean_9af96955f8cc), confirming that the scheduler infrastructure can
support this optimization with targeted extensions.

## The Pattern

```
Producer (op3):  [M, N, K] -> reduce K -> [M, N]
                 numel=4194304 (=2048*2048), rnumel=8
                 (gather + scale + mask + sum over 8 experts)

Consumer (op4_op5): [M, N] -> reduce N -> [M], then pointwise epilogue
                    numel=2048, rnumel=2048
                    (add + square + mean + rsqrt + weight*x)

Fused Oracle:  Grid=[2048], each program:
               - Loops K=8 experts, accumulating N=2048 in registers
               - Reduces N=2048 in-register for statistics
               - Applies normalization and writes output
```

**Savings:** Eliminates 8MB intermediate write + 8MB intermediate read = 16MB bandwidth.
Oracle achieves 2.4x speedup (87us -> 36us).

## Why Existing Fusion Mechanisms Fail

### 1. Standard Fusion (SIMDScheduling.can_fuse)
```python
if node1.is_reduction() and node2.is_reduction():
    reduction_can_fuse = numel1 == numel2 and rnumel1 == rnumel2
```
**Fails:** numel1=4194304 != numel2=2048, rnumel1=8 != rnumel2=2048

### 2. MixOrderReduction
- Requires siblings (not producer-consumer) - line 352-356 rejects dependent pairs
- Requires swapped dimensions: (numel, rnumel) == reversed(other) - not our case

### 3. NestedReduction
- Requires `outer_total == grouped_total` (same number of total elements)
- Here: outer_total = 4194304 * 8 = 33554432 vs grouped_total = 2048 * 2048 = 4194304
- Fundamentally different: NestedReduction handles *partition + sub-reduce* over the
  same logical elements, not *produce-then-consume* different logical shapes.

### 4. V.choices.can_fuse (heuristic gate)
Even if the backend said "yes", the `shared_data_score` check (line 594-621 in
choices.py) blocks the pair because the read/write MemoryDep objects have different
indexing expressions (4194304 flat vs 2048x2048), causing `shared_data_score = 0`.

## What We Built

### Detection Module: `torch/_inductor/reduction_chaining.py`

A `ReductionChaining` class that detects when:
1. Producer writes [M, N] via a reduction (numel=M*N, rnumel=K)
2. Consumer reads [M, N] and reduces over N (numel=M, rnumel=N)
3. `prod_numel == cons_numel * cons_rnumel` (dimension compatibility)
4. `cons_rnumel <= 2048` (fits in persistent register tile)
5. Producer's output buffer has no external consumers

### Integration Point: `torch/_inductor/codegen/simd.py`

Hooked into the reduction-reduction fusion check in `SIMDScheduling.can_fuse`,
after NestedReduction's check:

```python
if not reduction_can_fuse:
    from torch._inductor.reduction_chaining import ReductionChaining
    reduction_can_fuse = ReductionChaining.can_fuse(node1, node2)
```

### Config: `torch/_inductor/config.py`

```python
reduction_chaining = os.environ.get("TORCHINDUCTOR_REDUCTION_CHAINING", "0") == "1"
```

### Current Status: Detection-only

`can_fuse()` returns False after confirming the pattern matches, because the
existing `FusedSchedulerNode` codegen cannot handle incompatible iteration spaces.
A full implementation requires:

1. **New node type:** `FusedChainedReductions(FusedSchedulerNode)` - similar to
   `FusedNestedReductions` or `FusedMixOrderReductions`

2. **Custom codegen:** `codegen_chained_reduction()` method in SIMDScheduling
   that generates a single Triton kernel with:
   - Grid = [M] (consumer's outer dim)
   - Persistent RBLOCK = N (the intermediate dimension)
   - Inner loop over K (producer's reduction dim)
   - Phase 1: accumulate N values from K iterations
   - Phase 2: reduce N values for statistics
   - Phase 3: apply pointwise epilogue using statistics

3. **Scheduler integration:** In `Scheduler.fuse()` (line 9770), add:
   ```python
   elif ReductionChaining.can_fuse(node1, node2):
       return FusedChainedReductions(node1, node2)
   ```

4. **Codegen dispatch:** In `Scheduler.codegen_node_impl()` (line 9485):
   ```python
   elif isinstance(node, FusedChainedReductions):
       self.get_backend(device).codegen_chained_reduction(node)
   ```

## Key Architectural Insight

The difficulty is NOT in detecting the pattern (that works fine) but in
CODE GENERATION. The two reductions have fundamentally different loop structures:
- Producer: 4194304 work items, each reducing 8 elements
- Consumer: 2048 work items, each reducing 2048 elements

The fused kernel must use the CONSUMER's grid (2048) and restructure the
producer's work to fit within each consumer tile. Each consumer tile (one row)
must process K=8 iterations of the producer's reduction, keeping N=2048 values
in registers. This is effectively a "transpose" of the producer's grid structure.

## Next Steps for Full Implementation

1. Create `FusedChainedReductions` with metadata about M, N, K dimensions
2. Write codegen that generates the two-phase persistent kernel
3. Handle the indirect indexing (the producer uses `index.Tensor` via inv_perm)
4. Ensure correctness when producer has masking/conditional logic
5. Add autotuning for the tiling parameters (how to split M across SMs)

## Validated Approach: TritonTemplate Alternative

As a simpler alternative to modifying the scheduler's codegen, a
`TritonTemplate` for the specific "gather+reduce → normalize" pattern could
be registered via a post-grad pattern match. This would:
- Match the specific op sequence (index + mul + where + sum + add + pow + mean + rsqrt + mul)
- Replace it with a call to a hand-written Triton template
- Avoid modifying the scheduler's fusion logic entirely

The oracle kernel at `repros/canonical/sum_sum_mean_9af96955f8cc/oracle_rmsnorm_bwd.py`
serves as the template for this approach.
