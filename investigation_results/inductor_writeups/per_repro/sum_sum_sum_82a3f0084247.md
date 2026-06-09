# sum_sum_sum_82a3f0084247

## Queue Position

- Rank: 35
- Family: `multi_output_reduction_templates`
- Owner: `Fermi`
- Closure status: `measured`
- Oracle status: `BAD_ORACLE`
- Diagnosis: `RECOMPUTE_FUSION`

## Current Gap (measured 2026-06-03, B200)

- Best compile (CD+scatter_reduce_fusion): `607.7 us`
- Memcopy SOL: `96.3 us`
- Bandwidth floor (IO=623MB): `107.4 us`
- Realistic floor (3-pass dual LN backward): `322.2 us`
- Compile / realistic floor: `1.89x`
- Oracle (Triton multi-output): `5974.2 us` (BAD - 9.8x slower than compile)
- Oracle path: `repros/canonical/sum_sum_sum_82a3f0084247/oracle_multi_output_reduction.py`

## Oracle State

- Oracle measured at 5974.2 us, but compile is 9.83x FASTER.
- Oracle is severely broken/unoptimized.
- Oracle needs complete rewrite.

## Pattern Analysis

Swin Transformer (timm_swin_base_patch4_window7_224) dual LayerNorm backward:
1. Large matmul result [401408,128] reshaped + permuted + windowed to [128,56,56,128]
2. First LN backward: reductions over dim [3] (C=128) producing [128,56,56,1]
   - sum(mul_tensor, [3]) and sum(mul_tensor*mul_tensor_4, [3])
   - Then pointwise using reduction results
3. Second LN backward: same pattern on the combined gradient
   - sum(mul_tensor_9, [3]) and sum(mul_tensor_9*mul_tensor_2, [3])
   - Then pointwise
4. Five [128]-sized final reductions over [0,1,2]

The data is re-read multiple times due to dependent reductions.

## Inductor Closure Path

- Implementation track: RECOMPUTE_FUSION - dual LN backward requires re-reading [128,56,56,128].
- Compile is already at 1.89x realistic floor - very efficient.
- The permute + view chain at the start prevents early fusion.
- Main remaining opportunity: fuse the two LN backward blocks more aggressively.

## Done Criteria

- Oracle needs rewrite (current is 10x worse than compile).
- Compile at ~1.9x realistic floor suggests limited further optimization opportunity.
