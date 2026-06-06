# pointwise_002b387bc089

## Classification: PERMUTE_COPY_FLAT_TILING (RESOLVED on B200)

## Current Result (B200, 126MB L2)

- Oracle path: `repros/canonical/pointwise_002b387bc089/oracle_layout_copy.py`
- Correctness: PASS
- Oracle: 30.08 us
- Compile (cd=True, combo=True): 29.47 us
- Ratio: 0.98 (compile is FASTER than oracle)
- Status: AT_FLOOR (no gap remains)

## Previous Result (H100/A100, smaller L2)

- Oracle: 24.38 us
- Compile (cd=True): 29.50 us
- Ratio: 1.21
- Status: GOOD (oracle won by 21%)

## Config Exploration (B200)

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Oracle | 31.52 | 1.000 |
| default (cd=True, combo=True) | 30.35 | 0.963 |
| prefer_nd_tiling=True | 30.43 | 0.965 |
| max_autotune_pointwise=True | 31.07 | 0.986 |
| multi_kernel=1 | 30.13 | 0.956 |
| multi_kernel=2 | 30.33 | 0.962 |
| multi_kernel=3 | 30.34 | 0.963 |

All configs match or beat the oracle on B200. The gap is hardware-dependent.

## Root Cause Analysis

The repro is a pure layout copy: `permute([0,2,1,3]) -> clone(contiguous) -> view` on a `[128, 12, 198, 64]` tensor (DeiT attention head rearrangement).

**Oracle approach**: 2D tiling with BLOCK_ROWS=4 rows at a time, processing 512+256 columns per row. Each row corresponds to a (batch, seq) pair, and the oracle computes head/dim indices explicitly. This gives good memory coalescing: adjacent threads process adjacent columns in the output (contiguous stores) and read with known strides.

**Inductor approach**: 1D flat tiling with modular arithmetic decomposition (`x0 = xindex % 64`, `x1 = (xindex//64) % 12`, etc.). The load pattern is `in_ptr0 + (x0 + 64*x2 + 12672*x1 + 152064*x3)` which produces scattered reads when the head dimension (x1, stride 12672) varies across adjacent elements.

Both produce exactly 1 kernel with 1 load and 1 store per element (same memory traffic). The difference is purely in memory access pattern quality - the oracle's 2D tiling gives better L2 hit rates and coalescing for this specific permute pattern.

**Why the gap disappears on B200**: The B200 has 126MB L2 cache. The working set is ~78MB (19.5M elements * 4 bytes * 2 for input+output). The large L2 absorbs the non-coalesced access pattern, making the tiling strategy irrelevant. On GPUs with smaller L2 (A100: 40MB, H100: 50MB), the working set doesn't fit and the access pattern matters more.

**Note on `prefer_nd_tiling`**: With `config.triton.prefer_nd_tiling = True`, Inductor does generate a 2D kernel (y=batch*heads*seq, x=dim=64) which is structurally similar to the oracle's approach. On B200 it performs identically to the 1D version.

## Kernel Count

- Inductor: 1 kernel (triton_poi_fused_clone_permute_0)
- Oracle: 1 kernel (_layout_copy_kernel)

## Design Issue (for smaller-L2 GPUs)

Inductor's pointwise codegen uses a flat 1D linearization for all permute-clone patterns. For transpose-heavy layouts (permuting the inner dimensions), a 2D tiling approach that tiles along output rows and processes full column widths per tile would give better coalescing. This requires the scheduler/codegen to detect "permute-clone" patterns and select a 2D grid with row-major output writes.

The `prefer_nd_tiling` flag already generates 2D tiling, but on GPUs where this matters (smaller L2), coordinate descent tuning would need to find the right tile sizes.

**Relevant files**:
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (tiling strategy selection)
- `/tmp/pytorch-work/torch/_inductor/ir.py` (layout detection for permute patterns)

## Conclusion

On B200 (126MB L2), the gap is fully resolved - Inductor's default 1D tiling already matches or beats the oracle (0.96-0.98x ratio). The working set fits in L2, neutralizing the access pattern advantage of 2D tiling. No code changes needed.

On GPUs with smaller L2 caches where this gap may persist, `config.triton.prefer_nd_tiling = True` generates the appropriate 2D kernel structure. The remaining question for those platforms would be whether coordinate descent tuning finds optimal tile sizes.
