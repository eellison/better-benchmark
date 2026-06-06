# pointwise_78e5e6e20190

## Classification: KV_CACHE_SLICE_SCATTER

## Current Result

- Oracle path: `repros/canonical/pointwise_78e5e6e20190/oracle_kv_cache_head_layout.py`
- Correctness: PASS
- Oracle: 6.98 us
- Compile (cd=True): 9.98 us
- Ratio: 1.431
- Status: GOOD (significant gap)

## Root Cause

The oracle computes the Llama inference KV-cache update and key head-layout materialization in one Triton launch. It writes the [0:32, 1:33, :, :] cache slice from the packed real view while directly producing the fresh contiguous [B*H, D, S] = [256, 64, 33] attention-key output.

Inductor lowers the captured slice/copy/slice_scatter/permute/clone/view/copy_ chain as generic layout and mutation operations. It does not recognize the cache-slice update plus immediately consumed head-layout clone as one guarded indexed store pattern with a mutable output alias.

## Kernel Count

- Oracle: 1 kernel (fused cache write + head-layout output)
- Inductor: Multiple kernels (slice_scatter mutation + permute/clone layout copy)

## Config Exploration

| Config | Ratio |
|--------|-------|
| Default (cd=True) | 1.431 |
| multi_kernel=2 | 1.430 |
| multi_kernel=3 | 1.438 |

No config helps. The gap is structural: the issue is about fusing mutation (slice_scatter into cache) with the downstream layout transformation.

## File/Line References

- `/tmp/pytorch-work/torch/_inductor/scheduler.py`: slice_scatter creates a mutation barrier that prevents fusion with the downstream clone/permute
- `/tmp/pytorch-work/torch/_inductor/ir.py`: copy_ and slice_scatter are lowered as separate mutation nodes

## Design Doc

The fix requires a KV-cache update/materialize template that fuses the slice overwrite with direct head-major output generation while preserving the returned input alias. This is a common pattern in inference (cache update + immediate attention key preparation) but Inductor's mutation handling creates barriers between the write and the subsequent read-for-layout-change.

This pattern affects all Llama-style KV-cache inference workloads.
