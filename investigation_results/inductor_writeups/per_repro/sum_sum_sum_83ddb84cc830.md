# sum_sum_sum_83ddb84cc830

## Summary

- Model: MobileBERT backward (scatter-reduce)
- Oracle: `oracle_mobilebert_scatter_reduce.py`
- Classification: SCATTER_REDUCE
- Ratio: 1.725x (oracle 114.59us, compile 197.63us)
- Kernel count: Inductor multiple kernels, Oracle fused scatter-reduce with direct side output

## Root Cause

The oracle computes the full MobileBERT backward tuple by producing the dense transposed side output directly while fusing the sibling channel reductions and the two duplicate-preserving indexed accumulations into Triton reduction/scatter kernels.

Inductor materializes the shared `[256,128,512]` intermediates and lowers the reductions, `index_put(accumulate=True)`, and transpose as separate generic scheduled work. It does not recognize this gather-mask-reduce/indexed-scatter family as one shared producer with several accumulator destinations.

The 1.725x gap comes from:
1. Materializing the full `[256,128,512]` intermediate
2. Separate scatter kernels for each destination
3. Cannot keep row-local reductions live while writing to multiple scatter targets

## Config Exploration

| Config | Time (us) | Ratio vs Oracle |
|--------|-----------|-----------------|
| Default (CDT) | 197.63 | 1.725x |
| multi_kernel=2 | error (multi_kernel cache_file_path) | N/A |
| multi_kernel=3 | error (multi_kernel cache_file_path) | N/A |

multi_kernel=2/3 both error for this workload (multi_kernel cache path issue). Default CDT is the only working config. The gap is purely structural.

## Fix Assessment

**Design doc** -- requires SCATTER_REDUCE lowering.

### What's needed:
Teach Inductor to fuse rowwise producers into direct duplicate-safe scatter-reduce epilogues and sibling reductions without materializing the full intermediates. The scheduler/codegen needs a gather-mask-reduce/indexed-scatter template.

### Files:
- `torch/_inductor/fx_passes/post_grad.py` (scatter-reduce pattern)
- `torch/_inductor/scheduler.py` (fusion across scatter boundaries)
- `torch/_inductor/codegen/triton.py` (atomic accumulation + side output)
