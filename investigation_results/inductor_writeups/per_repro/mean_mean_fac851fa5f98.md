# mean_mean_fac851fa5f98 - Qwen RoPE Inputs RMSNorm

## Benchmark Result
- Oracle: 16.1 us
- Compile: 18.18 us
- Ratio: 1.129x
- Status: GOOD

## Root Cause
The oracle computes both query and key RMSNorm + RoPE in 2 specialized kernels:
1. `_query_rmsnorm_rope_kernel`: RMSNorm over mm_189 rows, RoPE application, writes directly to non-contiguous [B, Hq, S, 128] query output layout
2. `_key_rmsnorm_rope_repeat_kernel`: RMSNorm over mm_190 rows, RoPE, AND writes the repeated grouped-KV expansion directly into contiguous [B, Hq, S, 128] output (repeat=2 heads)

Inductor emits 2 kernels:
1. `triton_per_fused_0`: fused persistent reduction doing both query and key RMSNorm + RoPE (32768 reductions)
2. `triton_poi_fused_add_clone_convert_element_type_expand_mean_mul_permute_pow_rsqrt_unsqueeze_view_1`: a separate pointwise kernel to materialize the expand+clone for the grouped-KV repeat (4194304 elements)

The gap is that Inductor cannot sink the grouped-KV repeat (unsqueeze -> expand -> clone -> view) into the RMSNorm+RoPE reduction epilogue. The key reduction produces [4, 8, 512, 128] but the output needs [4, 16, 512, 128] (each of 8 KV heads repeated 2x). Inductor schedules a separate clone kernel to materialize this expansion.

## Kernel Count
- Oracle: 2 kernels (query RMSNorm+RoPE, key RMSNorm+RoPE+repeat)
- Inductor: 2 kernels (fused RMSNorm+RoPE for both Q/K, separate expand+clone)

## Config Exploration
- `combo_kernels = True`: helps fuse the Q and K reductions into one launch but doesn't eliminate the expand/clone
- `coordinate_descent_tuning = True`: helps tune the reduction kernel but doesn't fix the extra kernel

## File/Line References
- `/tmp/pytorch-work/torch/_inductor/scheduler.py`: the scheduler realizes the expand+clone as a separate node because the reduction output shape [4, 8, 512, 128] differs from the final shape [4, 16, 512, 128]
- `/tmp/pytorch-work/torch/_inductor/ir.py`: expand+clone triggers materialization since it's a shape-changing operation that the reduction epilogue cannot express

## Design Doc
Classification: SCHEDULER_FUSION

The scheduler needs to allow reduction epilogues to write repeated/expanded output layouts directly. When a reduction produces [B, Hkv, S, D] and the consumer is expand([B, Hkv, repeat, S, D]).clone().view([B, Hq, S, D]), the reduction kernel can write each result row to `repeat` output locations in its store phase.

This is a common pattern in grouped-query attention (GQA) models like Qwen, where KV heads are fewer than Q heads and must be replicated. The fix is to teach the scheduler that an expand+clone consumer of a reduction can be absorbed into the reduction's epilogue store by emitting a static loop over the repeat factor.

### Affected Repro Count
This pattern appears in all GQA models (Qwen, Llama-3, Mistral) that use grouped KV heads with RMSNorm+RoPE. Likely affects 5-10 repros in the corpus.
