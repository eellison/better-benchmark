# amax_sum_any_511c5ade6e96

## Classification: `BANDWIDTH_BOUND`

Artifact: `repros/canonical/amax_sum_any_511c5ade6e96/oracle_online_softmax.py`

Parent integration note: diagnosis-only. Keep the main queue `oracle_path`
blank because the full-scope Triton artifact loses the required local combo
compile config even though it beats the historical queue best.

This is intended as a full-scope oracle for the compiled `repro.py` region. It
consumes the same five inputs as `Repro.forward`, computes the f32
`[512,128,128] -> [16,32,128,128]` score view, broadcasts the bool
`[16,1,128,128]` mask to `0/-inf`, applies the `any(eq(-inf))` all-masked-row
guard, computes stable last-dimension softmax, zero-fills all--inf rows, and
returns the final contiguous f32 `[512,128,128]` tensor. It is not a
softmax-only subset.

The oracle differs from Inductor by consuming scores and the broadcast mask
directly in a shape-specialized head-tiled Triton row-softmax kernel, avoiding
materialization of the mask add, boolean guard, and intermediate softmax
tensors. The required combo config already lowers this repro to a faster
full-scope online-softmax kernel, so this artifact does not expose a missing
lower floor. No actionable Inductor change is recommended for this repro beyond
preserving the existing online-softmax lowering and tuning.

## Results

Commands:

```bash
python -m py_compile repros/canonical/amax_sum_any_511c5ade6e96/oracle_online_softmax.py
python repros/canonical/amax_sum_any_511c5ade6e96/oracle_online_softmax.py --check
python repros/canonical/amax_sum_any_511c5ade6e96/oracle_online_softmax.py --bench --warmup 10 --rep 50
```

Correctness: `PASS`; output shape `[512, 128, 128]`, dtype `torch.float32`,
stride `[16384, 128, 1]`. Default-input and forced edge-row probes both
matched eager `Repro.forward` with `max_abs=5.960464e-08` and
`max_rel=6.407153e-07`.

Timing (`--warmup 10 --rep 50`, CUDA graph median):

| Path | Time (us) |
| --- | ---: |
| Full-scope Triton diagnostic oracle (`block_h=8,num_warps=2,load_all_scores=True`) | `23.296` |
| `coordinate_descent_tuning=True` | `28.640` |
| `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3` | `21.472` |
| Historical queue `best_compile_us` | `24.480000138282776` |

True floor: no. The oracle beats the coordinate-descent local config and the
historical queue best, but the required combo local config is faster. Queue
status should be diagnosis-only / `BANDWIDTH_BOUND`, with the main
`oracle_path` intentionally blank.
