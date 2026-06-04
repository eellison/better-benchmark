# pointwise_f9c1d1b08ddb

## Classification: `SCHEDULER_FUSION`

Artifact: `repros/canonical/pointwise_f9c1d1b08ddb/oracle_layout_stencil.py`

This is a true full-scope oracle for the compiled `repro.py` region. It
consumes the same two `f16[512,128,27,27]` inputs as `Repro.forward`, computes
both branch ReLUs, the channel-dimension cat, and the full
`prims._low_memory_max_pool_with_offsets.default(..., kernel=[3,3],
stride=[2,2], padding=[0,0], dilation=[1,1], ceil_mode=True)` result. It
returns both required outputs with eager-compatible contiguous layout:
`f16[512,256,13,13]` pooled values and `i8[512,256,13,13]` pool offsets. It is
not a pointwise-only, pool-only, or single-branch subset.

The oracle differs from Inductor by launching a branch-specialized Triton
stencil for each input branch. Each kernel applies ReLU in registers while
reading the 3x3 pool window and writes directly into the final concatenated
output channel range, so no full relu/cat producer needs to be materialized
before the max-pool-with-offsets stencil. The Inductor fix is scheduler/codegen
support for sinking cat branch producers into a following structured pool
stencil and writing the final concatenated value/offset layout directly.

## Results

Commands:

```bash
python repros/canonical/pointwise_f9c1d1b08ddb/oracle_layout_stencil.py --check
python repros/canonical/pointwise_f9c1d1b08ddb/oracle_layout_stencil.py --bench --warmup 10 --rep 50
```

Correctness: `PASS`.

Check details:

| Output | Shape | Dtype | Stride | Result |
| --- | --- | --- | --- | --- |
| 0 | `[512, 256, 13, 13]` | `torch.float16` | `[43264, 169, 13, 1]` | `PASS`, max_diff `0.00e+00` |
| 1 | `[512, 256, 13, 13]` | `torch.int8` | `[43264, 169, 13, 1]` | `PASS`, exact |

Timing (`--warmup 10 --rep 50`, CUDA graph median):

| Path | Time (us) |
| --- | ---: |
| Oracle full-scope Triton (`block_m=16,num_warps=4`) | `126.88000500202179` |
| `coordinate_descent_tuning=True` | `332.96000957489014` |
| `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3` | `330.6879997253418` |
| Historical best_compile_us from CSV | `312.1280074119568` |

The oracle beats both required local compile configs and the historical-best
gate, so this is a true floor.

CSV notes:
`classification=SCHEDULER_FUSION; true_floor=True; oracle_us=126.88000500202179; best_required_compile_us=330.6879997253418; coordinate_descent_tuning_us=332.96000957489014; combo_required_us=330.6879997253418; historical_best_compile_us=312.1280074119568; status=GOOD; oracle_path=repros/canonical/pointwise_f9c1d1b08ddb/oracle_layout_stencil.py`.
