# amax_sum_cd5b9c2339af

## Classification: `NEW_PATTERN`

Artifact: `repros/canonical/amax_sum_cd5b9c2339af/oracle_online_softmax.py`

This is a true full-scope oracle for the compiled `repro.py` region. It
consumes the same six inputs as `Repro.forward`, computes the f32
`[192,128,128] -> [32,6,128,128]` view, f32 position-bias add, stable
last-dimension softmax, Inductor stateless `tl.rand(seed, flat_offset)` dropout,
dropout scaling, expand/view, and returns the final f32 `[192,128,128]` tensor
with stride `(16384, 1, 128)`. It is not a softmax-only subset.

The oracle differs from Inductor by using a shape-specialized row-blocked
online-softmax/dropout Triton template that groups eight K=128 rows per program
and writes the final permuted output stride directly. Inductor already fuses the
graph into one online-softmax/dropout kernel, but the generic persistent
reduction schedule selected by the required configs does not choose this
small-K row-grouped template. The Inductor fix is to add or extend an
online-softmax dropout/layout template for K=128 attention rows and gate the
`block_m=8, num_warps=4` schedule by shape.

`--check` compares against the compiled repro rather than eager `Repro.forward`
because the eager `prims.inductor_random` implementation intentionally ignores
the seed tensor and calls `torch.rand`, while the compiled target uses
`tl.rand(seed, flat_offset)`.

## Results

Commands:

```bash
python -m py_compile repros/canonical/amax_sum_cd5b9c2339af/oracle_online_softmax.py
python repros/canonical/amax_sum_cd5b9c2339af/oracle_online_softmax.py --check
python repros/canonical/amax_sum_cd5b9c2339af/oracle_online_softmax.py --bench --warmup 10 --rep 50
```

Correctness: `PASS`; output shape `[192, 128, 128]`, dtype `torch.float32`,
stride `[16384, 1, 128]`, max_abs `1.788139e-07`, max_rel `8.274892e-07`.

Timing (`--warmup 10 --rep 50`, CUDA graph median):

| Path | Time (us) |
| --- | ---: |
| Oracle full-scope Triton (`block_m=8,num_warps=4`) | `16.96000061929226` |
| `coordinate_descent_tuning=True` | `22.87999913096428` |
| `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3` | `19.071999937295914` |
| Historical best_compile_us from queue | `19.36000026762485` |

The oracle beats both required local compile configs and the historical-best
gate, so this is a true floor. Queue integration should use
`oracle_path=repros/canonical/amax_sum_cd5b9c2339af/oracle_online_softmax.py`,
`classification=NEW_PATTERN`, `status=implemented_unmeasured` (or the parent
queue's equivalent implemented status), and `main_oracle_us=16.96000061929226`.
