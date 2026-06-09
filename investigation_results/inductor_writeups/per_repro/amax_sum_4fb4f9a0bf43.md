# amax_sum_4fb4f9a0bf43

Gap diagnosis (classification: BANDWIDTH_BOUND): the oracle covers the full T5 additive-score softmax materialization from `repro.py`, including the fp16 `[H,2048,2048] + [1,H,2048,2048]` add rounded through fp16, fp32 last-dimension softmax, fp16 cast, and contiguous `[H,2048,2048]` output view, using one Triton row program per query row instead of Inductor's generated softmax schedule. Inductor can already keep this default shape in a single compiled kernel under the required combo-looped configuration, so the remaining difference is a hand-written row template rather than a missing fusion across surrounding ops; Inductor cannot do materially better through the current scheduler-fusion classes because the required two fp16 reads, one fp16 write, and 2048-wide exp/reduction dominate this repro. The fix class is BANDWIDTH_BOUND.

## Scope

Full captured repro scope:

- `bmm_34`: `f16[8, 2048, 2048]`
- `add_48`: `f16[1, 8, 2048, 2048]`
- Output: `f16[8, 2048, 2048]`, stride `(4194304, 2048, 1)`

The oracle includes the reshape/view semantics, additive score bias, fp32 softmax over the last dimension, fp16 materialization, expand, and final `[8, 2048, 2048]` view. It does not time eager/PyTorch as a floor.

## Correctness

Command:

```bash
python repros/canonical/amax_sum_4fb4f9a0bf43/oracle_online_softmax.py --check
```

Result: PASS. Shape, dtype, and stride match the eager repro output. Max abs diff was `3.051758e-05`; max relative diff was `1.694915e-02`.

## Measurements

Command:

```bash
python repros/canonical/amax_sum_4fb4f9a0bf43/oracle_online_softmax.py --bench --warmup 10 --rep 50
```

CUDA graph replay timings on the default repro shape:

| Metric | Time |
|---|---:|
| Full-scope Triton oracle | `59.552 us` |
| `torch.compile coordinate_descent_tuning=True` | `78.272 us` |
| `torch.compile combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3` | `60.096 us` |

## Floor Status

Valid full-scope floor on this run: yes. The oracle beats both required compile configs while covering the same computation scope, output dtype, and output stride as `repro.py`, but the margin over the combo-looped config is only `0.544 us`, so this row should be treated as bandwidth-bound rather than evidence for a larger missing fusion.
