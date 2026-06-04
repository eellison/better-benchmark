# sum_sum_sum_87fd4ec7f051

## Oracle

- Path: `repros/canonical/sum_sum_sum_87fd4ec7f051/oracle_multi_output_reduction.py`
- Classification: `COOPERATIVE_SPLIT_K`
- true_floor: yes
- Hardware measured locally: NVIDIA GH200 480GB

## Gap Diagnosis

The oracle covers the full `Repro()(*make_inputs())` scope: it consumes the original Swin inputs, performs the `mm_201` window-unpartition mapping, computes the channels-last convolution normalization path, evaluates both dependent layernorm-backward row reductions, and returns the same five contiguous `f32[128]` reductions as eager.

What the oracle does differently: it uses one Triton producer pass over the full `[128, 56, 56, 128]` logical domain with five channel accumulators, computing the row-local C=128 reductions for both layernorm-backward formulas in registers and writing only `[ceil(401408/768), 5, 128]` partial sums before a small finalizer.

Why Inductor cannot do this today: the scheduler sees the window clone/reshape, first row reductions, dependent pointwise layernorm-backward expression, second row reductions, and five `sum([0, 1, 2])` consumers as separate schedulable regions. It does not have a cooperative split-K multi-output reduction template that keeps row-local reductions and sibling NHW channel accumulators in one coordinated plan.

## Measurements

Correctness:

```bash
python repros/canonical/sum_sum_sum_87fd4ec7f051/oracle_multi_output_reduction.py --check
```

Result: `PASS`; max diffs by output were `2.32e-03`, `1.10e-03`, `1.27e-02`, `1.95e-03`, `5.47e-02`.

Oracle vs local compile:

```bash
python repros/canonical/sum_sum_sum_87fd4ec7f051/oracle_multi_output_reduction.py --bench --warmup 10 --rep 50
```

Result:

```json
{"repro_id": "sum_sum_sum_87fd4ec7f051", "oracle_us": 299.81, "compile_us": 663.62, "ratio": 2.213, "status": "GOOD"}
```

Required local compile config comparison:

```bash
python scripts/bench_compare.py repros/canonical/sum_sum_sum_87fd4ec7f051/repro.py --config "coordinate_descent_tuning=True" --label "cdt" --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label "combo_multi3" --rounds 5 --n-warmup 10 --n-rep 50
```

Result: `cdt=655.26 us`, `combo_multi3=771.58 us`; fastest local compile config was `cdt`.

Historical CSV comparison: historical `best_compile_us=497.5680112838745`; oracle is `299.81 us`, a `1.660x` speedup with `197.758 us` remaining gap vs the historical best compile number.

## CSV Notes

`oracle_multi_output_reduction.py`; true full-scope Triton floor; `COOPERATIVE_SPLIT_K`; local GH200 timings: `oracle_us=299.81`, harness `compile_us=663.62`, required `cdt=655.26`, required `combo_multi3=771.58`; historical `best_compile_us=497.5680112838745`; true_floor yes.
