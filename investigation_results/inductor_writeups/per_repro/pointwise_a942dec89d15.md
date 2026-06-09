# pointwise_a942dec89d15

## Queue Position

- Family: `layout_indexing_stencil_fusion` (expanded worklist inference)
- Owner: `Codex-template-layout-a942`
- Closure status: `not_true_floor`
- Oracle status: `implemented_not_true_floor`

## Current Gap

- Historical best compile: `4.927999805659056 us`
- Historical worklist best compile: `4.9 us`
- Historical SOL: `0.0 us`
- Effective bytes: `8`
- Oracle path: `repros/canonical/pointwise_a942dec89d15/oracle_layout_stencil.py`

## Oracle State

- Full-scope oracle: the oracle accepts `Repro.make_inputs()` unchanged and computes the complete `Repro.forward` result, `lift_fresh_copy(input) * 1`, as one Triton scalar load/store kernel.
- Check: `PASS` against eager `Repro()(*make_inputs())`.
- Bench command: `python repros/canonical/pointwise_a942dec89d15/oracle_layout_stencil.py --bench --warmup 10 --rep 50`
- Bench result: `oracle_us=3.30`, `compile_us=3.23`, `ratio=0.981`, `status=AT_FLOOR`.
- Oracle conclusion: not a true lower floor; it does not beat local compile, and the historical `4.9 us` value is best interpreted as the same launch-floor regime on an older/noisier run.

## Gap Diagnosis

- Classification: `BANDWIDTH_BOUND`
- Diagnosis: the expanded-worklist gap is caused by a near-zero SOL denominator for an 8-byte scalar CUDA materialization, not by a missed layout/stencil fusion opportunity.
- Inductor closure path: no scheduler fusion work is indicated for this repro; the compiled path is already a one-launch scalar materialization and timing is dominated by launch/CUDAGraph replay overhead.

## Local Compile Comparison

Command:

```bash
python scripts/bench_compare.py repros/canonical/pointwise_a942dec89d15/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo_mk3 --rounds 5 --n-warmup 10 --n-rep 50 --output /dev/stdout
```

Results:

- `cd`: `3.8079998921602964 us`, rounds `[3.871999913826585, 3.9679999463260174, 3.8079998921602964, 3.8399999029934406, 3.8399999029934406]`
- `combo_mk3`: `3.8079998921602964 us`, rounds `[3.8399999029934406, 3.8399999029934406, 3.8399999029934406, 3.8079998921602964, 3.8079998921602964]`
- `total_bytes`: `8`
- `rep_per_round`: `500`

## Done Criteria

- Canonical full-scope Triton oracle measured.
- Repro classified as `BANDWIDTH_BOUND`.
- `true_floor`: `no`
