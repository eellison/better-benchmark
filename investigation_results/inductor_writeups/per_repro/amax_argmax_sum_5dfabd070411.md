# amax_argmax_sum_5dfabd070411

## Current Result

- Family: `online_softmax_cross_entropy`
- Classification: `NEW_PATTERN`
- Oracle path: `repros/canonical/amax_argmax_sum_5dfabd070411/oracle_online_softmax.py`
- Correctness: PASS
- Oracle: `5.98 us`
- `torch.compile coordinate_descent_tuning=True`: `9.76 us`
- `torch.compile combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `7.74 us`
- Parent status: `implemented_unmeasured`

## Diagnosis

The oracle covers the full `Repro.forward` scope by consuming the original `mm_72`, `arg0_1`, `iota_1`, `arg342_1`, and `full_1` inputs, reducing `arg0_1 != 0` to the last nonzero token per batch, gathering the selected two logits through the dynamic row index, computing the ignore-index two-class log-softmax/NLL mean, and returning the same scalar loss plus contiguous `[32,2]` bool eq side output. It differs from Inductor by recognizing the complete sequence-classification tail as one fused Triton kernel, while Inductor currently schedules the argmax, gather, log-softmax/NLL/count, and side eq as generic reduction/pointwise work. The fix is to add a guarded lowering for argmax-selected two-class cross-entropy with side-mask output.

## Commands

```bash
python repros/canonical/amax_argmax_sum_5dfabd070411/oracle_online_softmax.py --check
python repros/canonical/amax_argmax_sum_5dfabd070411/oracle_online_softmax.py --bench --warmup 10 --rep 50
python scripts/bench_compare.py repros/canonical/amax_argmax_sum_5dfabd070411/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo --n-warmup 10 --n-rep 50 --rounds 5 --output /tmp/amax_argmax_sum_5dfabd070411_bench_compare.json
python -m py_compile repros/canonical/amax_argmax_sum_5dfabd070411/oracle_online_softmax.py
python scripts/validate_corpus_invariants.py
```
