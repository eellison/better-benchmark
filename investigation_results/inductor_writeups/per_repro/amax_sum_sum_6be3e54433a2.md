# amax_sum_sum_6be3e54433a2

## Current Result

- Family: `two_class_softmax_backward_index_put_pad`
- Classification: `SCATTER_REDUCE`
- Oracle path: `repros/canonical/amax_sum_sum_6be3e54433a2/oracle_online_softmax.py`
- Correctness: PASS
- Oracle: `5.632 us`
- `torch.compile coordinate_descent_tuning=True`: `10.016 us` from the oracle harness, `7.74399982765317 us` from interleaved `bench_compare`
- `torch.compile combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `20.576 us` from the oracle harness, `7.071999832987785 us` from interleaved `bench_compare`
- True floor: yes
- Parent status: `implemented_unmeasured`

## Scope

The oracle covers the full `Repro.forward` output, not only the two-class softmax rows. It computes the scalar-scaled masked `[32,2]` softmax-backward-like update, adds `arg544_1`, accumulates duplicate-safe indexed updates into the zero logical `[32,128,2]` destination using `arg221_1` and `arg418_1`, and returns the final materialized contiguous `[4,4096]` tensor after the equivalent view, permute, and two padded zero rows.

## Diagnosis

Gap diagnosis (classification: SCATTER_REDUCE): the custom Triton path keeps the row-local two-class softmax-backward arithmetic and the indexed accumulate producer together, then writes directly into the returned `[4,4096]` layout with rows 2 and 3 zero-filled, whereas Inductor schedules the reductions, `index_put(accumulate=True)`, view/permute, and pad as generic kernels over an intermediate `[32,128,2]`; the missing compiler capability is a duplicate-safe indexed scatter-add lowering whose producer can target a layout-changing padded output directly.

## Commands

```bash
python repros/canonical/amax_sum_sum_6be3e54433a2/oracle_online_softmax.py --check
python repros/canonical/amax_sum_sum_6be3e54433a2/oracle_online_softmax.py --bench --warmup 10 --rep 50
python scripts/bench_compare.py repros/canonical/amax_sum_sum_6be3e54433a2/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo --n-warmup 10 --n-rep 50 --rounds 5 --output /tmp/amax_sum_sum_6be3e54433a2_bench_compare.json
python -m py_compile repros/canonical/amax_sum_sum_6be3e54433a2/oracle_online_softmax.py
python scripts/validate_corpus_invariants.py
```
