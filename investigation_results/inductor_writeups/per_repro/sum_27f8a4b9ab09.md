# sum_27f8a4b9ab09

## Current Result

- Family: `multi_output_reduction_templates`
- Classification: `BANDWIDTH_BOUND`
- Oracle path: `repros/canonical/sum_27f8a4b9ab09/oracle_multi_output_reduction.py`
- Correctness: PASS
- Oracle: `36.512 us`
- `torch.compile coordinate_descent_tuning`: `35.552 us`
- `torch.compile combo_looped_cd`: `36.608 us`
- Valid floor: no, the full-scope oracle is slower than coordinate-descent compile.

## Diagnosis

The oracle covers the full NFNet repro scope by consuming the original `mm`
and contiguous `[128, 3072, 6, 6]` activation tensor, fusing the reshape/view,
broadcasted divide, GELU derivative, gamma multiply, and returned channel sum
into one Triton reduction over `[0, 2, 3]`. It differs from the eager graph by
never materializing the broadcasted pooled-gradient tensor or pointwise
intermediates, but the measured `coordinate_descent_tuning=True` compile is
already slightly faster on the same full scope. That means this repro does not
show a remaining actionable fusion gap for Inductor on the contiguous layout;
the practical classification is `BANDWIDTH_BOUND` / already at floor, and this
artifact should stay diagnosis-only rather than being recorded as a true oracle
floor.

## Commands

```bash
python -m py_compile repros/canonical/sum_27f8a4b9ab09/oracle_multi_output_reduction.py
python repros/canonical/sum_27f8a4b9ab09/oracle_multi_output_reduction.py --check
python repros/canonical/sum_27f8a4b9ab09/oracle_multi_output_reduction.py --bench --warmup 10 --rep 50
```
