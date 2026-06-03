# sum_ab7572b27d2c

## Current Result

- Family: `multi_output_reduction_templates`
- Classification: `SCHEDULER_FUSION`
- Oracle path: `repros/canonical/sum_ab7572b27d2c/oracle_multi_output_reduction.py`
- Correctness: PASS
- Oracle: `41.248 us`
- `torch.compile coordinate_descent_tuning`: `48.576 us`
- `torch.compile combo_looped_cd`: `59.328 us`
- Valid floor: yes, the full-scope Triton oracle is faster than both required compile configs.

## Diagnosis

The oracle covers the full LayoutLM repro scope by consuming the original
strided `[32, 12, 512, 64]` input, materializing the contiguous clone backing
storage for the returned `[768, 16384]` tensor with stride `(1, 768)`, and
computing the returned `[768]` feature sum from the same input pass. It differs
from Inductor by fusing the layout-copy output and the reduction output over the
same source iteration space; Inductor currently schedules the materializing
clone and sum as separate work because the two returned values have different
layouts. This is a `SCHEDULER_FUSION` gap rather than a reduction-only
microbenchmark: the oracle returns the same output count, shapes, dtypes, and
strides as `repro.py`.

## Commands

```bash
python -m py_compile repros/canonical/sum_ab7572b27d2c/oracle_multi_output_reduction.py
python repros/canonical/sum_ab7572b27d2c/oracle_multi_output_reduction.py --check
python repros/canonical/sum_ab7572b27d2c/oracle_multi_output_reduction.py --bench --warmup 10 --rep 50
```
