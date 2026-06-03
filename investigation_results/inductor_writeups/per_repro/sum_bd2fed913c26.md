# sum_bd2fed913c26

## Current Result

- Family: `multi_output_reduction_templates`
- Classification: `SCHEDULER_FUSION`
- Oracle path: `repros/canonical/sum_bd2fed913c26/oracle_multi_output_reduction.py`
- Correctness: PASS
- Oracle: `41.440 us`
- `torch.compile coordinate_descent_tuning`: `48.256 us`
- `torch.compile combo_looped_cd`: `50.048 us`
- Valid floor: yes, the full-scope Triton oracle is faster than both required compile configs.

## Diagnosis

The oracle covers the full BERT repro scope by consuming the original `[384, 512, 64]` input, applying the scalar multiply, materializing the contiguous clone backing storage for the returned `[768, 16384]` tensor with stride `(1, 768)`, and computing the returned `[768]` feature sum from the same input pass. It differs from Inductor by fusing the layout-copy output and the reduction output over the same logical source iteration space; Inductor currently schedules the scalar/layout materialization and sum as separate work because the two returned values have different layouts and the scheduler has no materializing multi-output reduction template for this pattern. This is a `SCHEDULER_FUSION` gap rather than a reduction-only microbenchmark: the oracle returns the same original-input signature, output count, shapes, dtypes, and strides as `repro.py`.

## Commands

```bash
python -m py_compile repros/canonical/sum_bd2fed913c26/oracle_multi_output_reduction.py
python repros/canonical/sum_bd2fed913c26/oracle_multi_output_reduction.py --check
python repros/canonical/sum_bd2fed913c26/oracle_multi_output_reduction.py --bench --warmup 10 --rep 50
```
