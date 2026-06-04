# sum_sum_e6b32bb1b384

## Oracle State

- Family: `multi_output_reduction_templates`
- Owner: `Codex-bottom-multi-e6b3`
- Diagnostic artifact: `repros/canonical/sum_sum_e6b32bb1b384/oracle_multi_output_reduction.py`
- Main queue `oracle_path`: intended blank
- Verdict: diagnosis-only, not a true floor
- Classification: `BANDWIDTH_BOUND`

## Scope

The diagnostic Triton candidate covers the full `repro.py` computation. It
consumes the same six tensor inputs plus shape parameters and returns the same
two outputs: contiguous fp32 `[512]` and fp32 `[512, 8192]` with stride
`(1, 512)`.

## Correctness

Command:

```bash
python repros/canonical/sum_sum_e6b32bb1b384/oracle_multi_output_reduction.py --check
```

Result: PASS.

- output 0: max abs `1.907349e-06`, max rel `3.344049e-05`, stride match true
- output 1: max abs `1.192093e-07`, max rel `1.182112e-02`, stride match true

## Measurements

Command:

```bash
python repros/canonical/sum_sum_e6b32bb1b384/oracle_multi_output_reduction.py --bench --warmup 10 --rep 50
```

- Diagnostic full-scope Triton candidate, `BLOCK_M=2`: `39.328 us`
- `torch.compile coordinate_descent_tuning=True`: `38.464 us`
- `torch.compile combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `37.152 us`
- Historical best compile from queue: `25.37599951028824 us`

The diagnostic candidate is slower than both required local compiled configs
and the historical best, so it must not be used as a true floor.

## Gap Diagnosis

The only plausible full-scope fused schedule reads the shared scaled-mask
producer once, computes the row-wise hidden reduction needed by the returned
transposed tensor, and simultaneously writes row-block partials for the sibling
column reduction before a small finalizer writes the returned `[512]` vector.
Inductor currently emits a split column reduction for output 0 and a separate
row-reduction/materialization kernel for output 1 because the two reductions
use different axes.

The measured fused candidate loses: row-block partial stores plus larger
register pressure cost more than the saved producer reread. The closure for
this repro is to treat the historical tuned split schedule as the realistic
floor and leave the queue's main oracle path blank. Any future cross-axis
fusion for this shape should be gated by a cost model that rejects this case.
