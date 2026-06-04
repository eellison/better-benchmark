# sum_ad4468aef9e0

## Status

- Family: `multi_output_reduction_templates`
- Claimed owner: `Codex-bottom-multi-ad44`
- Classification: `BANDWIDTH_BOUND`
- Diagnostic artifact: `repros/canonical/sum_ad4468aef9e0/oracle_multi_output_reduction.py`
- Oracle status: diagnosis-only, not a true floor
- Parent integration note: leave the main queue `oracle_path` blank for true-floor tracking.

## Full-Scope Contract

The oracle consumes the same three strided `float32[32, 12, 512, 64]` inputs and
five shape parameters as `repro.py`, and returns the same single contiguous
`float32[2304]` output with stride `(1,)`.

The repro computes:

```text
out[0:768]       = sum_{batch, seq} getitem_20[batch, head, seq, dim]
out[768:1536]    = sum_{batch, seq} getitem_21[batch, head, seq, dim]
out[1536:2304]   = sum_{batch, seq} getitem_22[batch, head, seq, dim]
```

where `head = col // 64` and `dim = col % 64`. The timed Triton path indexes
the original strided inputs directly as storage-contiguous `[batch, seq, 768]`
streams, accumulates all three output segments in the same reduction tile, and
finalizes directly into the returned vector. It is not a reduction-subset
benchmark over a pre-materialized cat.

## Measurements

Measured with:

```bash
python repros/canonical/sum_ad4468aef9e0/oracle_multi_output_reduction.py --bench --warmup 10 --rep 50
```

- Triton full-scope oracle: `56.576 us`
- `torch.compile` with `coordinate_descent_tuning=True`: `56.160 us`
- `torch.compile` with `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `59.296 us`
- Historical queue `best_compile_us`: `37.82400116324425 us`

The oracle loses the local coordinate-descent config and the historical-best
gate, so it must be treated as diagnosis-only rather than a true performance
floor.

## Gap Diagnosis

The oracle differs from Inductor by reducing the three original Q/K/V-style
strided inputs directly into the concatenated output segments, skipping the
logical `permute -> view -> cat -> view` materialization. Inductor can already
lower this graph as an equivalent fused cat-sum reduction over the original
inputs; on this machine, the coordinate-descent compile is slightly faster than
the hand-written Triton oracle.

There is no evidence here for a missing scheduler fusion, scatter-reduce,
cooperative split-K, algebraic-elimination, or recompute-fusion optimization.
The relevant Inductor change would be narrower tuning of the existing cat-sum
reduction template for this storage-contiguous QKV layout. The classification
is `BANDWIDTH_BOUND`.

## Validation

- `python -m py_compile repros/canonical/sum_ad4468aef9e0/oracle_multi_output_reduction.py`: passed
- `python repros/canonical/sum_ad4468aef9e0/oracle_multi_output_reduction.py --check`: passed; max abs `9.155273e-05`, max rel `8.786187e-05`, stride matched `(1,)`
- `python repros/canonical/sum_ad4468aef9e0/oracle_multi_output_reduction.py --bench --warmup 10 --rep 50`: completed; diagnosis-only
