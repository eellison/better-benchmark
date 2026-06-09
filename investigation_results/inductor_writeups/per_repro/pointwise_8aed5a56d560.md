# pointwise_8aed5a56d560


## Measured Timings
- Oracle: 5.15 us
- Compile (CDT): 4.99 us
- Ratio: 0.97x

Full-scope oracle: `repros/canonical/pointwise_8aed5a56d560/oracle_layout.py`.

## Classification: `BANDWIDTH_BOUND`

The repro takes one shape parameter, creates
`torch.ops.prims.iota.default(4096, dtype=torch.int64, device='cuda:0')`,
unsqueezes it to `[1, 4096]`, and expands it to the requested shape. The oracle
keeps that full scope: it fills the fresh `int64[4096]` iota storage with a
Triton kernel and returns the same expanded output layout.

This is not a layout/stencil fusion gap. There is no input producer, consumer,
scatter, reduction, or algebraic simplification opportunity around the iota.
The remaining work is a launch plus a 32 KiB int64 output store. Inductor is
already in the same launch-floor regime, so no scheduler-fusion change is
recommended for this repro.

## Results

Commands:

```bash
python repros/canonical/pointwise_8aed5a56d560/oracle_layout.py --check
python repros/canonical/pointwise_8aed5a56d560/oracle_layout.py --bench --warmup 10 --rep 50
python scripts/bench_compare.py repros/canonical/pointwise_8aed5a56d560/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo_mk3 --n-warmup 10 --n-rep 50 --rounds 5 --output /tmp/pointwise_8aed5a56d560_compare.json
```

Correctness: `PASS`.

Check details:

| Output | Shape | Stride | Dtype | Result |
| --- | --- | --- | --- | --- |
| 0 | `[1, 4096]` | `(4096, 1)` | `torch.int64` | `PASS`, exact |

Timing:

| Path | Time (us) |
| --- | ---: |
| Oracle full-scope Triton iota | `3.20` |
| Template harness compile comparison | `3.30` |
| `coordinate_descent_tuning=True` | `3.648000070825219` |
| `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3` | `3.7120000924915075` |

Parent disposition: `implemented_unmeasured`. The oracle is a full-scope timed
floor for the current required CD/combo configs, but the margin is launch-scale
and the classification remains `BANDWIDTH_BOUND`; this should not be treated as
a scheduler-fusion or new-pattern Inductor gap.
