# pointwise_ce7169287d93


## Measured Timings
- Oracle: 3.07 us
- Compile (CDT): 2.88 us
- Ratio: 0.94x

## Classification: `BANDWIDTH_BOUND`

Artifact: `repros/canonical/pointwise_ce7169287d93/oracle_layout_stencil.py`

This is a full-scope oracle for the compiled `repro.py` region. The repro has
no inputs and returns exactly one CUDA scalar:
`torch.full([], 0.001953125, dtype=torch.float32, device='cuda')`. The oracle
does not benchmark a subset; it allocates the scalar output and launches one
Triton program that stores the same `float32` value.

The expanded-worklist family label points at layout/stencil fusion, but this
case is a degenerate scalar materialization. There is no stencil, layout chain,
or producer/consumer fusion boundary to optimize. The only measurable work is
one scalar output store behind one launch, so the appropriate diagnosis is
`BANDWIDTH_BOUND` / launch-floor behavior rather than a scheduler-fusion gap.

## Results

Commands:

```bash
python repros/canonical/pointwise_ce7169287d93/oracle_layout_stencil.py --check
python repros/canonical/pointwise_ce7169287d93/oracle_layout_stencil.py --bench --warmup 10 --rep 50
python scripts/bench_compare.py repros/canonical/pointwise_ce7169287d93/repro.py --config "coordinate_descent_tuning=True" --label cdt --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo_multi3 --n-warmup 10 --n-rep 50 --rounds 5 --output /tmp/pointwise_ce7169287d93_compare.json
```

Correctness: `PASS`.

Check details:

| Output | Shape | Dtype | Result |
| --- | --- | --- | --- |
| 0 | `[]` | `torch.float32` | `PASS`, max_diff `0.00e+00` |

Timing:

| Path | Time (us) |
| --- | ---: |
| Oracle full-scope Triton scalar store | `3.07` |
| Template harness compile comparison | `3.23` |
| `coordinate_descent_tuning=True` | `3.5200000274926424` |
| `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3` | `3.5200000274926424` |
| Historical best compile from expanded queue/interleaved data | `4.608000162988901` |

The Triton scalar-store oracle beats both required local compile configs and the
historical-best gate, so it is a true full-scope timed floor. The gap is only
launch-scale and does not imply a layout/stencil fusion optimization.

CSV notes:
`classification=BANDWIDTH_BOUND; true_floor=True; oracle_us=3.07; template_compile_us=3.23; coordinate_descent_tuning_us=3.5200000274926424; combo_required_us=3.5200000274926424; best_required_compile_us=3.5200000274926424; historical_best_compile_us=4.608000162988901; status=GOOD; oracle_path=repros/canonical/pointwise_ce7169287d93/oracle_layout_stencil.py`.
