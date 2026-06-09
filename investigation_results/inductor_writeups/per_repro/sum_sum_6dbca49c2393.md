# sum_sum_6dbca49c2393 (T5 LN/dropout backward)


## Measured Timings
- Oracle: 20.93 us
- Compile (CDT): 24.19 us
- Ratio: 1.16x

## Classification: BANDWIDTH_BOUND

## Scope

The diagnosis oracle at
`repros/canonical/sum_sum_6dbca49c2393/oracle_multi_output_reduction.py`
covers the full `repro.py` computation scope:

- inputs: the same eight tensors plus shape parameters from `make_inputs()`
- outputs: `[512]` float32 reduction and `[512, 4096]` float32 transpose view
- output strides: `[1]` and `[1, 512]`, matching eager
- timed work: Triton producer kernel plus Triton partial-finalizer kernel; no
  torch/eager computation is used for the timed oracle path

## What The Oracle Does Differently

The oracle fuses the three `[4096,512]` matmul-gradient adds, affine multiply,
row-local `sum(weighted * arg210)`, dropout-mask epilogue store, and sibling
column reduction `sum((mm0 + mm1 + mm2) * arg210 * arg211)` into one Triton
producer pass. The producer writes the required materialized side output base
buffer and records `[row_block, 512]` column partials; a small Triton finalizer
reduces those partials to the returned `[512]` vector.

## Why Inductor Cannot Do That Today

This schedule mixes a dependent row reduction, a materialized side output, and a
compatible column reduction that should be accumulated while the side output is
written. Inductor can schedule the component reductions and pointwise epilogue,
but it does not currently model this as one cooperative split-K multi-output
template with a side-output store plus column partial accumulation.

## Inductor Change That Would Fix It

The mechanical codegen feature would be cooperative split-K support for
dependent multi-output reductions: while producing the materialized epilogue
output, the kernel would also accumulate compatible column partials into a
workspace and run a tiny finalizer. For this repro, however, that feature should
not be claimed as a floor improvement unless it beats the historical compiled
best.

## Measurements

Command:
`python repros/canonical/sum_sum_6dbca49c2393/oracle_multi_output_reduction.py --bench --warmup 10 --rep 50`

| Path | Time (us) |
|---|---:|
| full-scope Triton oracle | 27.744000777602196 |
| compile: `coordinate_descent_tuning=True` | 37.31200098991394 |
| compile: combo + per-subkernel blocks + coordinate descent + benchmark combo + `triton.multi_kernel=3` | 32.92800113558769 |
| queue historical best compile | 24.480000138282776 |

Correctness:
`python repros/canonical/sum_sum_6dbca49c2393/oracle_multi_output_reduction.py --check`
passed. Maximum absolute differences were `6.103516e-05` for output 0 and
`3.814697e-06` for output 1.

## Status For Parent Integration

Status: diagnosis-only / not a true floor.

Reason: although the full-scope Triton oracle is faster than both local required
compile configs in this run, it is slower than the queue historical best
`best_compile_us=24.480000138282776`. Leave the main `oracle_path` blank in
shared queue integration; if desired, mention the diagnosis artifact path only
in notes.
