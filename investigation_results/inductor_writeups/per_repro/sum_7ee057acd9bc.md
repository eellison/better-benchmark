# sum_7ee057acd9bc

## Queue Position

- Work queue rank: `551`
- Gap closure rank: `50`
- Family: `structured_pool_upsample_backward_reduce`
- Claim owner: `Codex-template-structured-7ee0`
- Oracle status: `active_subagent`

## Current Result

- Historical best compile: `446.4319944381714 us`
- Oracle path: `repros/canonical/sum_7ee057acd9bc/oracle_structured_scatter_reduce.py`
- Classification: `SCATTER_REDUCE`
- True floor: yes; the oracle beats historical best compile by `1.039x`.

## Scope

The oracle computes the exact `Repro()(*make_inputs())` output:

- output 0: `[192]` float32 channel reduction, contiguous stride `[1]`

The captured graph builds a zero `[196608, 729]` scatter target, decodes max-pool
offsets into input indices, scatter-adds `[196608, 169]` gradients, overwrites
masked output positions with the scalar `full`, reshapes to
`[1024, 192, 27, 27]`, then sums over batch and spatial dimensions.

The Triton oracle uses the algebraically equivalent direct reduction:
`sum(full where mask)` plus each pooled gradient only when its decoded scatter
destination mask is false. Timed work is Triton-only: a small zero-output kernel
and a gather-mask-reduce kernel with relaxed atomic accumulation into the final
channel vector.

## Benchmarks

- Correctness: `python repros/canonical/sum_7ee057acd9bc/oracle_structured_scatter_reduce.py --check` passes with max diff `1.25e-01`.
- Standard bench (`--bench --warmup 10 --rep 50`, `coordinate_descent_tuning=True`): oracle `429.47 us`, compile `835.81 us`, ratio `1.946x`, status `GOOD`.
- Combo config (`combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`): oracle `429.34 us`, compile `842.98 us`, ratio `1.963x`, status `GOOD`.
- Historical gate: oracle `429.47 us` vs historical best compile `446.4319944381714 us`; pass.

## Inductor Closure Path

- Implementation track: structured max-pool backward scatter-reduce fusion.
- Candidate hook: recognize `_low_memory_max_pool_offsets_to_indices` feeding
  `scatter_add`, `where(mask, full, scatter)`, and a channel/spatial reduction,
  then emit a gather-mask-reduce producer directly for the live reduction.

## Parent Integration Values

- Suggested `canonical_oracle_path`: `repros/canonical/sum_7ee057acd9bc/oracle_structured_scatter_reduce.py`
- Suggested `oracle_us`: `429.47`
- Suggested classification/status: `SCATTER_REDUCE`, true canonical oracle measured
- Shared CSVs were not edited.

## Commands

```bash
python -m py_compile repros/canonical/sum_7ee057acd9bc/oracle_structured_scatter_reduce.py
python repros/canonical/sum_7ee057acd9bc/oracle_structured_scatter_reduce.py --check
python repros/canonical/sum_7ee057acd9bc/oracle_structured_scatter_reduce.py --bench --warmup 10 --rep 50
python scripts/validate_corpus_invariants.py
```
