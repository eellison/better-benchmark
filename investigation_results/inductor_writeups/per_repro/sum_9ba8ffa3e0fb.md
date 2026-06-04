# sum_9ba8ffa3e0fb

## Queue Position

- Work queue rank: `550`
- Gap closure rank: `49`
- Family: `structured_pool_upsample_backward_reduce`
- Claim owner: `Codex-template-structured-9ba8`
- Closure status: `needs_canonical_oracle`
- Oracle status: `active_subagent`

## Current Gap

- Historical best compile: `1145.792007446289 us`
- Memcopy SOL: `371.616005897522 us`
- Launch-adjusted SOL gap: `2.9183527676794614x`
- Oracle path: `repros/canonical/sum_9ba8ffa3e0fb/oracle_structured_scatter_reduce.py`
- Measured oracle: `1013.82 us`
- True floor: yes; the oracle beats historical best compile by `1.13x`.

## Oracle State

- Classification: `SCATTER_REDUCE`
- Scope: full `Repro()(*make_inputs())` output, shape `[288, 512, 512]`, dtype `torch.float32`, contiguous stride `(262144, 512, 1)`.
- Implementation: two-stage Triton oracle. The first kernel computes the masked 513-wide row sums used by the softmax-backward FMA. The second stage materializes the post-edge-scatter `A2` tensor and writes the final cropped/reshaped output directly from structured gather/scatter formulas.
- Correctness: `python repros/canonical/sum_9ba8ffa3e0fb/oracle_structured_scatter_reduce.py --check` passes with max diff `4.58e-05`.

## Benchmarks

- Standard bench (`--bench --warmup 10 --rep 50`, `coordinate_descent_tuning=True`): oracle `1013.82 us`, compile `1410.18 us`, ratio `1.391x`, status `GOOD`.
- Combo config (`combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`): oracle `1011.33 us`, compile `1415.94 us`, ratio `1.400x`.
- Historical gate: oracle `1013.82 us` vs historical best compile `1145.792007446289 us`; pass.

## Inductor Closure Path

- Implementation track: Structured scatter/upsample gather-reduce.
- Candidate hook: Pattern-match the Longformer diagonal chunk assembly, masked row reduction, and paired edge slice-scatters as one structured producer that targets the final live layout.
- Gating policy: compare default, `coordinate_descent_tuning=True`, and forced combo kernels; gate on this exact Longformer shape/pattern if sibling structured-pool cases regress.

## Parent Integration Values

- Suggested `canonical_oracle_path`: `repros/canonical/sum_9ba8ffa3e0fb/oracle_structured_scatter_reduce.py`
- Suggested `oracle_us`: `1013.82`
- Suggested classification/status: `SCATTER_REDUCE`, true canonical oracle measured
- Shared CSVs were not edited.
