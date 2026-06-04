# sum_sum_3219a09ab96a

## Queue Position

- Rank: 63
- Family: `structured_pool_upsample_backward_reduce`
- Owner: `Codex-template-structured-3219`
- Closure status: `needs_inductor_gap_closure`
- Oracle status: `canonical_oracle_measured`

## Current Gap

- Historical best compile: `326.4000117778778 us`
- Historical memcopy SOL: `36.768000572919846 us`
- Historical best/SOL gap: `8.877284777303775x`
- Historical launch-adjusted gap at 3 us/launch: `6.6929135487076525x`
- Oracle path: `repros/canonical/sum_sum_3219a09ab96a/oracle_structured_scatter_reduce.py`

## Oracle State

- Classification: `SCATTER_REDUCE`
- True floor: `yes`
- Full-scope invariant: the oracle returns both `Repro()(*make_inputs())` outputs and covers the max-pool offset decode, dense scatter-add semantics, channel slices, both `where` masks, nonzero `full` mask contributions, and both final channel reductions.
- Oracle implementation: Triton gather-mask-reduce that avoids materializing the `[131072, 729]` scatter buffer. Each partial reduction counts true mask positions for the `full` contribution and sums source values only when the decoded scatter destination is unmasked.

## Measurements

- Correctness command: `python repros/canonical/sum_sum_3219a09ab96a/oracle_structured_scatter_reduce.py --check`
- Correctness result: `PASS`; output 0 max diff `4.69e-02`, output 1 max diff `6.25e-02`.
- Oracle benchmark command: `python repros/canonical/sum_sum_3219a09ab96a/oracle_structured_scatter_reduce.py --bench --warmup 10 --rep 50`
- Oracle benchmark result: `oracle_us=234.08`, harness compile with `coordinate_descent_tuning=True` `compile_us=570.02`, ratio `2.435x`, status `GOOD`.
- Local compile comparison command: `python scripts/bench_compare.py repros/canonical/sum_sum_3219a09ab96a/repro.py --config "coordinate_descent_tuning=True" --label coord_descent --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo_cd_multi3 --n-warmup 10 --n-rep 50 --rounds 5`
- Local compile comparison result: `coord_descent=550.72 us`, `combo_cd_multi3=580.22 us`, fastest `coord_descent`.
- Historical comparison: oracle `234.08 us` is faster than historical best compile `326.4000117778778 us`, so this is not diagnosis-only.

## Inductor Closure Path

- Implementation track: Structured scatter/upsample gather-reduce.
- Candidate hook: Pattern-match low-memory max-pool backward `scatter_add` feeding channel slices, `where` masks, and sibling reductions, then lower it as a structured scatter-reduce producer that accumulates final channel sums directly.
- Required semantic detail: preserve nonzero `full` contributions for all true mask positions separately from source scatter contributions; a source contribution is skipped when its decoded destination mask is true.
- Benchmark policy: compare against coordinate descent and the combo/multi-kernel config above; prioritize the best measured local compile runtime and keep historical `best_compile_us=326.4000117778778` as the CSV comparison note.

## Done Criteria

- Canonical oracle measured: done.
- Next Inductor work: add or extend structured scatter-reduce lowering for max-pool offset decode with sibling masked reductions, gated on this pool shape and reduction structure.
