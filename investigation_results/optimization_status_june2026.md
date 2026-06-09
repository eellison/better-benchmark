# Inductor Investigation Status - June 2026

Snapshot from the current `investigations-june-2026` worktree on 2026-06-03.
The per-repro CSV is actively shared by multiple agents; counts below reflect
the current local file state, not a landed baseline.

## Queue Snapshot

Priority queue state:

| Rank | Optimization | Priority | Status | Current implementation target |
|---:|---|---|---|---|
| 1 | `online_softmax_large_row` | P0 | `active_validation_gated` | scalar accumulators, expanded `R0_BLOCK`, persistent threshold tuning |
| 2 | `softmax_backward_attention` | P0 | `active_regression_validation` | enable partitioned scatter to remove extreme atomic contention |
| 3 | `structured_pool_upsample_reduce` | P0 | `active_oracle_and_writeup` | eliminate dense scatter via gather-mask-reduce |
| 4 | `multi_output_reductions` | P0 | `active_oracle_and_writeup` | multi-output reduction fusion, scalar accumulators, BN/backward templates |
| 5 | `norm_templates_bn_ln_rms` | P1 | `queued_gap_closure` | investigated as bandwidth-bound for current target configs |
| 6 | `layout_stencil_functional_updates` | P1 | `implemented` | `slice_scatter` elision prototype; needs upstreaming/extensions |
| 7 | `combo_horizontal_tiny_graphlets` | P2 | `queued` | launch-adjusted combo partitioning cleanup |
| 8 | `irregular_gather_reduce` | P2 | `deferred` | isolated Demucs gather/cat/reduce pattern |

Per-repro queue: 1090 rows.

| Bucket | Count | Notes |
|---|---:|---|
| closed / already at floor | 502 | includes bandwidth-bound, near-floor, non-target gap, and unique-pattern closures |
| fix identified / verified class | 325 | includes scatter-reduce, split-K/cooperative, scalar-acc, algebraic-elim, and scatter-dominated rows |
| deprioritized launch floor | 222 | launch-adjusted gap too small or negative |
| verified at floor | 24 | explicit at-floor or slice-elision confirmations |
| oracle implemented, unmeasured in CSV | 11 | recent subagent tranche around ranks 182-204 |
| active subagent | 4 | currently claimed in the same tranche |
| implemented | 2 | layout stencil rows already marked implemented |

Family totals:

| Family | Rows | Current read |
|---|---:|---|
| `norm_template_canonicalization` | 355 | mostly closed bandwidth-bound; no config-level win found |
| `multi_output_reduction_templates` | 306 | largest open class after closures: split-K/cooperative plus scheduler/algebraic cases |
| `layout_indexing_stencil_fusion` | 178 | slice-scatter elision covers the main known pattern; some layout-store sinking remains |
| `structured_pool_upsample_backward_reduce` | 143 | large scatter-reduce opportunity, especially UNet upsample backward |
| `online_softmax_cross_entropy` | 106 | scalar-acc path covers large-row online softmax; recent rows expose additional softmax idioms |
| `irregular_gather_reduce` | 1 | deferred isolated pattern |
| `misc_review` | 1 | deprioritized |

## Actionable Optimization Classes

| Class | What the oracle does differently | What Inductor lacks today | Representative repros / oracles | Next implementation step |
|---|---|---|---|---|
| `SCHEDULER_FUSION` | Reads shared inputs once and feeds multiple accumulators, side outputs, or final layouts from one scheduled producer. | Scheduler/codegen cannot represent several profitable fusions: sibling reductions with different expressions, dependent multi-output reductions, side-effect scatter stores, or layout-only consumer indexing sunk into producer stores. | `sum_sum_ee85624361a0/oracle_multi_output_reduction.py`; `sum_sum_sum_64f701d26f0a/oracle_multi_output_reduction.py`; `sum_sum_sum_d0496bdeedba/oracle_fused_embedding_scatter.py`; `pointwise_cf3acd87ba9e/oracle_stencil_fusion.py` | Start with same-domain sibling reductions (`numel/rnumel` equal, shared bases, different expressions). Then add a dependent multi-output template that allows required materialized side outputs and layout-store sinking through reshape/permute/clone/split-only chains. |
| `SCATTER_REDUCE` | Avoids dense scatter/index_put materialization by iterating output positions, applying the mask, gathering structured source contributions, and reducing directly by channel. | No pass eliminates `scatter_add`/`index_put(accumulate=True)` when the scatter output only feeds masks and reductions; structured bilinear/maxpool mappings are treated as arbitrary index writes. | `sum_18262b26934c/oracle_maxpool_direct_reduce.py`; `sum_sum_sum_f90d684d32cb/oracle_structured_upsample_reduce.py`; `sum_sum_sum_45f02142ecfd/oracle_structured_scatter_reduce.py`; `sum_sum_sum_dadf6aa035dd/oracle_structured_scatter_reduce.py` | Implement an FX/post-grad match for `scatter/index_put -> optional pointwise/mask -> sum(over scatter dims)`. Lower first to a maxpool direct-reduce template, then to the bilinear upsample gather-mask-reduce template. |
| `COOPERATIVE_SPLIT_K` | Writes a large required side output while cooperatively accumulating a sibling small reduction across row tiles. | Reductions attached to materialized producers are kept as producer/consumer work; there is no split-K producer mode with partial or atomic coordination for the sibling reduction. | `sum_sum_sum_6770dc6efbe8/oracle_multi_output_reduction.py`; many `fix_identified_split_k_cooperative` multi-output rows | Add codegen support for pointwise/reduction producers that both store the returned buffer and accumulate a sibling reduction with atomics or partial buffers. Autotune split count and validate against high-rnumel multi-output rows. |
| `ALGEBRAIC_ELIMINATION` | Computes compact sibling summaries once and derives dependent reductions from those summaries instead of materializing/rereading large intermediates. | The graph is preserved literally when a later reduction depends on earlier reductions; Inductor does not prove GRN/linear-reduction identities and replace them with smaller summary reductions. | `sum_sum_sum_f68c9f1fa09b/oracle_multi_output_reduction.py`; existing `verified_algebraic_elim` rows | Extend the algebraic-elimination pass to GRN/GELU backward and linear dependent-reduction forms. Emit a multi-accumulator summary kernel plus small follow-up reductions. |
| `RECOMPUTE_FUSION` | Recomputes cheap pointwise terms inside a later pass instead of materializing and rereading an intermediate. | The scheduler lacks a costed "recompute cheap producer" option across reduction boundaries and materialized sibling outputs. | No rank-182-204 diagnosis landed as this class; it appears as a supporting tactic in relative-position and dependent-reduction patterns. | Treat as a cost-model extension after the concrete scheduler/scatter cases: allow recompute when producer FLOPs are cheap, input reuse is high, and the materialized tensor is large. |
| `BANDWIDTH_BOUND` | Confirms the generated kernels are already at the practical memory floor or dominated by required outputs. | No actionable scheduler/template change for the current pattern; further wins require a new semantic template outside the current investigation scope. | `pointwise_8389b1fec310/gap_diagnosis.txt`; norm-template rows such as `var_mean_765fb8f2c85e` | Stop spending oracle time on these unless a new semantic template is explicitly scoped. Keep them closed or verified-at-floor. |
| `NEW_PATTERN` | Rewrites a decomposed model idiom into a semantic template that avoids large intermediates. | Inductor does not canonicalize these model-level idioms into dedicated lowerings. Current scheduler sees generic reductions, pointwise ops, sort boundaries, or indexed scatter-reduces. | `amax_sum_sum_4c98004c3aa3/oracle_online_softmax.py`; `amax_sum_sum_6939e2db29e3/oracle_online_softmax.py`; `amax_sum_7fea03f0412b/oracle_online_softmax.py`; `argmax_mean_180a59791f51/oracle_reformer_lsh_routing.py`; `sum_sum_2dc2fbd588d9/oracle_relative_position_scatter_reduce.py` | Group before implementing. Immediate candidates are shifted ignore-index cross entropy, masked attention softmax with dropout/layout epilogue, Reformer LSH signed-argmax routing, and relative-position-bias fused materialized-producer plus indexed partial reduction. |

## Recent Oracle Tranche: Launch Ranks 182-204

Implemented/unmeasured rows in this band:

| Rank | Repro | Family | Class | Oracle / diagnosis |
|---:|---|---|---|---|
| 182 | `argmax_mean_180a59791f51` | online softmax / CE bucket | `NEW_PATTERN` | Reformer LSH routing avoids materialized `cat([x, -x])` before sort/gather. |
| 184 | `sum_sum_sum_d0496bdeedba` | structured pool bucket | `SCHEDULER_FUSION` | Layer-norm backward plus embedding scatter side outputs in one producer pass. |
| 185 | `sum_sum_ee85624361a0` | multi-output reductions | `SCHEDULER_FUSION` | Batch-norm backward dual accumulator over same domain. |
| 186 | `amax_sum_sum_4c98004c3aa3` | online softmax / CE | `NEW_PATTERN` | Shifted ignore-index cross-entropy mean via online row accumulators. |
| 187 | `amax_sum_7fea03f0412b` | online softmax / CE | `NEW_PATTERN` | Masked BERT attention softmax with dropout/layout epilogue. |
| 188 | `amax_sum_sum_6939e2db29e3` | online softmax / CE | `NEW_PATTERN` | Same shifted cross-entropy idiom at another shape. |
| 189 | `sum_sum_sum_64f701d26f0a` | multi-output reductions | `SCHEDULER_FUSION` | BEiT layer-norm backward as two-phase multi-output reduction with side output. |
| 190 | `pointwise_8389b1fec310` | layout/stencil | `BANDWIDTH_BOUND` | Inductor already fuses the pool path; remaining mask output is required. |
| 191 | `sum_sum_sum_f68c9f1fa09b` | multi-output reductions | `ALGEBRAIC_ELIMINATION` | GRN/GELU dependent reduction derivable from per-`(N,C)` summaries. |
| 192 | `sum_sum_2dc2fbd588d9` | structured pool bucket | `NEW_PATTERN` | Relative-position indexed reduction fused with required `fma` output. |
| 193 | `sum_sum_sum_6770dc6efbe8` | multi-output reductions | `COOPERATIVE_SPLIT_K` | Patch-expand/dropout producer should also reduce sibling `[128]` output. |
| 204 | `pointwise_cf3acd87ba9e` | layout/stencil | `SCHEDULER_FUSION` | Sink channel-shuffle layout indexing into the branch producers. |

Active but not completed in this rank band: `sum_sum_sum_88c9224fa05a`,
`sum_sum_sum_512dcf4a167b`, and `sum_sum_b7f94adef30f`.
`amax_sum_55ae6a130879` was removed from oracle-floor status because the
existing script is a softmax/dropout prototype, not a full-scope floor for the
compiled repro region.

## Recommended Next Steps

1. Finish validation and upstream plan for `online_softmax_large_row`: scalar
   accumulators, expanded `R0_BLOCK`, persistent threshold changes, and guards
   for small-rnumel regressions.
2. Start the P0 scatter-reduce implementation with the maxpool direct-reduce
   case, then extend to bilinear upsample gather-mask-reduce. This is still the
   largest absolute savings opportunity.
3. Implement the smallest `SCHEDULER_FUSION` slice first: same-domain sibling
   reductions with shared inputs and different expressions. Validate on
   `sum_sum_ee85624361a0` and existing BN-backward representatives.
4. Add a second scheduler/codegen path for dependent multi-output reductions
   with required side outputs. This covers `sum_sum_sum_64f701d26f0a`,
   `sum_sum_sum_d0496bdeedba`, and related layer-norm/backward cases.
5. Prototype cooperative split-K for materialized producers with sibling small
   reductions. Use `sum_sum_sum_6770dc6efbe8` as the first correctness target.
6. Extend algebraic elimination for GRN/GELU and linear dependent reductions;
   keep the proof narrow and shape-guarded.
7. Keep norm-template and confirmed layout floor rows closed. Do not spend more
   oracle capacity there unless a new semantic template is explicitly chosen.
8. Treat `NEW_PATTERN` rows as separate pattern proposals, not as generic
   scheduler work, until enough repeats are found to justify implementation.
