Gap diagnosis (classification: BANDWIDTH_BOUND): the full-scope diagnosis
oracle computes the slice `mm_5[:, 64:100]`, preserves duplicate-index
`index_put(accumulate=True)` semantics, and materializes the returned
`float32[2048, 9, 9]` tensor with a Triton gather-reduce over the 36 source
columns. Inductor's tuned path keeps the natural zero-fill plus indexed
accumulate decomposition; although Inductor does not have a scatter-reduce
gather lowering for this pattern today, that missing lowering is not an
actionable gap for this repro because the best required coordinate-descent
compile is already slightly faster than the full-scope oracle. The remaining
cost is the launch/allocation and required tiny output/input materialization,
so the appropriate disposition is not a new scatter-reduce optimization.

Status: `not_true_floor`. `oracle_index_put.py --check` passes with max diff
`0.00e+00`. The requested oracle benchmark measured `oracle_us=6.82`,
`compile_us=6.40`, ratio `0.939`, `BAD_ORACLE`, and the required interleaved
compile comparison measured `coordinate_descent_tuning=True` at `6.27 us` and
the combo-required config at `7.10 us`. Since the best required compile is
faster than the full-scope oracle, this should not be wired as a canonical true
floor.

CSV recommendation: leave the coordination CSV unedited here. Recommended
parent status is `not_true_floor`, with no canonical floor path; note
`diagnosis oracle exists at repros/canonical/pointwise_be855b961be7/oracle_index_put.py;
classification=BANDWIDTH_BOUND; full-scope check PASS; oracle=6.82us;
cdt=6.27us; combo=7.10us; true_floor=no`.
