# sum_sum_sum_64f701d26f0a
## Oracle: oracle_multi_output_reduction (BEiT layer-norm backward)
## Compile: 206.7us, Oracle: 172.4us, Gap: 1.20x
## Diagnosis: SCHEDULER_FUSION
## What the oracle does differently: Implements a two-phase schedule: Phase 1 does a single Triton pass per row computing dual row reductions (sum and dot-product over D=768); Phase 2 uses one tiled pass that recomputes pointwise terms, writes the returned gradient matrix, and feeds four column accumulators -- avoiding materialization of intermediate tensors.
## Inductor fix: Scheduler/codegen support for dependent multi-output reduction templates that carry row-reduction outputs into a second tiled reduction while fusing same-base column accumulators and the materialized transpose side output under an autotuned shape guard.
