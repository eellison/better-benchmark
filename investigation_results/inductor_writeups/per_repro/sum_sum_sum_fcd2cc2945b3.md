# sum_sum_sum_fcd2cc2945b3

## Queue Position

- Rank: 546
- Family: `multi_output_reduction_templates`
- Owner: `Codex-bottom-multi-2`
- Closure status at claim time: `writeup_ready_oracle_active`
- Oracle status: full-scope attempt, `not_true_floor`

## Current Gap

- Queue best compile: `62.43199855089188 us`
- Queue memcopy SOL: `26.62399969995021 us`
- Launch-adjusted SOL gap: `1.2332490305176986x`
- Oracle path: keep blank in CSV unless a faster full-scope implementation replaces this attempt.

## Oracle State

- Artifact: `repros/canonical/sum_sum_sum_fcd2cc2945b3/oracle_multi_output_reduction.py`
- Scope: full `repro.py` scope, consuming `mm` plus the six BN/ReLU branch input groups and returning all 12 outputs in repro order.
- Correctness: `python repros/canonical/sum_sum_sum_fcd2cc2945b3/oracle_multi_output_reduction.py --check` passes for all outputs.
- Benchmark command: `python repros/canonical/sum_sum_sum_fcd2cc2945b3/oracle_multi_output_reduction.py --bench --warmup 10 --rep 50`
- Measured timings on this run: oracle `474.048 us`, `coordinate_descent_tuning=True` compile `105.280 us`, combo-looped-CD compile `84.416 us`.
- Follow-up oracle-only samples after compile: `501.792 us`, `480.224 us`.

## Gap Diagnosis

The attempted oracle fuses each of the six Adv-Inception BN/ReLU-backward channel groups as a multi-output reduction template: the `mm / 64` slice and branch activation are read for two sibling reductions, then the mask and centered input are recomputed in the dependent pointwise epilogue that writes the returned tensor and scale-gradient vector. This is the scheduler fusion shape Inductor would need if this repro had a true oracle gap, because the current graph exposes expanded/sliced `mm`, six masks, 12 reductions, and six dependent epilogues as separate scheduling regions. Classification for the attempted optimization: `SCHEDULER_FUSION`.

## Closure Recommendation

Do not promote this artifact as an oracle floor. The full-scope Triton implementation is substantially slower than the compiled repro on the same command, and the queue's recorded best compile is faster still. Treat this row as requiring either a materially better full-scope oracle design or closure as no measured oracle gap.
