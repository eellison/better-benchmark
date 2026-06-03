# Active Optimization Queue

Updated: 2026-06-03T14:15:19+00:00

## Tracking Files

- `investigation_results/oracle_gap_closure_queue.csv`: per-repro path to close compile-vs-oracle/SOL gap.
- `investigation_results/oracle_kernel_work_queue.csv`: oracle floor implementation/measurement queue.
- `investigation_results/inductor_optimization_per_repro_queue.csv`: slowdown list with assigned closure owners.
- `investigation_results/inductor_optimization_priority_queue.csv`: family-level gated implementation queue.

## Ownership Model

- Every top-80 slowdown row now has an owner or backlog marker in the gap-closure queue.
- Optimizations may improve one repro and regress another; each implementation must benchmark coordinate descent, forced persistent, and forced looped configs, then gate by pattern/shape when needed.
- Closing a row means either measured Inductor reaches the oracle/realistic floor, or the row has a written gated implementation path and blocker.

## Top-80 Owner Counts

- `Curie`: 39
- `Averroes`: 18
- `Fermi`: 10
- `Kepler`: 4
- `Confucius`: 4
- `Noether`: 3
- `Maxwell`: 2

## Top-80 Closure Status

- `needs_canonical_oracle`: 34
- `needs_oracle_measurement`: 33
- `oracle_active_before_gap_closure`: 8
- `needs_inductor_gap_closure`: 5

## Active Tracks

- `Poincare`: online-softmax stash validation and gated PR plan.
- `Turing`: softmax-backward/partitioned-scatter regression validation.
- `Kepler`: structured oracle repair/implementation.
- `Noether`: top structured/online per-repro gap-closure writeups.
- `Fermi`: ranks 31-40 plus multi-output per-repro writeups.
- `Curie`: ranks 41-80 norm/layout/long-tail gap-closure triage.
