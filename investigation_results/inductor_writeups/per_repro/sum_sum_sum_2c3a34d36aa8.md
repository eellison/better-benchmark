# sum_sum_sum_2c3a34d36aa8
## Compile: 149us, Oracle: 114us, Gap: 1.31x
## Diagnosis: SCHEDULER_FUSION
## Root cause: Inductor schedules the sigmoid/multiply/add producer, global sum, per-(N,C) spatial sum, sigmoid-derivative epilogue, and channel sum as separate kernels; the oracle streams the shared [128,1536,12,12] producer once into per-(N,C) summaries and finalizes both scalar and [1536] channel outputs together.
## Fix path: Scheduler fusion: form one multi-output reduction that shares the pointwise producer, emits per-(N,C) partial summaries, and lowers scalar + channel finalizers together. Borderline gap (1.31x, 36us).
## Status: borderline
