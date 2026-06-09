# sum_sum_sum_6770dc6efbe8
## Oracle: oracle_multi_output_reduction (Swin patch-expand backward)
## Compile: 232.1us, Oracle: 286.8us, Gap: 0.81x (compile wins)
## Diagnosis: ALREADY_FIXED
## What the oracle does differently: Attempts a cooperative split-K producer that simultaneously writes the [401408, 128] buffer and accumulates the sibling [128] column reduction via atomic/partial coordination, avoiding a re-read of the materialized buffer.
## Inductor fix: Already at floor -- current Inductor with scalar_acc + linear_reduction_elimination + CD tuning outperforms this oracle's atomic-heavy approach. The oracle's cooperative strategy incurs overhead from atomic accumulation across 100K+ spatial positions that exceeds the bandwidth saved by avoiding one re-read.
