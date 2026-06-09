# sum_0becf9609ad7
## Compile: 86us, Oracle: 86us, Gap: 1.00x
## Diagnosis: COOPERATIVE_SPLIT_K
## Root cause: Oracle cooperative split-K reduction over broadcasted avgpool-backward channel sum achieves the same latency as Inductor combo_persistent; the pattern is already at its performance floor on B200.
## Fix path: No further action needed; Inductor with persistent reduction already matches the cooperative split-K oracle within measurement noise.
## Status: at_floor
