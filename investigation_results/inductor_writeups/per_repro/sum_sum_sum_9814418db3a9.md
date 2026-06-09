# sum_sum_sum_9814418db3a9


## Measured Timings
- Oracle: measurement failed (runtime error)
- Compile (CDT): 63.36 us
- Ratio: N/A

## Classification: BROKEN_ORACLE

## Oracle: oracle_cooperative_split_k.py

## Measurements

- Check: PASS (shape [512, 25088])
- Bench: FAILED (CUDA graph capture error)

## Diagnosis

The oracle passes correctness checking but fails during benchmarking with: "RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned." The oracle's cooperative split-K kernel uses a CPU tensor operation (`fmod.detach().cpu().tolist()`) during CUDA graph capture, which is incompatible with the CUDAGraph-based benchmarking harness.

## Status: broken_oracle (benchmark infrastructure incompatibility)

## Details
- Error occurs in fmod_values computation during CUDA graph capture
- The oracle kernel itself works (correctness PASS) but cannot be measured with CUDAGraph timing
- Would need oracle to pin memory or avoid CPU<->GPU copies in the hot path
