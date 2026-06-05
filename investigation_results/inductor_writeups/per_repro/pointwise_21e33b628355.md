# pointwise_21e33b628355

## Compile: 10.91us, Oracle: 11.81us, Gap: 0.924x (BAD_ORACLE)

## Diagnosis: BAD_ORACLE

## Root cause: The oracle (layout transpose) is slower than torch.compile output. Compile outperforms the oracle by ~8%.

## Status: closed_no_gap

## Details

- Model: layout transpose operation
- Pattern: pure pointwise transpose/copy [2048, 4096] bf16
- Shape: [2048, 4096] bfloat16, output stride [4096, 1] (contiguous)
- The oracle is slower than compile -- no investigation needed.
- Output exactly matches (max_diff=0.0, exact layout and values verified).
- Inductor's autotuned pointwise kernel already handles this simple copy/transpose optimally.
