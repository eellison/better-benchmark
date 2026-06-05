# pointwise_644fbe16a3f2 — BART Index Put Accumulate

## Summary
- **Ratio**: 0.950x (BAD_ORACLE)
- **Classification**: BAD_ORACLE
- **Status**: Oracle is slower than torch.compile

## Benchmark Results
- Oracle: 9.57us
- Compile: 9.09us

## Analysis

The oracle is 5% slower than Inductor's compiled output. The workload involves an index_put with accumulation producing `[1026, 1024]` f32 output. Inductor's autotuned kernel configuration is slightly better than the oracle's hand-tuned approach for this specific shape.

## Conclusion

No action needed. The oracle needs to be improved, but there is no Inductor optimization gap here.
