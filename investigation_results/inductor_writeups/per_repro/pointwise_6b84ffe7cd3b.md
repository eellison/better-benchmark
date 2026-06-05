# pointwise_6b84ffe7cd3b — HardSwish + Dropout

## Summary
- **Ratio**: 0.873x (BAD_ORACLE)
- **Classification**: BAD_ORACLE
- **Status**: Oracle is slower than torch.compile

## Benchmark Results
- Oracle: 15.62us
- Compile: 13.63us

## Analysis

The oracle is 12.7% slower than Inductor's compiled output. This is a hardswish activation + dropout pattern with stochastic ops. Inductor's autotuned kernel finds a significantly better configuration than the oracle for this specific workload.

## Conclusion

No action needed. The oracle needs to be improved to match Inductor's performance. There is no Inductor optimization gap -- Inductor is already faster.
