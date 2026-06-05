# pointwise_604e62335f7c — MobileNet BN Residual

## Summary
- **Ratio**: 0.872x (BAD_ORACLE)
- **Classification**: BAD_ORACLE
- **Status**: Oracle is slower than torch.compile

## Benchmark Results
- Oracle: 8.99us
- Compile: 7.84us

## Analysis

The oracle is 13% slower than Inductor's compiled output. This indicates the oracle kernel is suboptimal for this particular pattern -- Inductor's generic codegen with autotuning finds a better configuration than the hand-written oracle.

This repro involves a MobileNet BN + residual pattern with stochastic ops. Inductor's combo_kernels or coordinate_descent_tuning likely finds a better tile size or warp count for the specific tensor dimensions.

## Conclusion

No action needed. The oracle needs to be updated to match or beat Inductor's performance, but there is no Inductor optimization gap here.
