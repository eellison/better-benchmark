# sum_sum_sum_ecba845b8a4b

- **Gap**: 1.50x (confirmed with CUDAGraph measurement)
- **Classification**: SCHEDULER_FUSION
- **Root cause**: Grouped BatchNorm backward with channels_per_group=2. Same family as 80113e346555 -- K3 re-reads 3MB of data already loaded by K0. Retiling from (batch, channel) to (batch, group) would allow a single-pass kernel that performs the reduction and writes the epilogue together.
- **Fix approach**: Scheduler retiling for grouped patterns: detect when a reduction node and a pointwise node share the same input but tile along different axes, and retile to the coarser grouping that enables fusion.
- **Status**: DESIGN_TODO
- **Affected repros**: ~5 grouped-BN-backward variants
