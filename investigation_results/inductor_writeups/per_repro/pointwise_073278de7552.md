# pointwise_073278de7552 - ReLU MaxPool Copy

## Benchmark Result
- Oracle: 8.1 us
- Compile: 9.44 us
- Ratio: 1.166x
- Status: GOOD

## Root Cause
The oracle fuses ReLU + low_memory_max_pool_with_offsets + copy_ into a single kernel launch. It computes ReLU inline for both the maxpool stencil (9 neighbors per output pixel) and the copy_ side-effect (writing ReLU result back to input buffer), all in one pass per spatial plane.

Inductor emits 2 kernels:
1. `triton_poi_fused__low_memory_max_pool_with_offsets_relu_0`: fused ReLU + maxpool (262144 elements = 64*64*8*8 output)
2. `triton_poi_fused_copy__relu_1`: separate ReLU + copy_ back to the input (1048576 elements = 64*64*16*16 input)

The issue is that the scheduler does not recognize that the copy_ mutation and the maxpool stencil both consume the same ReLU result. It schedules them as separate kernels, causing the input data to be loaded twice: once for maxpool and once for copy_.

## Kernel Count
- Oracle: 1 kernel (ReLU + maxpool + copy_ all in one)
- Inductor: 2 kernels (ReLU+maxpool, ReLU+copy_)

## Config Exploration
- `combo_kernels = True`: does not help (stencil consumer + mutation side-effect is not handled)
- `coordinate_descent_tuning = True`: helps autotuning but doesn't fix the kernel split

## File/Line References
- `/tmp/pytorch-work/torch/_inductor/scheduler.py`: the scheduler splits because copy_ is a mutation that aliases the input, while the maxpool is a stencil consumer with shifted index access. The scheduler cannot today fuse a stencil reduction consumer with an aliasing in-place copy side-effect around one common pointwise producer.
- `/tmp/pytorch-work/torch/_inductor/ir.py`: the copy_ mutation creates a "realize" barrier that prevents fusion with the maxpool consumer.

## Design Doc
Classification: SCHEDULER_FUSION

The scheduler needs enhancement to recognize when:
1. A pointwise producer (ReLU) feeds both a stencil consumer (maxpool) and a mutation side-effect (copy_)
2. The mutation target is the same buffer as the pointwise input
3. Both can be computed in a single pass over the input data

The fix would be in `scheduler.py`'s `can_fuse` logic: when a copy_ mutation writes back to the same buffer that a pointwise node reads from, and the pointwise node's output feeds another consumer, allow all three to be scheduled in one kernel. The stencil (maxpool) reads with shifted indices, so the kernel would need to iterate over both the full input plane (for copy_) and the output plane (for maxpool), but since both are derived from the same data, one pass suffices.

### Affected Repro Count
This pattern occurs in models using maxpool after ReLU with in-place residual updates. Likely 1-3 repros in the corpus.
