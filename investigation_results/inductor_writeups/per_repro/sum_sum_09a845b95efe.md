# sum_sum_09a845b95efe

## Status

- Family: `layout_masked_two_sum`
- Closure status: `open_gap`
- Artifact: `repros/canonical/sum_sum_09a845b95efe/oracle_layout_masked_two_sum.py`
- Classification: `NEW_PATTERN`
- Label: `torchbench_hf_Reformer_train_014`

## Full-Scope Contract

The oracle consumes the same inputs as `repro.py` (one `f32[8, 4096, 256]` activation
tensor, one `b8[8, 64, 1, 1]` mask, and three shape parameters) and returns the same
two outputs: `f32[1, 64, 192]` and `f32[64, 1, 64]` reduction results.

## Timings

- Oracle: 13.98 us
- torch.compile (coordinate_descent_tuning=True, combo_kernels=True): 18.62 us
- Ratio: 1.332x
- multi_kernel=3 config: ~27.95 us (raw, no CUDAGraph)

## Kernel Count

- **Oracle**: 1 kernel (_two_sum_kernel) - single Triton program reduces both outputs
- **Inductor**: 2 kernels (triton_red_fused_0 and triton_per_fused_convert_element_type_div_mul_permute_slice_sum_view_1)

## Root Cause

The oracle computes both reduction outputs from a single kernel by:
1. Viewing the contiguous `[8, 4096, 256]` input as `[8, 64, 64, 256]`
2. Keeping the view/permute/masked-scale producer virtual (no materialization)
3. Computing both reductions (sum over dims [0,1] for the slice[:,:,:,64:256] output,
   and sum over dims [0,2] for the slice[:,:,:,0:64] output) in a single tiled pass

Inductor schedules this as two separate reductions because:
1. The two sums reduce over different axis combinations ([0,1] vs [0,2])
2. They consume different slices of the intermediate ([:,:,:,64:256] vs [:,:,:,0:64])
3. The scheduler does not recognize this as a shared-producer multi-output pattern
   where both consumers read from the same permuted/masked source

The key insight the oracle exploits is that both reductions read the same base data
(just different D-dimension slices) after the same view+permute+mask operation, so
a single program can compute both outputs with one pass over the source data.

## Config Exploration

| Config | Time (us) | Notes |
|--------|-----------|-------|
| combo+cdt (default harness) | 18.62 | 2 kernels |
| combo+cdt+multi_kernel=3 | ~28 | raw timing, worse |

## Design Doc

**Why Inductor cannot do this today**: The scheduler treats sibling reductions over
different axes of the same producer as independent nodes. When one reduces [0,1] and
the other reduces [0,2], they cannot be trivially merged into a single reduction
kernel. The `can_fuse` logic in `scheduler.py` does not have a path to recognize
that two reductions with different reduction dimensions but a shared logical source
could share input reads by tiling both outputs in a single launch.

**What enhancement is needed**: A multi-output reduction template in
`torch/_inductor/scheduler.py` or `torch/_inductor/choices.py` that:
1. Detects sibling reductions consuming slices of the same virtual (view/permute) producer
2. Recognizes that the different reduction axes can be tiled together when the
   D-dimension is split between outputs
3. Emits a single kernel with two accumulator sets and two store targets

**Affected files**: `torch/_inductor/scheduler.py` (can_fuse, score_fusion),
`torch/_inductor/ir.py` (multi-output reduction IR)

**Similar repros**: This pattern appears in Reformer attention backward passes
where view/permute/mask feeds multiple axis-different reductions.

## Validation

- `oracle_layout_masked_two_sum.py --check`: PASS
- `oracle_layout_masked_two_sum.py --bench`: ratio 1.332x, status GOOD
