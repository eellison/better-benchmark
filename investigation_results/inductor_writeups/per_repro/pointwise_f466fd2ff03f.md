# pointwise_f466fd2ff03f


## Measured Timings
- Oracle: 1598.46 us
- Compile (CDT): 1576.03 us
- Ratio: 0.99x

## Classification: BANDWIDTH_BOUND

## Current Result

- Family: `dual_pad_tile`
- Oracle path: `repros/canonical/pointwise_f466fd2ff03f/oracle_dual_pad_tile.py`
- Correctness: PASS (assumed from oracle docstring)
- Status: `at_floor`

## Diagnosis

The oracle hand-runs the full-scope fused output-space pad schedule for the shared `[16384, 50257]` input view, loading each valid element once and storing it into both the padded transpose `[50260, 16384]` and padded row-major `[16384, 50260]` outputs, whereas Inductor already emits the same one-kernel one-load/two-store materialization for this captured graph. Inductor cannot materially improve this isolated repro today because the returned tensors require the full input read, two full output writes, and the tiny zero tails with no removable intermediate left inside the scope.

## Root cause

The workload is a view/permute/two-constant-pad scope producing two large output tensors from one input. Both Inductor and the oracle emit a single kernel with 1 load and 2 stores per element -- the minimum possible traffic. No fusion or algebraic work remains.

## Kernel count

- Oracle: 1 kernel (_dual_pad_kernel, 2D tiled pointwise)
- Inductor: 1 kernel (equivalent fused dual-pad materialization)

## Config exploration

No config improvement possible for a bandwidth-floor single-kernel scope.

## Recommendation

Record as at-floor pad/layout case. No action unless broader memory-codegen improvements or consumer fusion outside this repro removes a required materialized output.

## Relevant files

- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (multi-output pointwise codegen)
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` (fusion of sibling stores)
