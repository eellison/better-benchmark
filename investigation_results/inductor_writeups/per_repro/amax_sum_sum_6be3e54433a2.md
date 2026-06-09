# amax_sum_sum_6be3e54433a2

## Compile: 8.96us (best w/ combo), Oracle: 7.04us, Gap: 1.273x

## Diagnosis: SCATTER_REDUCE

## Root cause: Inductor emits 3 kernels for this GPT-Neo sequence classification backward pattern: (1) a fused softmax-backward with row reductions, (2) an index_put (scatter-add) kernel for the [32, 128, 2] accumulation, and (3) a view/permute/pad kernel producing the final [4, 4096] output. The oracle fuses all of this into 2 kernels: a zero-fill plus a single-CTA kernel that computes the full softmax-backward arithmetic on the tiny [32, 2] rows, atomically scatters the results into the [4096, 2] destination, and directly produces the padded/permuted output layout. The data volumes are extremely small (32 rows x 2 classes), so the gap is dominated by kernel launch overhead (3 launches vs 2) rather than memory bandwidth.

## Fix path: Teach Inductor to fuse tiny row-local softmax-backward arithmetic with the downstream index_put(accumulate=True) scatter and the layout-changing view/permute/pad into a single kernel when the scatter destination is small and the index_put has few conflicting writes. This requires the scheduler to recognize that the index_put consumer is just a scatter-add to a small buffer that can be written atomically, and that the subsequent permute/pad is a pure layout transformation that can be folded into the store offset calculation.

## Status: design_todo

## Details

- Model: hf_GPTNeoForSequenceClassification_train (2 shapes)
- Pattern: amax+sum+sum reduction (softmax-backward + scatter-add + permute/pad)
- Shape: [32, 2] logits -> softmax backward -> index_put into [32, 128, 2] -> view [4096, 2] -> permute [2, 4096] -> pad [4, 4096]
- Data volumes are tiny: 32 rows x 2 classes = 64 elements of real computation
- Inductor kernels: 3 (softmax-backward, scatter, layout)
- Oracle kernels: 2 (zero-fill + fused scatter-softmax-backward)
- The combo_kernels config helps (11.1us -> 8.96us) but still leaves a 1.27x gap.
- The fundamental issue is Inductor's inability to fuse across index_put boundaries into the downstream layout ops.
- File references: /tmp/pytorch-work/torch/_inductor/scheduler.py, /tmp/pytorch-work/torch/_inductor/lowering.py (index_put lowering)
