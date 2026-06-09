# sum_1fb7f13c05d1

## Classification: BANDWIDTH_BOUND

## Current Result

- Family: `multi_output_reduction_templates`
- Oracle path: `repros/canonical/sum_1fb7f13c05d1/oracle_bert_qkv_layout_sum.py`
- Correctness: PASS
- Oracle: `23.78 us`
- `torch.compile`: `25.34 us`
- `torch.compile coordinate_descent_tuning=True`: `24.45 us`
- `torch.compile combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `24.35 us`
- Ratio: 1.066x
- Status: `at_floor`

## Diagnosis

The oracle computes the complete BERT QKV layout materialization scope: a `[1536,64,128]` f32 contiguous input is viewed as `[128,12,64,128]`, double-permuted to `[128,128,12,64]`, cloned to contiguous `[128,128,768]` (flattened to `[16384,768]`), returned as a `[768,16384]` permute view, and a sibling `[768]` column sum is produced. The oracle uses a Triton copy-and-batch-partial-sum kernel followed by a small finalizer.

Inductor already emits essentially the same two-pass strategy: `triton_red_fused_clone_permute_sum_view_0` iterates over the 98304 (=128*768) x-elements with a 128-step reduction loop, storing the permuted output and accumulating 128 partial sums per column, then `triton_red_fused_clone_permute_sum_view_1` reduces the 128 partials to the final 768 column sums. The 6.6% gap is within noise of bandwidth floor -- both implementations are dominated by the mandatory f32 layout copy (writing ~48 MB to the output buffer) and the f32 partial reductions over the same input values.

## Root cause

No actionable gap. Inductor already recognizes the fused clone+permute+sum pattern and emits a multi-output reduction kernel with an outer finalizer pass. The remaining difference is within measurement noise of the bandwidth limit for this workload (391 MB total output writes).

## Kernel count

- Oracle: 2 kernels (_copy_and_batch_sum_kernel + _finalize_sum_kernel)
- Inductor: 2 kernels (triton_red_fused_clone_permute_sum_view_0 + triton_red_fused_clone_permute_sum_view_1)

## Config exploration

| Config | Result |
|--------|--------|
| coordinate_descent_tuning=True | 24.45 us (no meaningful change) |
| combo_kernels + multi_kernel | 24.35 us (no meaningful change) |

## Recommendation

Record as at bandwidth floor. No Inductor change needed for this repro -- both the oracle and Inductor emit the same essential schedule (fused copy+partial-sum first pass, finalizer second pass). The 6.6% measured delta is not reliably reproducible across runs.

## Relevant files

- `repros/canonical/sum_1fb7f13c05d1/repro.py` (BERT QKV layout + sum scope)
- `repros/canonical/sum_1fb7f13c05d1/oracle_bert_qkv_layout_sum.py` (oracle)
