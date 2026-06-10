# amax_sum_9940b361e5b4

## Compile: 322.4us, Oracle: 174.1us, Gap: 1.85x

## Diagnosis: NEW_PATTERN

## Root cause: Computes the full Longformer sliding-window attention assembly, key/query masking, online row softmax, and final padded output layout directly in Triton.

## Fix path: Add a Longformer sliding-window attention pattern lowering that fuses the structured band assembly with the softmax reduction epilogue and destination-layout scatter.

## Status: needs_work (STATUS CORRECTED 2026-06-10)

## Details

- Model: torchbench_hf_Longformer_infer_002
- Pattern: amax, sum reduction (236 ops)
- Oracle: oracle_online_softmax.py


## Re-measurement 2026-06-10 (fresh cache, CUDAGraph + GPU lock, B200)
- Oracle: 152.5us, Compile: 392.1us, **Ratio: 2.57x** — gap is real and large
- Prior "Status: implemented" was WRONG: no commit in /tmp/pytorch-work
  references this repro or a Longformer lowering; queue still says needs_work.
  Likely the oracle (not a fix) was what got "implemented".
- Config exploration: multi_kernel=2 and combo_kernels both neutral (391.9us) —
  kernel structure, not tuning.
- Compile emits ~12 kernels per shape config (36 def triton_ across 3 configs)
  for the 236-op graph: padded band assembly (constant_pad_nd + view +
  as_strided overlapping windows) materialized separately from the online
  softmax reduction and the masked scatter into the padded output layout.
- Assessment: genuine NEW_PATTERN family (Longformer band attention assembly).
  The oracle fuses assembly + online softmax + output layout in one pass. A
  generic fix needs Inductor to fuse overlapping-window as_strided producers
  into reduction consumers — related to the structured scatter/gather work but
  on the READ side. Park behind the scatter-elimination family; revisit when
  that lands (its gather-at-destination machinery may transfer).
