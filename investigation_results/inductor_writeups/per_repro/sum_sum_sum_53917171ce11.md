# Gap Diagnosis: sum_sum_sum_53917171ce11 (scatter_reduce)

## Classification: SCATTER_REDUCE

## Measurements

| Config | Time (us) | Ratio vs Default |
|--------|-----------|------------------|
| Default | 50.9 | 1.00x |
| Combo Looped (CD) | 46.7 | 1.09x |
| Combo Persistent (CD) | 46.8 | 1.09x |
| Coord Descent (best) | 48.1 | 1.06x |
| Oracle (structured eager) | ~29.4 | ~1.73x |

- **Bytes accessed**: 155.7 MB (logical)
- **Best compile SOL**: 1.58 GB/s effective BW ratio (combo_looped)
- **Model**: timm_beit_base_patch16_224_train

## Pattern Description

BEiT average-pool backward via slice_scatter. The graph computes:
1. Layer-norm-style row gradient: weighted = mm * primals_221, row_sum, row_dot
2. Structured token scatter: unsqueeze/expand/div feeding slice_scatter into [128, 197, 768]
3. Two [768] column reductions from the same gradient
4. A returned [768, 25216] transposed side output

## Root Cause

Inductor materializes the full [128, 197, 768] slice_scatter result (37.5MB) and
schedules the reductions and returned permute as generic consumers that reread from
this materialized buffer. The slice_scatter is essentially broadcasting a scaled value
into token positions -- a structured scatter-reduce pattern.

The oracle derives the two [768] reductions directly from the pre-scatter row gradient
and token sums, writing only the required [768, 25216] transposed side output, without
materializing the full intermediate.

## Oracle Strategy

1. Compute the row gradient (layernorm backward formula) per row
2. During the same pass, accumulate partial column sums for both [768] outputs
3. Write the transposed [768, 25216] output directly from the row gradient
4. Never materialize the [128, 197, 768] slice_scatter result

## Inductor Fix Required

**SCATTER_REDUCE**: Add a structured average-pool-backward slice_scatter lowering that:
1. Recognizes unsqueeze/expand/div feeding slice_scatter as a structured scatter pattern
2. Accumulates column reductions from the source tile while computing
3. Emits any required side-output stores (the transposed layout store)

This is part of the broader SCATTER_REDUCE fix family that applies to pool/upsample
backward patterns.

## Generalizability

HIGH - This pattern (structured scatter from pool/upsample backward + sibling reductions)
appears across multiple vision models (BEiT, DeiT, ViT variants). The SCATTER_REDUCE
fix family covers ~8 repros in the corpus.

## Remaining Gap After CD

~17-19us (46.7us best compile vs ~29.4us oracle). The gap is the materialized
intermediate elimination.
