# Gap Diagnosis: sum_sum_2dc2fbd588d9 (relative_position_scatter_reduce)

## Classification: NEW_PATTERN (fused materialized-producer + indexed partial-reduction)

## Measurements

| Config | Time (us) | Ratio vs Default |
|--------|-----------|------------------|
| Default | 222.2 | 1.00x |
| Combo Looped (CD) | 197.7 | 1.12x |
| Combo Persistent (CD) | 202.6 | 1.10x |
| Coord Descent (from realistic floor) | 140.3 | 1.58x |
| Oracle (handwritten Triton) | ~41 | ~5.4x |

- **Bytes accessed**: 236.1 MB (logical)
- **Best compile SOL**: 4.83 GB/s effective BW ratio (combo_looped)
- **Oracle SOL**: ~5.4 TB/s if achieving 41us
- **Model**: timm_swin_base_patch4_window7_224_train

## Pattern Description

Swin Transformer relative position bias backward. The graph computes:
1. Row-wise attention gradient: `fma(-div, row_sum(bmm * div), bmm * div)` over
   [512, 16, 49, 49] (batch=512 windows, heads=16, query=49, key=49)
2. The full fma output must be materialized (it's a returned tensor)
3. A relative-position bias gradient: sum over batch dimension then scatter-accumulate
   into a [169, 16] bias table using position indices

## Root Cause

Inductor emits:
1. One kernel to compute row sums plus the materialized fma output
2. A copy kernel for the bias-table base
3. A SECOND reduction kernel that RE-READS the inputs and RECOMPUTES fma before the
   indexed index_put(accumulate=True)

The recomputation in kernel 3 is the main waste -- it rereads 236MB of data that was
already computed in kernel 1. The reason is that the profitable fusion crosses two
different reductions AND a materialized sibling output:
- First reduction: per attention row over keys (dim=-1)
- Second reduction: batch reduction followed by duplicate-index accumulation into [169, 16]

## Oracle Strategy

Single Triton kernel with grid = (cdiv(512, 64), 16, 49):
1. For each (batch_block, head, query) tile:
   - Load bmm and div values
   - Compute row_sum = sum(bmm * div) over key dim
   - Compute fma = bmm*div - div*row_sum
   - Store fma to output (materialized side output)
   - Accumulate partial sums over the batch block
2. Scatter-accumulate the batch-reduced partials into the bias table using atomic_add
   indexed by the position bias index

One pass over the data serves both outputs.

## Inductor Fix Required

**NEW_PATTERN**: A dedicated fused materialized-producer plus indexed partial-reduction
template for relative-position-bias style scatter-reduces. The template must:
1. Compute the row-reduction (fma) and store the full materialized output
2. In the same pass, accumulate per-block partial sums
3. Scatter-accumulate those partials into an indexed output via atomics

This requires the scheduler to recognize that a reduction kernel's materialized output
can serve as an in-register source for a sibling batch-reduction + indexed scatter.

## Generalizability

MEDIUM - This pattern is specific to relative position bias in window attention
(Swin Transformer family). However, the general principle (materialized reduction output
+ sibling scatter-accumulate from same producer) could apply to other attention bias
gradient computations.

## Remaining Gap After CD

~100-157us (140-198us best compile vs ~41us oracle). This is a LARGE gap -- the oracle
achieves 5.4x over default by eliminating the redundant recomputation pass entirely.
