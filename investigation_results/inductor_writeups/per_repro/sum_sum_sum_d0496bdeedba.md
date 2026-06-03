# Gap Diagnosis: sum_sum_sum_d0496bdeedba (fused_embedding_scatter)

## Classification: SCHEDULER_FUSION (new pattern -- multi-output reduction/scatter)

## Measurements

| Config | Time (us) | Ratio vs Default |
|--------|-----------|------------------|
| Default | 159.5 | 1.00x |
| Combo Looped (CD) | 143.4 | 1.11x |
| Combo Persistent (CD) | 143.1 | 1.11x |
| Coord Descent (best) | 136.2 | 1.17x |
| Oracle (handwritten Triton) | ~37 | ~4.3x |

- **Bytes accessed**: 203.5 MB (logical)
- **Best compile SOL**: 3.89 GB/s effective BW ratio
- **Oracle SOL**: ~5.5 TB/s effective BW
- **Models**: hf_GPTNeoForSequenceClassification_train (2 shapes)

## Pattern Description

GPT-Neo layer-norm backward plus embedding gradient scatter-adds. The graph:
1. Computes layer-norm backward: mm0 + mm1 + mm2 = summed input, apply LN backward
   formula with row sums and dot products
2. Produces two [2048]-sized column reductions (sum_ax, sum_a -- weight gradients)
3. Scatter-adds the row gradient into a word embedding table [50257, 2048] via index
4. Scatter-adds the same gradient into a position embedding table [2048, 2048] via index

## Root Cause

Inductor can fuse the word-embedding scatter with the row-reduction producer, but it
SPLITS the position path because:
- That consumer is a second reduction/scatter over the batch dimension
- It has a different output layout
- The scheduler has no representation for one dependent multi-output reduction with
  TWO side-effect scatter stores

Result: Inductor materializes the intermediate `sub_2` gradient tensor, then rereads it
for the position-embedding scatter in a separate pass. This doubles the memory traffic
for the position path.

## Oracle Strategy

Single Triton kernel with grid = [ROW_BLOCKS]:
1. For each row block (16 rows):
   - Compute LN backward gradient in registers
   - Atomic-add gradient into word_embedding_table[word_idx[row], :]
   - Atomic-add gradient into position_embedding_table[pos_idx[token], :]
   - Accumulate partial column sums (sum_ax, sum_a) for weight gradients
2. Second small finalization kernel sums the partial column accumulators

All 4 outputs (sum_ax, sum_a, position_embed_grad, word_embed_grad) produced in a single
pass over the input data.

## Inductor Fix Required

**SCHEDULER_FUSION**: Teach the scheduler/codegen to support a fused multi-output
reduction/scatter template where:
1. A row-reduction producer (LN backward) feeds sibling column accumulators
2. The same producer feeds MULTIPLE final scatter-add side outputs (word + position)
3. The scatter outputs go to different tables with different indexing

The current scheduler cannot model a single reduction producing both:
- Cross-row column accumulators (for weight gradients)
- Multiple indexed scatter-add stores (for embedding gradients)

## Generalizability

MEDIUM - This specific pattern (LN backward + dual embedding scatter) appears in GPT-Neo
and similar models with both word and position embeddings. The general principle (one
reduction producer feeding multiple scatter-add consumers) is broader and could apply to
any model with multiple embedding tables receiving gradients from the same source.

## Remaining Gap After CD

~100-106us (136-143us best compile vs ~37us oracle). This is a VERY LARGE gap (3.7-4.3x).
The oracle eliminates an entire reread pass by doing both scatters in-line with the LN
backward computation.
