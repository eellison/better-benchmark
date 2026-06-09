# sum_sum_sum_b3091402bb28

## Classification: BAD_ORACLE

## Oracle: oracle_bert_embedding_scatter_reduce.py

## Measurements

- Compiled: 138.2 us
- Oracle: 190.3 us
- Ratio: 0.726x (oracle 38% slower than compile)
- Oracle correctness: PASS

## Diagnosis

The oracle (BERT embedding scatter-reduce) is substantially slower than torch.compile output on this hardware. Despite the oracle's fused scatter-reduce approach, Inductor's separate kernels with optimized autotuned configs outperform the monolithic oracle kernel for these particular shapes (vocab [20005, 768]).

## Status: closed_no_gap (BAD_ORACLE)
