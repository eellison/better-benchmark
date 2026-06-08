# sum_7de69e2bde32 - oracle_head_blocked_mask

## Status: BAD_ORACLE (compile beats oracle)

- Oracle: 58.05 us
- Compile: 53.41 us
- Ratio: 0.920x

## Classification: NO_GAP

Oracle path: `repros/canonical/sum_7de69e2bde32/oracle_head_blocked_mask.py`

The compiled Inductor output outperforms the oracle. No Inductor improvement needed.

## Details
- Model: torchbench BERT_pytorch training
- Pattern: BERT dropout-weighted softmax-backward with broadcast fill mask loaded once per batch-row across heads
- Shape: [1536, 128, 128] output (4*12*32 batch-head combinations, 128x128 attention)
- The oracle attempts to tile broadcast head dimensions for the [batch,1,row,col] mask reuse, but Inductor's flattened row approach is more efficient at this shape
- Correctness: PASS
