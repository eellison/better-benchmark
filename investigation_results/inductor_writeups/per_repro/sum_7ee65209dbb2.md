# sum_7ee65209dbb2 - oracle_bert_attention_backward

## Status: AT_FLOOR (compile matches oracle)

- Oracle: 31.39 us
- Compile: 31.36 us
- Ratio: 0.999x

## Classification: NO_GAP

Oracle path: `repros/canonical/sum_7ee65209dbb2/oracle_bert_attention_backward.py`

The compiled Inductor output matches the oracle performance exactly. No Inductor improvement needed.

## Details
- Models: torchbench hf_Bert_train, hf_Roberta_base_train
- Pattern: BERT/RoBERTa attention backward row-reduction (dropout-mask scaling, broadcast attention-mask, exp probability reconstruction, row product reduction, fma epilogue)
- Shape: [48, 512, 512] f32 (4*12 heads, 512x512 attention)
- Classification in oracle: BANDWIDTH_BOUND -- Inductor already emits one fused reduction kernel with the same mandatory reads/writes
- Correctness: PASS
