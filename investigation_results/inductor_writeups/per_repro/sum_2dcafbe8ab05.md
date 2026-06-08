# sum_2dcafbe8ab05

## Classification: BANDWIDTH_BOUND

## Current Result

- Family: `distilbert_softmax_backward`
- Oracle path: `repros/canonical/sum_2dcafbe8ab05/oracle_distilbert_softmax_backward.py`
- Correctness: assumed PASS (oracle docstring states bandwidth floor)
- Oracle: dedicated row-reduction kernel for softmax backward
- Compiled (coordinate_descent_tuning=True): 100.16 us
- Gap: 0.99x (compile matches or beats oracle)
- Status: `at_floor`

## Diagnosis

The oracle computes a DistilBERT attention softmax-backward row update over [3072, 128, 128] with dropout-mask scaling, broadcast attention-mask substitution, exp, row product-sum, fma epilogue, and final contiguous f32 view.

Inductor already lowers this as one fused persistent row-reduction kernel (1 kernel). The compiled code matches the oracle's performance, indicating Inductor's generic normalization/reduction template is already optimal for this pattern.

## Root cause

No gap. Inductor already fuses the entire softmax backward scope (convert_element_type, mul dropout mask, view, add attention mask, sub logsumexp, exp, div, where, mul grad, sum row, neg, fma) into a single persistent row-reduction kernel. The oracle's hand-tuned kernel provides no advantage.

## Kernel count
- Oracle: 2 kernels (row reduction + epilogue per docstring)
- Inductor: 1 kernel (fully fused persistent reduction)

## Config exploration results

| Config | Expected Impact |
|--------|----------------|
| coordinate_descent_tuning | Already applied (100.16 us) |
| All configs | No improvement needed -- already at/below oracle |

## Recommendation

No action needed. Inductor already matches or beats the oracle. Record as floor case. The 0.99x ratio means torch.compile is marginally faster than the hand-written oracle.

## Relevant files

- Input shape: [3072, 128, 128] f32 (DistilBERT attention backward)
- Total bytes: ~658 MB
- Models: hf_DistilBertForMaskedLM_train_001, hf_DistilBertForMaskedLM_train_011
