# amax_sum_sum_424d7adcff6b

## Status: NO GAP (BAD_ORACLE)

## Measurement
- Oracle: 381.76 us
- Compiled: 341.86 us
- Ratio: 0.895x (oracle is slower than compiled)

## Pattern
Masked-LM ignore-index cross-entropy mean from BERT MLM training.
Pattern hash `424d7adcff6b`, from `hf_BertForMaskedLM_train_000`.
Shape: `[16384, 30522]` logits (batch=32, seq=512, vocab=30522).

## What the Oracle Does
Custom Triton kernel computing:
- Sliced logits layout (from [16384, 30524] -> [16384, 30522])
- Stable row logsumexp (online softmax)
- Safe masked target gather with ignore_index=-100
- Valid-count reduction
- Loss reduction and final scalar division

## Root Cause of BAD_ORACLE
At shape [16384, 30522], rows have 30522 elements -- a large reduction dimension.
Inductor's split-reduction or multi-kernel strategy handles this well because it can
use multiple thread blocks per row. The oracle attempts to do the entire row in a
single program instance which, at vocab_size=30522, requires multiple loop iterations
and heavy register usage. Inductor's approach of separating the log_softmax reduction
from the gather/mask/sum operations allows better parallelism for this large-vocab case.
Inductor wins by ~10%.

## Classification
NEW_PATTERN (per oracle docstring), but no performance gap exists. Inductor's
multi-pass approach is actually faster for large vocab sizes. No fix needed.

## Kernel Count
Not investigated (no gap to close).
