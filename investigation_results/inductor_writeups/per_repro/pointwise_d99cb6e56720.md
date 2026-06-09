# pointwise_d99cb6e56720

## Compile: 56.99us, Oracle: 59.07us, Gap: 0.965x (BAD_ORACLE)

## Diagnosis: BAD_ORACLE

## Root cause: The oracle (NFNet gated GELU + avgpool) is slower than torch.compile output. Compile outperforms the oracle by ~3.5%.

## Status: closed_no_gap

## Details

- Model: NFNet gated GELU + average pooling
- Pattern: pointwise gated GELU activation followed by spatial average pooling [128, 1536, 6, 6] fp32
- Shape: [128, 1536, 6, 6] fp32 (post-activation feature map)
- The oracle is slower than compile -- no investigation needed.
- Output verified correct (max_diff=4.77e-07).
- Inductor's decomposed approach with autotuned kernels handles this efficiently.
