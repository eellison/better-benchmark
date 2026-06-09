# mean_72eb7eeed84f

## Compile: 9.82us, Oracle: 10.14us, Gap: 0.968x (BAD_ORACLE)

## Diagnosis: BAD_ORACLE

## Root cause: The oracle (residual RMSNorm with aliases) is slightly slower than torch.compile output. Compile outperforms the oracle by ~3%.

## Status: closed_no_gap

## Details

- Model: residual RMSNorm with aliased outputs
- Pattern: mean reduction (RMSNorm) with residual connection, producing 3 aliased [4096, 512] fp32 outputs
- Shape: [4096, 512] with reduction over last dim (hidden size 512)
- The oracle is slower than compile -- no investigation needed.
- All three outputs are verified correct (max_diff=1.91e-06).
- Inductor already handles this norm pattern efficiently.
