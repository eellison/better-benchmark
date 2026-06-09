# pointwise_a4e42370cdb9 — Cat BN ReLU AvgPool (Inception V3)

## Summary
- **Model**: timm_adv_inception_v3_infer_000
- **Classification**: BAD_ORACLE
- **Ratio**: 0.838x (oracle 106.05us vs compile 88.86us)
- **Status**: Oracle is slower than compile; no gap to investigate

## Analysis

The oracle is slower than Inductor's compiled output. This indicates the oracle's hand-written Triton kernel is not optimal for this workload, or Inductor's fusion/tiling strategy for this large multi-branch BN+cat+avgpool pattern is already effective.

The repro involves 6 BN-inference branches (320, 384, 384, 384, 384, 192 channels) concatenated into [128, 2048, 8, 8], followed by avg_pool2d. The large number of branches and the avgpool stencil make this a complex pattern where Inductor's generic approach may already be near-optimal.

## No Action Required

Oracle is slower than compile. No Inductor improvement needed for this repro.
