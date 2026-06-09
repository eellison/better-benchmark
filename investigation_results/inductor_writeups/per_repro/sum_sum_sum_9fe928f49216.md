# sum_sum_sum_9fe928f49216

## Summary

- Model: MobileViT gradient layout
- Oracle: `oracle_mobilevit_grad_layout.py`
- Classification: BAD_ORACLE
- Ratio: 0.936x (oracle 221.86us, compile 207.74us)
- Status: Oracle is slower than Inductor compile

## Root Cause

The oracle is slower than Inductor's compiled output by ~6%. Inductor's generic lowering with coordinate descent tuning already handles the MobileViT gradient layout pattern (`[128, 144, 32, 32]` output) effectively, outperforming the hand-written oracle.

## Config Exploration

Not needed -- oracle is the slower baseline.

## Fix Assessment

**No fix needed / BAD_ORACLE** -- Inductor already achieves better performance than the oracle for this pattern.
