# var_mean_349ca1a29d3f

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_349ca1a29d3f/oracle_layernorm_aliases.py`
- Oracle: measured
- Ratio: 1.0x
- Status: `at_floor`

## Diagnosis

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Reformer LayerNorm alias scope in one generated-code-matching Triton row kernel, including the fresh clone-equivalent backing allocation, fp32 correction=0 mean and centered variance over hidden size 256, eps before libdevice.rsqrt, affine multiply then add, and the three returned `[32768,256]` alias views of the same final buffer, whereas Inductor already emits the same single persistent row-reduction shape for t

## Config exploration results

- Inductor already matches or beats the oracle (ratio <= 1.05x). No config exploration needed.
- Status: Inductor matches oracle within noise
