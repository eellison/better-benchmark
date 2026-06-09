# var_mean_7bd119719d27

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_7bd119719d27/oracle_bart_embedding_layernorm_aliases.py`
- Oracle: measured
- Ratio: 1.038x
- Status: `at_floor`

## Diagnosis

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete BART token embedding, generated position embedding with offset 2, fp32 population var_mean over hidden size 1024, eps=1e-5 rsqrt normalization, affine scale/bias epilogue, and three sibling [8192, 1024] views from one output buffer in one shape-specialized Triton row kernel, whereas Inductor already measures at the same floor for this full decomposed embedding/iota/var_mean/affine/view graph; Inductor cannot mater

## Config exploration results

- Inductor already matches or beats the oracle (ratio <= 1.05x). No config exploration needed.
- Status: Small gap (1.038x) within noise threshold
