# var_mean_a8fd2a07cabc

## Status: BAD_ORACLE (compile already faster)

- Oracle: 199.55 us
- Compile: 53.31 us
- Ratio: 0.267x (oracle is 3.74x slower than compile)

## Classification: NO_GAP

Oracle path: `repros/canonical/var_mean_a8fd2a07cabc/oracle_fnet_complex_layernorm.py`

The compiled Inductor output massively outperforms this oracle. The oracle is nearly 4x slower than Inductor's compiled output. No Inductor improvement needed.

## Previous diagnosis (superseded by measurement)

The oracle was designed to compute the FNet residual LayerNorm-to-complex64 scope in one kernel. However, at this shape the fused kernel suffers severely from the complex64 conversion overhead and/or register pressure, making Inductor's decomposed approach far superior.

## Details
- Model: hf_GoogleFnet_infer_000
- Pattern: view -> add(residual) -> var_mean -> LN affine -> complex64 conversion
- Shape: [16384, 768] + [32, 512, 768] f32 -> complex64 output
- Correctness: PASS (exact match)
