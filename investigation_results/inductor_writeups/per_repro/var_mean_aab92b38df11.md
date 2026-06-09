# var_mean_aab92b38df11

## Status: BAD_ORACLE (compile already faster)

- Oracle: 56.35 us
- Compile: 32.93 us
- Ratio: 0.584x (oracle is 1.71x slower than compile)

## Classification: NO_GAP

Oracle path: `repros/canonical/var_mean_aab92b38df11/oracle_patch_pos_layernorm.py`

The compiled Inductor output already significantly outperforms this oracle. No Inductor improvement needed. The oracle's row-blocked kernel appears to suffer from suboptimal occupancy or register pressure at this shape, while Inductor's persistent reduction path is highly efficient for hidden_size=768 LayerNorm.

## Details
- Pattern: patch position layernorm (ViT)
- Shape: [32768, 768] output from [128,768,16,16] conv + [1,256,768] pos_embed
- Correctness: PASS
