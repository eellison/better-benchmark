# var_mean_c564fa4c6115

## Classification: NO_GAP (BAD_ORACLE)

## Current Result

- Family: `embedding_layernorm`
- Oracle path: `repros/canonical/var_mean_c564fa4c6115/oracle_embedding_layernorm.py`
- Correctness: PASS (shape=[8, 1024, 768] dtype=torch.float32 max_diff=2.86e-06)
- Oracle: `17.22 us`
- `torch.compile coordinate_descent_tuning=True`: `16.00 us`
- Ratio: 0.929
- Status: `bad_oracle` (compile is 7% faster)

## Diagnosis

The oracle is slower than `torch.compile`. Inductor already generates optimal code for this embedding + LayerNorm pattern. No gap exists.

## Kernel count
- Oracle: 1 kernel
- Inductor: equivalent or better

## Recommendation

No action needed. The oracle does not demonstrate a gap.
