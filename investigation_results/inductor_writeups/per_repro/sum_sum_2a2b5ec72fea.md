# sum_sum_2a2b5ec72fea

## Status

- Family: `structured_pool_upsample_backward_reduce`
- Claim owner: `Codex-template-structured-2a2b`
- Classification: `SCATTER_REDUCE`
- Oracle path: `repros/canonical/sum_sum_2a2b5ec72fea/oracle_structured_scatter_reduce.py`
- True floor: yes

## Scope

The oracle computes the exact `Repro()(*make_inputs())` tuple, not a reduction
subset:

- output 0: `[128, 96, 112, 112]` float32 BN-gradient epilogue
- output 1: `[96]` float32 channel reduction scaled by `arg182_1`

The graph crops `getitem_222[:, :, :112, :112]`, builds a sigmoid/BN producer
from `arg180_1`, `arg181_1`, `arg182_1`, `arg11_1`, and `arg12_1`, computes the
two channel reductions over `[0, 2, 3]`, then uses those reductions in the
full-size output epilogue. The Triton oracle uses a partial channel-reduction
kernel, a small channel finalizer, and a channel-tiled epilogue kernel that
recomputes the structured crop/gather producer instead of materializing the
intermediate tensors.

## Gap Diagnosis

The oracle differs from Inductor by treating the negative-pad crop as a
structured gather feeding both channel reductions and the dependent full-tensor
BN epilogue, so the crop/sigmoid producer is never materialized as a generic
intermediate. Inductor cannot do this today because its scheduler sees the
`constant_pad_nd` crop, sigmoid/affine producer, sibling reductions, and
post-reduction BN backward expression as separate generic scheduled work rather
than a structured scatter/gather-reduce producer with a dependent epilogue. The
fix is `SCATTER_REDUCE`: add a structured crop/pool-backward scatter-reduce
lowering that keeps this producer in the reduction template and emits the
full-tensor epilogue from the reduced channel statistics.

## Measurements

- `python -m py_compile repros/canonical/sum_sum_2a2b5ec72fea/oracle_structured_scatter_reduce.py`: PASS
- `python repros/canonical/sum_sum_2a2b5ec72fea/oracle_structured_scatter_reduce.py --check`: PASS, output 0 max diff `8.58e-06`, output 1 max diff `2.84e-03`
- `python repros/canonical/sum_sum_2a2b5ec72fea/oracle_structured_scatter_reduce.py --bench --warmup 10 --rep 50`: `oracle_us=834.94`, `compile_us=1595.55`, `ratio=1.911`, `status=GOOD`
- Local `coordinate_descent_tuning=True`: `1554.53 us`
- Local `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `1554.85 us`
- Historical `best_compile_us`: `872.4160194396973 us`

The oracle is faster than both required local compile configs and the historical
best compile time, so it is a true full-scope floor for CSV/tracker updates.

## Commands

```bash
python -m py_compile repros/canonical/sum_sum_2a2b5ec72fea/oracle_structured_scatter_reduce.py
python repros/canonical/sum_sum_2a2b5ec72fea/oracle_structured_scatter_reduce.py --check
python repros/canonical/sum_sum_2a2b5ec72fea/oracle_structured_scatter_reduce.py --bench --warmup 10 --rep 50
python scripts/bench_compare.py repros/canonical/sum_sum_2a2b5ec72fea/repro.py --config "coordinate_descent_tuning=True" --label cdt --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo_cdt --n-warmup 10 --n-rep 50 --rounds 5
python scripts/validate_corpus_invariants.py
```
