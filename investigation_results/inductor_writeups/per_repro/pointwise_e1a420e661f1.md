# pointwise_e1a420e661f1

## Current Result

- Family: `layout_indexing_stencil_fusion`
- Classification: `BANDWIDTH_BOUND`
- Oracle artifact: `repros/canonical/pointwise_e1a420e661f1/oracle_permute_clone.py`
- Oracle status: full-scope diagnosis-only, `not_true_floor`
- Correctness: PASS
- Output layout: eager `shape=(64, 768) stride=(768, 1)`, oracle `shape=(64, 768) stride=(768, 1)`
- Oracle: `3.87 us`
- Template `torch.compile` with `coordinate_descent_tuning=True`: `3.94 us`
- Interleaved `bench_compare`: `cd=4.48 us`, `combo=4.42 us`
- Historical 3-config best: `5.760 us`
- True floor: no; leave `canonical_oracle_path` blank for parent integration.

## Diagnosis

The repro takes one contiguous `float32[1, 12, 64, 64]` CUDA input, applies
`permute([0, 2, 1, 3])`, clones that permuted view into contiguous
`float32[1, 64, 12, 64]`, then views the same storage as the final contiguous
`float32[64, 768]` output. The oracle preserves the full scope: it allocates
that same dense output layout and directly materializes
`out[h, c * 64 + w] = arg81_1[0, c, h, w]`.

Gap diagnosis (classification: `BANDWIDTH_BOUND`): the oracle computes the full
permute-clone-view scope as one direct Triton materialization from
`arg81_1[0, c, h, w]` to the final contiguous `float32[64, 768]` output,
whereas Inductor already emits one fused clone/permute materialization kernel,
`triton_poi_fused_clone_permute_0`, for the same isolated graph; Inductor
cannot materially improve this today because the user-visible clone requires
reading and writing all 49,152 float32 elements and this capture has no
surrounding consumer to absorb the materialization; the fixing classification
is `BANDWIDTH_BOUND`, meaning no local scheduler, scatter-reduce, split-K,
algebraic, recompute, or new-pattern change is indicated beyond broader graph
fusion that removes the clone materialization outside this repro boundary.

## Measurements

```bash
python -m py_compile repros/canonical/pointwise_e1a420e661f1/oracle_permute_clone.py
python repros/canonical/pointwise_e1a420e661f1/oracle_permute_clone.py --check
python repros/canonical/pointwise_e1a420e661f1/oracle_permute_clone.py --bench --warmup 10 --rep 50
python scripts/bench_compare.py repros/canonical/pointwise_e1a420e661f1/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo --n-warmup 10 --n-rep 50 --rounds 5 --max-workers 1
python repros/canonical/pointwise_e1a420e661f1/repro.py --count-kernels-only --no-gpu-lock
```

Results:

- `py_compile`: PASS.
- `--check`: PASS; output 0 `shape=[64, 768]`, `dtype=torch.float32`, `max_diff=0.00e+00`.
- `--bench --warmup 10 --rep 50`: `oracle_us=3.87`, `compile_us=3.94`, `ratio=1.017`, `status=AT_FLOOR`.
- `bench_compare`: `cd=4.48 us`, `combo=4.42 us`, fastest `combo`.
- Kernel count: one generated kernel, `triton_poi_fused_clone_permute_0`.

## Parent Integration Values

- `status`: `not_true_floor`
- `classification`: `BANDWIDTH_BOUND`
- `canonical_oracle_path`: leave blank
- `oracle_us`: `3.87`
- `template_cd_compile_us`: `3.94`
- `bench_compare_cd_us`: `4.48`
- `bench_compare_combo_us`: `4.42`
- `true_floor`: `false`
- CSV note: `full-scope diagnosis-only artifact repros/canonical/pointwise_e1a420e661f1/oracle_permute_clone.py plus writeup investigation_results/inductor_writeups/per_repro/pointwise_e1a420e661f1.md; --check PASS; oracle=3.87us; template_cd_compile=3.94us; bench_compare_cd=4.48us; bench_compare_combo=4.42us; classification=BANDWIDTH_BOUND; true_floor=no; main path intentionally blank because Inductor already emits one fused clone/permute materialization kernel`
