# pointwise_724b26e60185

## Current Result

- Family: `layout_indexing_stencil_fusion`
- Classification: `BANDWIDTH_BOUND`
- Oracle artifact: `repros/canonical/pointwise_724b26e60185/oracle_layout.py`
- Oracle status: full-scope diagnosis-only, `not_true_floor`
- Correctness: PASS
- Oracle: `4.00 us`
- Template `torch.compile` with `coordinate_descent_tuning=True`: `3.87 us`
- Template `torch.compile` with required combo config: `3.55 us`
- Interleaved `bench_compare`: `cd=4.06 us`, `combo=4.45 us`
- Historical 3-config best: `5.504 us`
- True floor: no; leave `canonical_oracle_path` blank for parent integration.

## Diagnosis

The repro takes one contiguous `float32[30522]` CUDA input, creates a
`float32[2]` zero tensor, and returns `aten.cat([arg204_1, zeros])` as a fresh
contiguous `float32[30524]` output. The oracle preserves the full scope: it
allocates that same output layout, copies the input prefix, writes the two
trailing zeros, and returns one tensor with the same shape, dtype, and stride as
`Repro()(*make_inputs())`.

This is not a layout/stencil fusion gap. Inductor already lowers the captured
graph to one fused pointwise kernel, `triton_poi_fused_cat_full_0`, covering
both `aten.full` and `aten.cat` with one input load and one output store. The
hand Triton materialization does not beat the required tuned compile paths, and
the remaining work is the unavoidable launch/allocation plus about 30k float
loads/stores for a fresh cat output.

Gap diagnosis (classification: `BANDWIDTH_BOUND`): the oracle writes the fresh
concat result directly in one simple Triton materialization kernel, whereas
Inductor emits one generic fused cat/full pointwise kernel for the same scope;
Inductor cannot remove the work today because `aten.cat` must return a new dense
tensor and this isolated capture has no surrounding producer or consumer to fuse
with; the fixing classification is `BANDWIDTH_BOUND`, meaning no scheduler,
scatter-reduce, split-K, algebraic, or recompute-fusion change is indicated
beyond broader launch/materialization overhead reduction or upstream graph
fusion outside this repro.

## Measurements

```bash
python -m py_compile repros/canonical/pointwise_724b26e60185/oracle_layout.py
python repros/canonical/pointwise_724b26e60185/oracle_layout.py --check
python repros/canonical/pointwise_724b26e60185/oracle_layout.py --bench --warmup 10 --rep 50
```

Results:

- `--check`: PASS; output 0 `shape=[30524]`, `dtype=torch.float32`, `max_diff=0.00e+00`, `stride=(1,)`.
- `--bench --warmup 10 --rep 50`: `oracle_us=4.00`, `compile_us=3.87`, `ratio=0.968`, `status=AT_FLOOR`.

Required combo-config oracle harness comparison:

```bash
cfg.combo_kernels = True
cfg.combo_kernel_per_subkernel_blocks = True
cfg.coordinate_descent_tuning = True
cfg.benchmark_combo_kernel = True
cfg.triton.multi_kernel = 3
```

Result:

- `oracle_us=4.00`, `compile_us=3.55`, `ratio=0.888`, `status=BAD_ORACLE`.

Interleaved local compile comparison:

```bash
python scripts/bench_compare.py repros/canonical/pointwise_724b26e60185/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo --n-warmup 10 --n-rep 50 --rounds 5 --max-workers 1
```

Result:

- `cd=4.06 us`, `combo=4.45 us`; coordinate descent was fastest.

## Parent Integration Values

- `status`: `not_true_floor`
- `classification`: `BANDWIDTH_BOUND`
- `canonical_oracle_path`: leave blank
- `oracle_us`: `4.00`
- `best_required_local_compile_us`: `3.55`
- `bench_compare_cd_us`: `4.06`
- `bench_compare_combo_us`: `4.45`
- `true_floor`: `false`
- CSV note: `full-scope diagnosis-only artifact repros/canonical/pointwise_724b26e60185/oracle_layout.py plus writeup investigation_results/inductor_writeups/per_repro/pointwise_724b26e60185.md; --check PASS; oracle=4.00us; template_cd_compile=3.87us; combo_compile=3.55us; bench_compare_cd=4.06us; bench_compare_combo=4.45us; classification=BANDWIDTH_BOUND; true_floor=no; main path intentionally blank because Inductor already emits one fused cat/full materialization kernel`
