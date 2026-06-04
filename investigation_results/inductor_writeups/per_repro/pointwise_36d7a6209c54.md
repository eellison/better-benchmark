# pointwise_36d7a6209c54

## Current Result

- Family: `layout_indexing_stencil_fusion`
- Classification: `BANDWIDTH_BOUND`
- Oracle artifact: `repros/canonical/pointwise_36d7a6209c54/oracle_layout_stencil.py`
- Oracle status: full-scope diagnosis-only, `not_true_floor`
- True floor: no
- Historical `best_compile_us`: `5.0 us` from the expanded oracle worklist (`4.991999827325344 us` in the interleaved 3-config table)

## Scope

The repro consumes two CUDA `int32[]` scalar indices and one contiguous
CUDA `int64[6144]` lookup table. The full compiled scope is:

```text
out = (arg2 >= arg3) & (arg4[arg2] == arg4[arg3])
```

The oracle keeps that exact scope: same three inputs from `make_inputs()`, one
CUDA `torch.bool[]` output, and no omitted surrounding work. It is not an eager
timed floor; the timed path is one Triton kernel that loads the two scalar
indices, performs the two table loads, compares, and stores the scalar bool.

## Measurements

Commands:

```bash
python -m py_compile repros/canonical/pointwise_36d7a6209c54/oracle_layout_stencil.py
python repros/canonical/pointwise_36d7a6209c54/oracle_layout_stencil.py --check
python repros/canonical/pointwise_36d7a6209c54/oracle_layout_stencil.py --bench --warmup 10 --rep 50
python scripts/bench_compare.py repros/canonical/pointwise_36d7a6209c54/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo --rounds 5 --n-warmup 10 --n-rep 50 --output /tmp/pointwise_36d7a6209c54_bench_compare.json
```

Results:

- `py_compile`: passed
- `--check`: PASS; output 0 exact, dtype `torch.bool`
- Oracle `--bench --warmup 10 --rep 50`: `oracle_us=4.480`, local compile inside the oracle harness `coordinate_descent_tuning=True=4.288`, combo config `4.352`, `true_floor=false`, `status=DIAGNOSIS_ONLY`
- Interleaved `bench_compare.py`: `coordinate_descent_tuning=True=4.064000211656094 us`, combo config `4.191999789327383 us`, `rep_per_round=500`

## Diagnosis

This is a scalar launch-floor repro. Inductor already emits one Triton kernel
for the full expression, and the adjusted byte count is only 25 bytes. The
hand-written kernel removes any generic pointwise scaffolding, but it is still
one launch and does not beat the required local compile configs when measured
with the repo's interleaved graph replay path.

The appropriate queue classification is `BANDWIDTH_BOUND` / already at launch
floor. There is no scheduler-fusion, scatter-reduce, cooperative split-K,
algebraic-elimination, or recompute-fusion opportunity to claim from this
artifact.

## CSV Notes

`full-scope Triton scalar oracle PASS; measured_oracle_us=4.480; cd_compile_us=4.064000211656094; combo_compile_us=4.191999789327383; historical_best=5.0; classification=BANDWIDTH_BOUND; true_floor=no; not_true_floor because required local compile is faster and repro is a 25-byte one-launch scalar index/compare`
