# sum_sum_sum_dc96c4651516

## Scope

Full-scope canonical oracle for `Repro()(*make_inputs())`.

- Inputs: original 13-argument repro input tuple.
- Outputs: 4 outputs, matching eager shapes/dtypes/strides:
  - `[128]`, `torch.float32`, stride `(1,)`
  - `[128]`, `torch.float32`, stride `(1,)`
  - `[128, 401408]`, `torch.float32`, stride `(1, 128)`
  - `[128]`, `torch.float32`, stride `(1,)`
- No reduction-only subset is timed. The oracle writes the materialized side output backing storage and returns it as the same transposed view shape/stride as eager.

## Classification

`COOPERATIVE_SPLIT_K`

The repro reconstructs Swin windows into `[128, 56, 56, 128]`, applies two indexed spatial gathers, computes row-local channel reductions, uses those scalars in a dependent epilogue, returns the materialized transposed `[128, 401408]` side output, and returns three channel reductions. The oracle streams that indexed producer once per row tile: it computes the two per-row channel reductions, writes the side-output backing buffer, and emits three channel-wise partial reductions for cooperative finalization.

The Inductor-side fix is cooperative split-K multi-output reduction support for producers that must both materialize a large side output and return small sibling reductions, instead of scheduling the side output and reductions as separate producer/consumer regions.

## Oracle

Path:

`repros/canonical/sum_sum_sum_dc96c4651516/oracle_multi_output_reduction.py`

Implementation:

- Maps each logical `[N, H, W, C]` output row back to the original `[8192, 49, 128]` window row after the two `arg196_1` indexed gathers.
- Uses one Triton main kernel over 128 logical rows x 128 channels to compute row-local reductions, write `out_base[401408, 128]`, and write partials for outputs 0, 1, and 3.
- Uses two small Triton finalize kernels to reduce cooperative partials into the three `[128]` vector outputs.
- Returns `out_base.permute(1, 0)` so the materialized side output matches eager shape and stride.

## Validation

`python -m py_compile repros/canonical/sum_sum_sum_dc96c4651516/oracle_multi_output_reduction.py`: PASS

`python repros/canonical/sum_sum_sum_dc96c4651516/oracle_multi_output_reduction.py --check`: PASS

Latest check output:

```text
Checking sum_sum_sum_dc96c4651516...
  output 0: PASS (shape=[128] dtype=torch.float32 max_diff=6.41e-04)
  output 1: PASS (shape=[128] dtype=torch.float32 max_diff=1.95e-03)
  output 2: PASS (shape=[128, 401408] dtype=torch.float32 max_diff=4.88e-04)
  output 3: PASS (shape=[128] dtype=torch.float32 max_diff=1.56e-01)
Correctness: PASS
```

Stride verification: PASS for all 4 outputs.

## Benchmarks

Required oracle benchmark command:

`python repros/canonical/sum_sum_sum_dc96c4651516/oracle_multi_output_reduction.py --bench --warmup 10 --rep 50`

Result:

```json
{"repro_id": "sum_sum_sum_dc96c4651516", "oracle_us": 284.77, "compile_us": 627.2, "ratio": 2.202, "status": "GOOD"}
```

Required local compile config comparison command:

`python scripts/bench_compare.py repros/canonical/sum_sum_sum_dc96c4651516/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo --rounds 5 --n-warmup 10 --n-rep 50 --output /tmp/sum_sum_sum_dc96c4651516_bench_compare.json`

Result:

```text
cd=613.5680079460144 us
combo=614.2079830169678 us
```

Historical best compile:

- `best_compile_us=524.3520140647888`
- oracle speedup vs historical best: `1.841x`

True floor status: yes.

## CSV Notes

Parent should integrate:

- `oracle_path`: `repros/canonical/sum_sum_sum_dc96c4651516/oracle_multi_output_reduction.py`
- `status`: `implemented_unmeasured` or measured-equivalent queue status
- `classification`: `COOPERATIVE_SPLIT_K`
- `true_floor`: `yes`
- notes: `full-scope --check PASS; measured_oracle_us=284.77; cd_compile_us=613.568; combo_compile_us=614.208; historical_best=524.3520140647888; classification=COOPERATIVE_SPLIT_K; true_floor=yes; owner=Codex-template-multi-dc96`
