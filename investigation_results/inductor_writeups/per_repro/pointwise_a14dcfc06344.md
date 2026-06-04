# pointwise_a14dcfc06344

## Current Result

- Family: `layout_indexing_stencil_fusion`
- Classification: `SCHEDULER_FUSION`
- Oracle path: `repros/canonical/pointwise_a14dcfc06344/oracle_layout_stencil.py`
- Correctness: PASS
- Oracle: `212.74 us`
- `torch.compile coordinate_descent_tuning=True`: `370.56 us`
- `torch.compile combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `369.89 us`
- Historical `best_compile_us`: `353.11999917030334 us`
- True floor: yes

## Diagnosis

The repro computes four full BN-affine + ReLU branches on
`f32[128, 192, 17, 17]`, concatenates them into `f32[128, 768, 17, 17]`, and
then applies `avg_pool2d(kernel=3, stride=1, padding=1)` to the concatenated
tensor. The oracle keeps that exact full scope: it takes the same 20 inputs from
`make_inputs()` and returns the same single contiguous `f32[128, 768, 17, 17]`
output as `Repro()(*make_inputs())`.

This is a `SCHEDULER_FUSION` gap. The oracle does the branch-local BN-affine and
ReLU computation inside the 3x3 avg-pool stencil and stores directly into the
final concatenated channel layout, so it avoids materializing and rereading the
post-ReLU/cat tensor. Inductor cannot do this today because the scheduler does
not fuse layout-producing `cat` operands through a same-channel stencil
consumer; it schedules the branch producers/final layout and the avg-pool
stencil as separate work. The fixing change is a layout-aware stencil fusion
path that recognizes `avg_pool2d(cat(branches), same-channel)` and sinks each
branch producer into the stencil tile while assigning stores to the final cat
channel offsets.

The generated random BN variance inputs produce deterministic NaNs in some
channels. The oracle `--check` path therefore compares with `equal_nan=True`;
finite values still match within `5.72e-06` max absolute error.

## Measurements

```bash
python -m py_compile repros/canonical/pointwise_a14dcfc06344/oracle_layout_stencil.py
python repros/canonical/pointwise_a14dcfc06344/oracle_layout_stencil.py --check
python repros/canonical/pointwise_a14dcfc06344/oracle_layout_stencil.py --bench --warmup 10 --rep 50
```

Results:

- `--check`: PASS; output shape `[128, 768, 17, 17]`, dtype `torch.float32`,
  `finite_max_diff=5.72e-06`, deterministic NaN mask matched.
- `--bench --warmup 10 --rep 50`: `oracle_us=213.57`,
  `compile_us=370.56`, `ratio=1.735`, `status=GOOD`.

Required combo config command:

```bash
python - <<'PY'
import importlib.util
from pathlib import Path
import torch._inductor.config as cfg

cfg.combo_kernels = True
cfg.combo_kernel_per_subkernel_blocks = True
cfg.coordinate_descent_tuning = True
cfg.benchmark_combo_kernel = True
cfg.triton.multi_kernel = 3

path = Path('repros/canonical/pointwise_a14dcfc06344/oracle_layout_stencil.py').resolve()
spec = importlib.util.spec_from_file_location('oracle_a14d_combo', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
inputs = mod.get_inputs()
instance = mod.get_repro_instance()
mod.bench_oracle(mod.oracle_forward, instance, inputs, mod.REPRO_ID, warmup=10, rep=50)
PY
```

Result:

- `oracle_us=212.74`, `compile_us=369.89`, `ratio=1.739`, `status=GOOD`.

## Parent Integration Values

- `classification`: `SCHEDULER_FUSION`
- `canonical_oracle_path`: `repros/canonical/pointwise_a14dcfc06344/oracle_layout_stencil.py`
- `oracle_us`: `212.74`
- `best_required_local_compile_us`: `369.89`
- `historical_best_compile_us`: `353.11999917030334`
- `true_floor`: `true`
- CSV notes: `full-scope --check PASS equal_nan; measured_oracle_us=212.74; cd_compile=370.56; combo_compile=369.89; historical_best=353.11999917030334; classification=SCHEDULER_FUSION; true_floor=yes; owner=Codex-template-layout-a14d`
