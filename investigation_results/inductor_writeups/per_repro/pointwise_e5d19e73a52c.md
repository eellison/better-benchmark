# pointwise_e5d19e73a52c

## Current Result

- Family: `layout_indexing_stencil_fusion`
- Classification: `BANDWIDTH_BOUND`
- Oracle path: `repros/canonical/pointwise_e5d19e73a52c/oracle_layout_stencil.py`
- Correctness: PASS
- Oracle: `3.23 us`
- `torch.compile coordinate_descent_tuning=True`: `3.36 us`
- `torch.compile combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `3.23 us`
- Historical `best_compile_us`: `4.896000027656555 us`
- True floor: no; this is a launch-floor scalar conversion and the required combo compile run matches/beats the oracle.

## Diagnosis

The repro takes one CUDA scalar `int64[]` input and returns the full
`lift_fresh_copy -> convert_element_type(float64)` result as a CUDA scalar
`float64[]`. The oracle keeps that exact scope: same input tuple, output count,
shape, dtype, and stride as `Repro()(*make_inputs())`.

Inductor already lowers this repro to one pointwise Triton launch
(`triton_poi_fused_convert_element_type_0`). The oracle uses one hand-written
Triton scalar pointwise kernel that loads the `int64` input and stores the
`float64` output directly, avoiding any eager-only timed floor. There is no
layout/stencil fusion work to recover: the measured delta is launch-floor noise
on a one-element transfer/convert, and the required combo-kernel compile config
is faster than the oracle in the local run.

This is `BANDWIDTH_BOUND` / launch-bound for queue integration. Do not treat
the oracle as a true floor or use it as evidence for a scheduler fusion change.

## Measurements

```bash
python -m py_compile repros/canonical/pointwise_e5d19e73a52c/oracle_layout_stencil.py
python repros/canonical/pointwise_e5d19e73a52c/oracle_layout_stencil.py --check
python repros/canonical/pointwise_e5d19e73a52c/oracle_layout_stencil.py --bench --warmup 10 --rep 50
```

Results:

- `--check`: PASS; output shape `[]`, dtype `torch.float64`, `max_diff=0.00e+00`.
- `--bench --warmup 10 --rep 50`: `oracle_us=3.23`, `compile_us=3.36`,
  `ratio=1.04`, `status=AT_FLOOR`.

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

path = Path('repros/canonical/pointwise_e5d19e73a52c/oracle_layout_stencil.py').resolve()
spec = importlib.util.spec_from_file_location('oracle_e5d1_combo', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
inputs = mod.get_inputs()
instance = mod.get_repro_instance()
mod.bench_oracle(mod.oracle_forward, instance, inputs, mod.REPRO_ID, warmup=10, rep=50)
PY
```

Result:

- `oracle_us=3.46`, `compile_us=3.23`, `ratio=0.935`, `status=BAD_ORACLE`.

## Parent Integration Values

- `classification`: `BANDWIDTH_BOUND`
- `canonical_oracle_path`: diagnosis-only; do not set as true-floor oracle
- `oracle_us`: `3.23`
- `best_required_local_compile_us`: `3.23`
- `historical_best_compile_us`: `4.896000027656555`
- `true_floor`: `false`
- CSV notes: `full-scope --check PASS; measured_oracle_us=3.23; cd_compile=3.36; combo_compile=3.23; historical_best=4.896000027656555; classification=BANDWIDTH_BOUND; true_floor=no; owner=Codex-template-layout-e5d1`
