# pointwise_3320bac17d54

## Current Result

- Family: `layout_indexing_stencil_fusion`
- Classification: `BANDWIDTH_BOUND`
- Oracle path: `repros/canonical/pointwise_3320bac17d54/oracle_layout_stencil.py`
- Correctness: PASS
- Oracle: `3.20 us`
- `torch.compile coordinate_descent_tuning=True`: `3.01 us` in the oracle harness; `3.49 us` in interleaved `bench_compare`
- `torch.compile combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `3.10 us` in the oracle harness; `3.46 us` in interleaved `bench_compare`
- Historical `best_compile_us`: about `4.9 us`
- True floor: no (`not_true_floor`)

## Diagnosis

The repro takes one `i64[1]` CUDA input, selects element 0, constructs scalar
`32`, writes it through `select_scatter`, then `copy_` mutates the original
input and returns that same one-element tensor. The oracle keeps that exact full
scope: it takes the same input tuple, launches one Triton scalar-store kernel
that writes `arg2_1[0] = 32`, and returns the mutated input tensor with the same
shape and dtype as `Repro()(*make_inputs())`.

This is a launch-floor scalar update, not a real layout/stencil fusion gap.
Inductor already emits one Triton kernel for the compiled repro
(`triton_poi_fused_copy_full_select_0`). The hand Triton oracle is at the same
scale and does not beat the local coordinate-descent compile path, so there is
no evidence for a scheduler fusion opportunity here. The expanded-worklist SOL
is `0.0 us` because the effective byte count rounds to zero for this one-element
mutation.

## Measurements

```bash
python -m py_compile repros/canonical/pointwise_3320bac17d54/oracle_layout_stencil.py
python repros/canonical/pointwise_3320bac17d54/oracle_layout_stencil.py --check
python repros/canonical/pointwise_3320bac17d54/oracle_layout_stencil.py --bench --warmup 10 --rep 50
```

Results:

- `--check`: PASS; output 0 exact, dtype `torch.int64`.
- `--bench --warmup 10 --rep 50`: `oracle_us=3.20`,
  `compile_us=3.01`, `ratio=0.940`, `status=BAD_ORACLE`.

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

path = Path('repros/canonical/pointwise_3320bac17d54/oracle_layout_stencil.py').resolve()
spec = importlib.util.spec_from_file_location('oracle_3320_combo', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
inputs = mod.get_inputs()
instance = mod.get_repro_instance()
mod.bench_oracle(mod.oracle_forward, instance, inputs, mod.REPRO_ID, warmup=10, rep=50)
PY
```

Result:

- `oracle_us=3.07`, `compile_us=3.10`, `ratio=1.010`,
  `status=AT_FLOOR`.

Interleaved local compile comparison:

```bash
python scripts/bench_compare.py repros/canonical/pointwise_3320bac17d54/repro.py --config "coordinate_descent_tuning=True" --label cd --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo --n-warmup 10 --n-rep 50 --rounds 5
```

Result:

- `cd=3.49 us`, `combo=3.46 us`; combo was `1.0093x` faster in the
  one-repro interleaved comparison.

## Parent Integration Values

- `classification`: `BANDWIDTH_BOUND`
- `canonical_oracle_path`: `repros/canonical/pointwise_3320bac17d54/oracle_layout_stencil.py`
- `oracle_us`: `3.20`
- `best_required_local_compile_us`: `3.01`
- `historical_best_compile_us`: `4.9`
- `true_floor`: `false`
- CSV notes: `full-scope --check PASS; measured_oracle_us=3.20; cd_compile=3.01; combo_compile=3.10; bench_compare_cd=3.49; bench_compare_combo=3.46; historical_best=4.9; classification=BANDWIDTH_BOUND; true_floor=no; not_true_floor; owner=Codex-template-layout-3320`
