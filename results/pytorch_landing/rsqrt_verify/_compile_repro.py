"""
Compile a repro at a single shape via torch.compile(instance) (static, the
bench_parallel default path). Output_code is emitted to stderr by the harness
when run with TORCH_LOGS=output_code. We additionally print kernel-count and a
correctness reference (eager-vs-compiled max abs diff) to stdout.

Usage:
  TORCH_LOGS=output_code PYTHONPATH=<worktree> \
    python3 _compile_repro.py <repro.py> <shape_name> <tag>
"""
import sys, os, importlib.util, math
from pathlib import Path

ROOT = Path("/tmp/scratch_space/better_benchmark")
sys.path.insert(0, str(ROOT))

repro_path = sys.argv[1]
shape_name = sys.argv[2]
tag = sys.argv[3]

import torch

pp = os.environ.get("PYTHONPATH", "/__none__").split(":")[0]
assert torch.__file__.startswith(pp), f"torch not shadowed: {torch.__file__} (PP={pp})"
sys.stderr.write(f"### TAG={tag} torch={torch.__file__}\n")

spec = importlib.util.spec_from_file_location("repro", repro_path)
mod = importlib.util.module_from_spec(spec)
mod.device = torch.device
mod.inf = math.inf
mod.nan = math.nan
spec.loader.exec_module(mod)

from repro_harness import load_shape_configs, make_inputs_from_config
configs = load_shape_configs(repro_path)
if shape_name == "default":
    cfg = next(iter(configs.values()))
elif shape_name in configs:
    cfg = configs[shape_name]
else:
    matches = [k for k in configs if shape_name in k or k in shape_name]
    assert matches, f"shape {shape_name} not in {list(configs)}"
    cfg = configs[matches[0]]
inputs = make_inputs_from_config(cfg)

instance = mod.Repro()
with torch.no_grad():
    eager_out = instance(*inputs)

torch._dynamo.reset()
compiled = torch.compile(instance)
with torch.no_grad():
    comp_out = compiled(*inputs)
    torch.cuda.synchronize()

# numerics: eager vs compiled
def to_f32(t): return t.float()
if isinstance(eager_out, (tuple, list)):
    eo, co = eager_out[0], comp_out[0]
else:
    eo, co = eager_out, comp_out
diff = (to_f32(eo) - to_f32(co)).abs()
denom = to_f32(eo).abs().clamp_min(1e-6)
sys.stderr.write(
    f"### TAG={tag} NUMERICS eager-vs-compiled: max_abs={diff.max().item():.3e} "
    f"mean_abs={diff.mean().item():.3e} max_rel={(diff/denom).max().item():.3e} "
    f"out_dtype={eo.dtype}\n"
)
sys.stderr.write(f"### TAG={tag} DONE\n")
