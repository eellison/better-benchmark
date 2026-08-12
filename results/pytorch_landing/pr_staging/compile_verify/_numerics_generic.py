"""
Generic shadow numerics driver: compile a repro at the shadowed branch commit,
compare compiled vs eager on realistic inputs. Optionally force given input
indices positive (BN variance inputs that feed sqrt).

Usage: python _numerics_generic.py <repro.py> <shape> <tag> [pos_idx,comma,sep]
"""
import sys, os, importlib.util, math
from pathlib import Path
ROOT = Path("/tmp/scratch_space/better_benchmark"); sys.path.insert(0, str(ROOT))
repro_path, shape, tag = sys.argv[1:4]
pos_idx = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 and sys.argv[4] else []
import torch
torch.manual_seed(0)
pp = os.environ.get("PYTHONPATH", "/__none__").split(":")[0]
assert torch.__file__.startswith(pp), f"NOT shadowed: {torch.__file__}"
print(f"### {tag} torch={torch.__file__}")
spec = importlib.util.spec_from_file_location("repro", repro_path)
mod = importlib.util.module_from_spec(spec)
mod.device = torch.device; mod.inf = math.inf; mod.nan = math.nan
spec.loader.exec_module(mod)
from repro_harness import load_shape_configs, make_inputs_from_config
cfg = load_shape_configs(repro_path)
key = shape if shape in cfg else [k for k in cfg if shape in k or shape == "default"][0]
inputs = list(make_inputs_from_config(cfg[key]))
for i in pos_idx:
    if i < len(inputs) and torch.is_floating_point(inputs[i]):
        inputs[i] = inputs[i].abs() + 0.5
inst = mod.Repro()
with torch.no_grad():
    eager = inst(*inputs)
torch._dynamo.reset()
comp = torch.compile(inst)
with torch.no_grad():
    out = comp(*inputs); torch.cuda.synchronize()
eo = eager[0] if isinstance(eager, (tuple, list)) else eager
co = out[0] if isinstance(out, (tuple, list)) else out
d = (eo.float() - co.float()).abs()
den = eo.float().abs().clamp_min(1e-3)
nan_e = torch.isnan(eo).any().item(); nan_c = torch.isnan(co).any().item()
print(f"### {tag} eager-vs-compiled: max_abs={d.max():.3e} mean_abs={d.mean():.3e} "
      f"max_rel={(d/den).max():.3e} nan_eager={nan_e} nan_compiled={nan_c} "
      f"shapes={tuple(eo.shape)} dtype={co.dtype}")
print(f"### {tag} NUMERICS_DONE")
