"""
Numerics: compile the repro at the shadowed commit, compare compiled output to
eager output on REALISTIC BN inputs (variance>0). Writes the compiled output to
a .pt so pos22 and pos23 can be cross-compared bit-for-bit-ish.
"""
import sys, os, importlib.util, math
from pathlib import Path
ROOT = Path("/tmp/scratch_space/better_benchmark"); sys.path.insert(0, str(ROOT))
repro_path, shape, tag, savepath = sys.argv[1:5]
import torch
torch.manual_seed(0)
spec=importlib.util.spec_from_file_location("repro",repro_path); mod=importlib.util.module_from_spec(spec)
mod.device=torch.device; mod.inf=math.inf; mod.nan=math.nan; spec.loader.exec_module(mod)
from repro_harness import load_shape_configs, make_inputs_from_config
cfg=load_shape_configs(repro_path); key=[k for k in cfg if shape in k][0]
inputs=list(make_inputs_from_config(cfg[key]))
# Make the BN "running_var" inputs (arg2_1 idx2, arg7_1 idx6 are var: positive)
# Repro forward: arg2_1 -> add(.,1e-5)->sqrt; arg7_1 likewise. Force positive.
for vi in (2, 7):
    inputs[vi] = inputs[vi].abs() + 0.5
inst=mod.Repro()
with torch.no_grad():
    eager=inst(*inputs)
torch._dynamo.reset()
comp=torch.compile(inst)
with torch.no_grad():
    out=comp(*inputs); torch.cuda.synchronize()
eo=eager[0] if isinstance(eager,(tuple,list)) else eager
co=out[0] if isinstance(out,(tuple,list)) else out
d=(eo.float()-co.float()).abs()
den=eo.float().abs().clamp_min(1e-3)
print(f"### {tag} eager-vs-compiled: max_abs={d.max():.3e} mean_abs={d.mean():.3e} "
      f"max_rel={(d/den).max():.3e} nan_in_eager={torch.isnan(eo).any().item()} "
      f"dtype={co.dtype}")
torch.save(co.float().cpu(), savepath)
print(f"### {tag} saved compiled output -> {savepath}")
