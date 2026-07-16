import sys, os, importlib.util, math
from pathlib import Path
ROOT = Path("/tmp/scratch_space/better_benchmark"); sys.path.insert(0, str(ROOT))
import torch
torch.manual_seed(0)
repro_path = "repros/canonical/sum_sum_sum_260a107eaf32/repro.py"
spec = importlib.util.spec_from_file_location("repro", repro_path)
mod = importlib.util.module_from_spec(spec)
mod.device = torch.device; mod.inf = math.inf; mod.nan = math.nan
spec.loader.exec_module(mod)
from repro_harness import load_shape_configs, make_inputs_from_config
cfg = load_shape_configs(repro_path)
inputs = list(make_inputs_from_config(cfg[next(iter(cfg))]))
inst = mod.Repro()
with torch.no_grad():
    eager = inst(*inputs)
    # fp64 reference
    inst64 = mod.Repro()
    in64 = [x.double() if torch.is_tensor(x) and torch.is_floating_point(x) else x for x in inputs]
    try:
        ref = inst64(*in64)
    except Exception as e:
        ref = None; print("fp64 ref failed:", e)
torch._dynamo.reset()
comp = torch.compile(inst)
with torch.no_grad():
    out = comp(*inputs); torch.cuda.synchronize()
for i,(e,c) in enumerate(zip(eager,out)):
    if not torch.is_tensor(e): continue
    ef, cf = e.float(), c.float()
    d = (ef-cf).abs()
    rel = (d/ef.abs().clamp_min(1e-3)).max().item()
    line = f"out[{i}] shape={tuple(e.shape)} dtype={e.dtype} max_abs={d.max().item():.3e} max_rel_vs_eager={rel:.3e}"
    if ref is not None:
        r = ref[i].float()
        de = (ef-r).abs().max().item(); dc = (cf-r).abs().max().item()
        line += f" | vs_fp64: eager_err={de:.3e} compiled_err={dc:.3e}"
    print(line, flush=True)
