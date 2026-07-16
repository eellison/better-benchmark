"""
Shadow compile-smoke for a corpus repro: compile at the shadowed arm, report
- inductor pass counters (layout_transform_store_sinking)
- number of triton kernels in the generated output_code (@triton.jit count)
- count of the 'triton_mor_finalize_sum' marker in generated code
- numerics compiled-vs-eager
NOT a timing harness (no do_bench) -- correctness/compile smoke only.

Usage: PYTHONPATH=<arm> TORCHINDUCTOR_CACHE_DIR=<fresh> \
    python _smoke_repro.py <repro.py> <tag> [env overrides via environment]
"""
import sys, os, importlib.util, math
from pathlib import Path
ROOT = Path("/tmp/scratch_space/better_benchmark"); sys.path.insert(0, str(ROOT))
repro_path, tag = sys.argv[1:3]
pos_idx = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 and sys.argv[3] else []
import torch
torch.manual_seed(0)
pp = os.environ.get("PYTHONPATH", "/__none__").split(":")[0]
assert torch.__file__.startswith(pp), f"NOT shadowed: {torch.__file__}"
print(f"### {tag} torch={torch.__file__}", flush=True)
spec = importlib.util.spec_from_file_location("repro", repro_path)
mod = importlib.util.module_from_spec(spec)
mod.device = torch.device; mod.inf = math.inf; mod.nan = math.nan
spec.loader.exec_module(mod)
from repro_harness import load_shape_configs, make_inputs_from_config
cfg = load_shape_configs(repro_path)
key = next(iter(cfg))
inputs = list(make_inputs_from_config(cfg[key]))
for i in pos_idx:
    if i < len(inputs) and torch.is_tensor(inputs[i]) and torch.is_floating_point(inputs[i]):
        inputs[i] = inputs[i].abs() + 0.5
inst = mod.Repro()
with torch.no_grad():
    eager = inst(*inputs)
torch._dynamo.reset()
comp = torch.compile(inst)
# capture generated code
codes = []
from torch._inductor import codecache
orig = codecache.PyCodeCache.load_by_key_path.__func__
def spy(cls, key_, path, *a, **k):
    try:
        codes.append(open(path).read())
    except Exception:
        pass
    return orig(cls, key_, path, *a, **k)
codecache.PyCodeCache.load_by_key_path = classmethod(spy)
with torch.no_grad():
    out = comp(*inputs); torch.cuda.synchronize()
from torch._dynamo.utils import counters
cnt = dict(counters.get("inductor", {}))
lts = cnt.get("layout_transform_store_sinking", 0)
njit = sum(c.count("@triton.jit") for c in codes)
nmor = sum(c.count("def triton_mor_finalize_sum") for c in codes)
print(f"### {tag} COUNTER layout_transform_store_sinking={lts}", flush=True)
print(f"### {tag} CODEGEN njit={njit} mor_finalize_defs={nmor} n_code_files={len(codes)}", flush=True)
eo = eager[0] if isinstance(eager, (tuple, list)) else eager
co = out[0] if isinstance(out, (tuple, list)) else out
if not torch.is_floating_point(eo):
    eo = eo.float(); co = co.float()
d = (eo.float() - co.float()).abs()
den = eo.float().abs().clamp_min(1e-3)
print(f"### {tag} NUMERICS max_abs={d.max():.3e} mean_abs={d.mean():.3e} "
      f"max_rel={(d/den).max():.3e} nan_eager={torch.isnan(eo).any().item()} "
      f"nan_compiled={torch.isnan(co).any().item()} shape={tuple(eo.shape)}", flush=True)
# check ALL outputs if multiple
if isinstance(eager, (tuple, list)) and len(eager) > 1:
    for i, (e_i, c_i) in enumerate(zip(eager, out)):
        if torch.is_tensor(e_i) and torch.is_floating_point(e_i):
            dd = (e_i.float() - c_i.float()).abs().max().item()
            print(f"### {tag} NUMERICS out[{i}] max_abs={dd:.3e}", flush=True)
print(f"### {tag} DONE", flush=True)
