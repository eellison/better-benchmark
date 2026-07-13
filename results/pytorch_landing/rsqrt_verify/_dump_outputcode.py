"""
Dump generated Triton output_code for a repro at the current (shadowed) torch.
Compiles with torch.compile(instance) (static) -- EXACTLY the bench_parallel
default path -- and captures output_code via TORCH_LOGS-equivalent
config.trace.output_code, writing every generated kernel to OUT.

Usage:
  PYTHONPATH=<worktree> python3 _dump_outputcode.py <repro.py> <shape_name> <out_dir> <tag>
"""
import sys, os, importlib.util, math
from pathlib import Path

ROOT = Path("/tmp/scratch_space/better_benchmark")
sys.path.insert(0, str(ROOT))

repro_path = sys.argv[1]
shape_name = sys.argv[2]          # e.g. "7ee473b0" or "default"
out_dir = Path(sys.argv[3])
tag = sys.argv[4]                  # "pos22" / "pos23"

import torch
import torch._inductor.config as inductor_config

print(f"[{tag}] torch={torch.__file__}", flush=True)
assert torch.__file__.startswith(os.environ.get("PYTHONPATH", "/__none__")), \
    f"torch not shadowed: {torch.__file__}"

# Load repro module
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
else:
    # shape_name may be a shape_hash; match the config key
    if shape_name in configs:
        cfg = configs[shape_name]
    else:
        # match by hash suffix
        matches = [k for k in configs if shape_name in k or k in shape_name]
        assert matches, f"shape {shape_name} not in {list(configs)}"
        cfg = configs[matches[0]]
inputs = make_inputs_from_config(cfg)
print(f"[{tag}] shape={shape_name} input shapes={[tuple(i.shape) for i in inputs]}", flush=True)

instance = mod.Repro()

# Capture output_code to a file
out_dir.mkdir(parents=True, exist_ok=True)
code_file = out_dir / f"output_code_{tag}.py"

# Redirect inductor's output_code logging into our file by enabling the
# debug trace and reading the generated module path. Simplest robust route:
# monkeypatch the codegen "write" via TORCH_LOGS is env-level; here we use the
# inductor config to dump to a known dir, then collect.
import logging
import io

# Use torch._logging to capture "output_code" artifact
import torch._logging
torch._logging.set_logs(output_code=True)

# output_code log goes to a logger; attach a handler capturing to string
buf = io.StringIO()
handler = logging.StreamHandler(buf)
handler.setLevel(logging.DEBUG)
oc_logger = logging.getLogger("torch._inductor.codecache")
# The output_code artifact logger name varies; attach to the graph logger too
for lname in ["torch._inductor", "torch._inductor.graph",
              "torch._inductor.codecache", "torch._inductor.scheduler"]:
    lg = logging.getLogger(lname)
    lg.addHandler(handler)
    lg.setLevel(logging.DEBUG)

torch._dynamo.reset()
compiled = torch.compile(instance)
with torch.no_grad():
    out = compiled(*inputs)
    torch.cuda.synchronize()

captured = buf.getvalue()
code_file.write_text(captured)
print(f"[{tag}] captured {len(captured)} chars of logs -> {code_file}", flush=True)

# count triton kernels and key markers
n_triton = captured.count("@triton.jit")
n_rsqrt = captured.count("rsqrt") + captured.count("libdevice.rsqrt")
n_recip = captured.lower().count("reciprocal") + captured.count("1 / ") + captured.count("1.0 / ")
print(f"[{tag}] @triton.jit kernels in log: {n_triton}", flush=True)
print(f"[{tag}] rsqrt occurrences: {n_rsqrt}", flush=True)
