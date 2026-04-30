"""Run a single repro with TORCH_LOGS to see fusion decisions."""
import sys
import os
import builtins
import torch
import torch._inductor.config as cfg
import importlib.util

cfg.force_disable_caches = True
builtins.device = torch.device
builtins.inf = float("inf")
builtins.nan = float("nan")

os.environ["TORCH_LOGS"] = "ir_pre_fusion,fusion,ir_post_fusion"
torch._logging._init_logs()

path = sys.argv[1]
spec = importlib.util.spec_from_file_location("region", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
model = mod.Repro().cuda()
inputs = mod.make_inputs()

import torch._inductor.metrics as m

m.reset()
compiled = torch.compile(model)
with torch.no_grad():
    compiled(*inputs)
print(
    f"\n=== KERNELS: {m.generated_kernel_count} triton + {m.generated_cpp_vec_kernel_count} cpp ==="
)
