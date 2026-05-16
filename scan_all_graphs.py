"""
Scan all post-grad graphs from model repros for pointwise_cat split candidates.

This runs the diagnostic scanner across every repro in multi_kernel_inference/
and reports what would/wouldn't be split.
"""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
import torch._inductor.split_cat_pointwise_diag as diag
import importlib.util
import os

cfg.force_disable_caches = True

repro_dir = "repo/repros/multi_kernel_inference"
repros = sorted(f for f in os.listdir(repro_dir) if f.endswith(".py"))

print(f"Scanning {len(repros)} repro files...\n")

diag.install()

for filename in repros:
    filepath = os.path.join(repro_dir, filename)
    spec = importlib.util.spec_from_file_location("repro_mod", filepath)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        continue

    if not hasattr(mod, "make_inputs") or not hasattr(mod, "Repro"):
        continue

    try:
        inputs = mod.make_inputs()
    except Exception:
        continue

    torch._dynamo.reset()
    try:
        compiled = torch.compile(mod.Repro())
        with torch.no_grad():
            compiled(*inputs)
    except Exception as e:
        pass

diag.report()
diag.uninstall()
