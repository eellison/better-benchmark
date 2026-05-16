"""For each extracted region, compile with inductor and count generated kernels.

Runs each region in a subprocess to isolate CUDA errors.
"""
import glob
import json
import os
import subprocess
import sys


WORKER_SCRIPT = '''
import sys, os, builtins, torch, torch._inductor.config as cfg, importlib.util
cfg.force_disable_caches = True
# FX print_readable uses bare `device(...)` — make it available
builtins.device = torch.device
builtins.inf = float("inf")
builtins.nan = float("nan")
path = sys.argv[1]
spec = importlib.util.spec_from_file_location("region", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
model = mod.Repro().cuda()
inputs = mod.make_inputs()
from torch._inductor.metrics import generated_kernel_count, generated_cpp_vec_kernel_count
import torch._inductor.metrics as m
m.reset()
compiled = torch.compile(model)
with torch.no_grad():
    compiled(*inputs)
print(f"KERNELS:{m.generated_kernel_count}:{m.generated_cpp_vec_kernel_count}")
'''


def count_kernels_subprocess(region_path, device_id=0):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device_id)
    env["PYTHONPATH"] = os.environ.get("PYTORCH_DIR", "/tmp/pytorch-work")
    result = subprocess.run(
        [sys.executable, "-c", WORKER_SCRIPT, region_path],
        capture_output=True, text=True, timeout=120, env=env,
    )
    for line in result.stdout.splitlines():
        if line.startswith("KERNELS:"):
            parts = line.split(":")
            return int(parts[1]), int(parts[2])
    return None, result.stderr[-500:] if result.stderr else "no output"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Directory or single .py file")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--multi-only", action="store_true")
    args = parser.parse_args()

    if os.path.isfile(args.path):
        files = [args.path]
    else:
        files = sorted(glob.glob(os.path.join(args.path, "**", "region_*.py"), recursive=True))

    total = 0
    multi = 0
    errors = 0
    results = []

    for f in files:
        rel = os.path.relpath(f)
        try:
            ret = count_kernels_subprocess(f, args.device)
            if ret[0] is not None:
                n_triton, n_cpp = ret
                total_k = n_triton + n_cpp
                results.append({"file": rel, "triton": n_triton, "cpp": n_cpp, "total": total_k})
                total += 1
                if total_k > 1:
                    multi += 1
                if total_k > 1 or not args.multi_only:
                    label = " <-- MULTI" if total_k > 1 else ""
                    print(f"{rel}: {n_triton} triton + {n_cpp} cpp = {total_k}{label}")
            else:
                errors += 1
                err_msg = ret[1][:100] if ret[1] else "unknown"
                print(f"{rel}: ERROR {err_msg}")
        except subprocess.TimeoutExpired:
            errors += 1
            print(f"{rel}: TIMEOUT")
        except Exception as e:
            errors += 1
            print(f"{rel}: ERROR {e}")

    print(f"\n{multi}/{total} regions produce >1 kernel ({errors} errors)")
