"""Probe kernel counts for a list of files, one per line from stdin or file."""
import glob
import json
import os
import subprocess
import sys

WORKER_SCRIPT = '''
import sys, os, builtins, torch, torch._inductor.config as cfg, importlib.util
cfg.force_disable_caches = True
builtins.device = torch.device
builtins.inf = float("inf")
builtins.nan = float("nan")
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
print(f"KERNELS:{m.generated_kernel_count}:{m.generated_cpp_vec_kernel_count}")
'''

def count_kernels(path, device_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device_id)
    env["PYTHONPATH"] = os.environ.get("PYTORCH_DIR", "/tmp/pytorch-work")
    try:
        r = subprocess.run(
            [sys.executable, "-c", WORKER_SCRIPT, path],
            capture_output=True, text=True, timeout=120, env=env,
        )
        for line in r.stdout.splitlines():
            if line.startswith("KERNELS:"):
                parts = line.split(":")
                return int(parts[1]), int(parts[2]), None
        return None, None, (r.stderr or "no output")[-200:]
    except subprocess.TimeoutExpired:
        return None, None, "TIMEOUT"
    except Exception as e:
        return None, None, str(e)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("filelist", help="File containing paths, one per line")
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", required=True, help="JSON output file")
    args = parser.parse_args()

    with open(args.filelist) as f:
        files = [l.strip() for l in f if l.strip()]

    results = []
    for i, path in enumerate(files):
        n_triton, n_cpp, err = count_kernels(path, args.device)
        entry = {"file": path, "triton": n_triton, "cpp": n_cpp, "error": err}
        if n_triton is not None:
            entry["total"] = n_triton + n_cpp
        results.append(entry)
        if (i + 1) % 50 == 0:
            print(f"  [{args.device}] {i+1}/{len(files)}...", file=sys.stderr)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    ok = [r for r in results if r.get("total") is not None]
    multi = [r for r in ok if r["total"] > 1]
    zero = [r for r in ok if r["total"] == 0]
    print(f"GPU {args.device}: {len(ok)}/{len(files)} ok, {len(multi)} multi-kernel, {len(zero)} zero-kernel, {len(files)-len(ok)} errors")
