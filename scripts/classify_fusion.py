"""Classify multi-kernel repros by their fusion failure reasons."""
import subprocess
import sys
import os
import json
import re

WORKER = '''
import sys, os, builtins, torch, torch._inductor.config as cfg, importlib.util, logging
cfg.force_disable_caches = True
builtins.device = torch.device
builtins.inf = float("inf")
builtins.nan = float("nan")
os.environ["TORCH_LOGS"] = "fusion"
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
print(f"KERNELS:{m.generated_kernel_count}:{m.generated_cpp_vec_kernel_count}")
'''

def classify(path, device_id=0):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device_id)
    env["PYTHONPATH"] = os.environ.get("PYTORCH_DIR", "/tmp/pytorch-work")
    try:
        r = subprocess.run(
            [sys.executable, "-c", WORKER, path],
            capture_output=True, text=True, timeout=180, env=env,
        )
        stderr = r.stderr or ""
        stdout = r.stdout or ""

        kernels = None
        for line in stdout.splitlines():
            if line.startswith("KERNELS:"):
                parts = line.split(":")
                kernels = int(parts[1]) + int(parts[2])

        # Extract "cannot fuse" reasons
        reasons = []
        for line in stderr.splitlines():
            if "cannot fuse" in line.lower():
                reasons.append(line.strip())

        # Extract op info from BEFORE FUSION
        ops = []
        for line in stderr.splitlines():
            if line.startswith("[DEBUG]:  SchedulerNode") or line.startswith("[DEBUG]:  NopKernel"):
                ops.append(line.strip())

        # Extract fusion results
        fused_into = []
        for line in stderr.splitlines():
            if "fused" in line and "nodes into" in line:
                fused_into.append(line.strip())

        # Deduplicate reasons
        reason_set = set()
        for r_text in reasons:
            # Normalize op names
            cleaned = re.sub(r'op\d+', 'opN', r_text)
            reason_set.add(cleaned)

        return {
            "kernels": kernels,
            "reasons": sorted(reason_set),
            "reason_count": len(reasons),
            "fused_into": fused_into,
            "error": None if kernels else (stderr[-300:] if stderr else "no output"),
        }
    except subprocess.TimeoutExpired:
        return {"kernels": None, "reasons": [], "error": "TIMEOUT"}
    except Exception as e:
        return {"kernels": None, "reasons": [], "error": str(e)}

if __name__ == "__main__":
    import glob
    repro_dir = "/tmp/scratch_space/better_benchmark/output/multi_kernel_inference"
    files = sorted(glob.glob(os.path.join(repro_dir, "k*.py")))

    device = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    results = []
    for i, f in enumerate(files):
        name = os.path.basename(f)
        print(f"[{i+1}/{len(files)}] {name}...", file=sys.stderr, flush=True)
        info = classify(f, device)
        info["file"] = name
        results.append(info)
        # Print intermediate
        reasons_str = "; ".join(info["reasons"][:3]) if info["reasons"] else "NO REASONS FOUND"
        print(f"  k={info['kernels']} reasons={reasons_str}", file=sys.stderr, flush=True)

    with open("/tmp/fusion_classification.json", "w") as out:
        json.dump(results, out, indent=2)

    # Summary
    from collections import Counter
    reason_counter = Counter()
    for r in results:
        for reason in r["reasons"]:
            reason_counter[reason] += 1

    print("\n=== FUSION FAILURE REASON FREQUENCY ===")
    for reason, count in reason_counter.most_common():
        print(f"  {count:3d}x  {reason}")
