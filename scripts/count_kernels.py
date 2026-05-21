"""Count generated kernels for each repro. Finds multi-kernel fusion opportunities."""
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch._dynamo
import torch._inductor.metrics as metrics


def count_kernels(repro_path: Path) -> int | None:
    metrics.reset()
    try:
        spec = importlib.util.spec_from_file_location("repro", repro_path)
        mod = importlib.util.module_from_spec(spec)
        mod.device = torch.device
        mod.inf = math.inf
        mod.nan = math.nan
        spec.loader.exec_module(mod)

        instance = mod.Repro()
        make_fn = mod.make_inputs if hasattr(mod, "make_inputs") else mod._default_make_inputs
        inputs = make_fn()

        torch._dynamo.reset()
        compiled = torch.compile(instance)
        with torch.no_grad():
            compiled(*inputs)
            torch.cuda.synchronize()

        return metrics.generated_kernel_count
    except Exception:
        return None


def main():
    canonical = Path("repros/canonical")
    repros = sorted(canonical.glob("*/repro.py"))
    print(f"Counting kernels for {len(repros)} repros...")

    multi_kernel = []
    single_kernel = []
    failed = []

    for i, repro in enumerate(repros):
        n = count_kernels(repro)
        name = repro.parent.name
        if n is None:
            failed.append(name)
        elif n > 1:
            multi_kernel.append({"name": name, "kernels": n})
            print(f"  [{i+1}/{len(repros)}] {n:2d} kernels: {name}")
        else:
            single_kernel.append(name)

        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/{len(repros)} done, {len(multi_kernel)} multi-kernel so far")

    print(f"\nResults:")
    print(f"  Single kernel: {len(single_kernel)}")
    print(f"  Multi kernel:  {len(multi_kernel)}")
    print(f"  Failed:        {len(failed)}")

    multi_kernel.sort(key=lambda x: -x["kernels"])
    print(f"\nTop multi-kernel repros:")
    for entry in multi_kernel[:30]:
        print(f"  {entry['kernels']:2d} kernels: {entry['name']}")

    # Save results
    output = Path("/tmp/kernel_counts.json")
    output.write_text(json.dumps({
        "multi_kernel": multi_kernel,
        "single_kernel_count": len(single_kernel),
        "failed": failed,
    }, indent=2))
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
