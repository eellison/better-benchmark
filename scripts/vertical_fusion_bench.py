"""Check and time hand-written Helion kernels for one vertical MM repro.

A canonical repro directory may hold ``helion_vertical.py`` that defines

  fused(*inputs)     the whole region as ONE Helion kernel: one top-level loop,
                     no hl.barrier, intermediates kept on chip (None if impossible)
  unfused(*inputs)   the same region with one Helion kernel per reduction stage
  NOTES              feasibility notes (str)

and ends with ``if __name__ == "__main__": run(__file__, fused, unfused, NOTES)``.

run() builds the inputs of a shape point of the repro (shapes.json), checks
fused/unfused against the repro's own eager Repro, and times eager,
torch.compile (Inductor defaults), unfused and fused with the repository's
methodology (3 warmups, CUDAGraph capture, triton do_bench min). Helion
kernels use autotune_effort="quick"; autotuning happens on the first call and
is reported separately. The CUDA kernels each version launches are counted
with the profiler (fused must be exactly 1). Results go to
helion_vertical_result.json.

    python scripts/with_gpu_lock.py --gpu 0 -- \\
        python benchmarks/vertical_helion/new/<repro>/helion_vertical.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

import torch
from torch.utils._pytree import tree_leaves

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from repro_harness import _bench_cudagraph_min_us, load_shape_configs  # noqa: E402

# This PyTorch build reports torch.version.cuda = None, which Helion's local
# autotune cache needs for its key: use the compiled CUDA version instead.
if torch.version.cuda is None and torch.cuda.is_available():
    _cuda = torch._C._cuda_getCompiledVersion()
    torch.version.cuda = f"{_cuda // 1000}.{_cuda % 1000 // 10}"


def load_repro(directory: Path):
    spec = importlib.util.spec_from_file_location(
        "vertical_repro_" + directory.name, directory / "repro.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tensors(value):
    return [t for t in tree_leaves(value) if isinstance(t, torch.Tensor)]


def compare(expected, actual):
    """Max abs error, scaled by the reference magnitude, per output."""
    expected, actual = _tensors(expected), _tensors(actual)
    if len(expected) != len(actual):
        return {"ok": False, "reason": f"{len(actual)} outputs, expected {len(expected)}"}
    errors = []
    ok = True
    for e, a in zip(expected, actual):
        if e.shape != a.shape:
            return {"ok": False, "reason": f"shape {tuple(a.shape)} != {tuple(e.shape)}"}
        if e.dtype == torch.bool:
            mismatch = (e != a).float().mean().item()
            errors.append(mismatch)
            ok &= mismatch == 0
            continue
        e32, a32 = e.float(), a.float()
        scale = e32.abs().max().item() or 1.0
        err = (e32 - a32).abs().max().item() / scale
        errors.append(err)
        # The networks run in bf16: f32 outputs usually hold bf16-rounded
        # values, so accumulation-order differences show up at bf16 level.
        ok &= err <= 2e-2
    return {"ok": bool(ok), "scaled_max_error": errors}


def count_kernels(fn, inputs):
    """CUDA kernels one call launches (memcpy/memset excluded), and their names."""
    from torch.profiler import ProfilerActivity, profile

    fn(*inputs)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(*inputs)
        torch.cuda.synchronize()
    names = [
        e.name
        for e in prof.events()
        if e.device_type == torch.autograd.DeviceType.CUDA
        and not e.name.startswith(("Memcpy", "Memset"))
    ]
    return len(names), sorted(set(names))


def time_us(fn, inputs, label, result):
    try:
        result[label + "_us"] = round(_bench_cudagraph_min_us(fn, inputs, 25, 100), 2)
    except Exception:  # noqa: BLE001 -- report and keep going
        result[label + "_error"] = traceback.format_exc(limit=3)


def run(kernel_file, fused, unfused, notes="", shape=None, variants=None):
    """variants: optional {name: fn} alternatives, checked and timed like fused."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=int, default=shape or 0, help="shape point index")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    kernel_file = Path(kernel_file).resolve()
    directory = kernel_file.parent
    module = load_repro(directory)
    configs = list(load_shape_configs(str(directory / "repro.py")).values())
    config = configs[args.shape]
    torch.manual_seed(0)
    inputs = module.make_inputs(config)
    model = module.Repro()
    result = {
        "repro": directory.name,
        "shape_index": args.shape,
        "inputs": [list(t.shape) if isinstance(t, torch.Tensor) else t for t in inputs],
        "device": torch.cuda.get_device_name(),
        "notes": notes,
    }
    with torch.no_grad():
        expected = model(*inputs)
        candidates = [("unfused", unfused), ("fused", fused), *(variants or {}).items()]
        for label, fn in candidates:
            if fn is None:
                result[label] = "not written"
                continue
            start = time.perf_counter()
            try:
                actual = fn(*inputs)  # first call autotunes
            except Exception:  # noqa: BLE001 -- report and keep going
                result[label + "_error"] = traceback.format_exc(limit=6)
                continue
            torch.cuda.synchronize()
            result[label + "_first_call_s"] = round(time.perf_counter() - start, 1)
            result[label + "_check"] = compare(expected, actual)
            result[label + "_kernels"], result[label + "_kernel_names"] = count_kernels(
                fn, inputs
            )
        time_us(model, inputs, "eager", result)
        if not args.no_compile:
            compiled = torch.compile(module.Repro())
            time_us(compiled, inputs, "inductor", result)
            result["inductor_kernels"], _ = count_kernels(compiled, inputs)
        for label, fn in candidates:
            if fn is not None and label + "_check" in result:
                time_us(fn, inputs, label, result)
    if "fused_us" in result and "unfused_us" in result:
        result["fused_speedup_vs_unfused"] = round(result["unfused_us"] / result["fused_us"], 3)
    if "fused_us" in result and "inductor_us" in result:
        result["fused_speedup_vs_inductor"] = round(result["inductor_us"] / result["fused_us"], 3)
    out = directory / "helion_vertical_result.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result
