"""
Shared benchmark harness for canonical repros.

Each canonical repro.py imports from here instead of inlining benchmark logic.
Supports: multi-shape configs, SOL measurement, CD tuning, CUDA graph timing,
alias-aware byte accounting, Triton kernel counting, and adjusted byte counting
via dispatch-mode tracing.
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch._inductor.config as inductor_config


def load_shape_configs(repro_file: str) -> dict:
    """Load shapes from shapes.txt (compact T()/S() format).

    Falls back to shapes.json for legacy repos.
    """
    repro_dir = Path(repro_file).parent

    shapes_txt = repro_dir / "shapes.txt"
    if shapes_txt.exists():
        return _parse_shapes_txt(shapes_txt)

    # Legacy: verbose JSON format
    shapes_json = repro_dir / "shapes.json"
    if shapes_json.exists():
        with open(shapes_json) as f:
            data = json.load(f)
        return data.get("configs", {})

    return {}


def _parse_shapes_txt(shapes_path: Path) -> dict:
    """Parse shapes.txt by eval'ing each line with T() and S() as constructors."""

    _DTYPE_MAP = {
        "f32": "torch.float32", "f16": "torch.float16",
        "bf16": "torch.bfloat16", "f64": "torch.float64",
        "i64": "torch.int64", "i32": "torch.int32",
        "i16": "torch.int16", "i8": "torch.int8",
        "b8": "torch.bool", "u8": "torch.uint8",
    }

    def T(shape, dtype, stride=None, max=None):
        return {"kind": "tensor", "shape": shape, "dtype": _DTYPE_MAP.get(dtype, f"torch.{dtype}"), "stride": list(stride) if stride else None, "device": "cuda", "max_val": max}

    def S(dims):
        return {"kind": "shape", "dims": dims}

    _eval_ns = {"__builtins__": {}, "T": T, "S": S,
                "f32": "f32", "f16": "f16", "bf16": "bf16", "f64": "f64",
                "i64": "i64", "i32": "i32", "i16": "i16", "i8": "i8",
                "b8": "b8", "u8": "u8"}

    configs = {}
    for line in shapes_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        colon = line.find(":")
        if colon < 0:
            continue
        label = line[:colon].strip()
        expr = line[colon + 1:].strip()

        try:
            inputs = eval(expr, _eval_ns)
            if isinstance(inputs, tuple):
                inputs = list(inputs)
            elif isinstance(inputs, dict):
                inputs = [inputs]
            if inputs:
                configs[label] = {"inputs": inputs}
        except Exception:
            continue

    return configs


def parse_shapes_config(config_str: str) -> list:
    """Parse a _shapes_config string and generate inputs. Used by repro._default_make_inputs()."""
    _DTYPE_MAP = {
        "f32": "torch.float32", "f16": "torch.float16",
        "bf16": "torch.bfloat16", "f64": "torch.float64",
        "i64": "torch.int64", "i32": "torch.int32",
        "i16": "torch.int16", "i8": "torch.int8",
        "b8": "torch.bool", "u8": "torch.uint8",
    }

    def T(shape, dtype, stride=None, max=None):
        return {"kind": "tensor", "shape": shape, "dtype": _DTYPE_MAP.get(dtype, f"torch.{dtype}"), "stride": list(stride) if stride else None, "device": "cuda", "max_val": max}

    def S(dims):
        return {"kind": "shape", "dims": dims}

    _ns = {"__builtins__": {}, "T": T, "S": S,
            "f32": "f32", "f16": "f16", "bf16": "bf16", "f64": "f64",
            "i64": "i64", "i32": "i32", "i16": "i16", "i8": "i8",
            "b8": "b8", "u8": "u8"}
    inputs = eval(config_str, _ns)
    if isinstance(inputs, dict):
        inputs = [inputs]
    elif isinstance(inputs, tuple):
        inputs = list(inputs)
    return make_inputs_from_config({"inputs": inputs})


def make_inputs_from_config(config: dict) -> list:
    """Create inputs from a shape config. Returns mix of tensors and shape-param lists."""
    result = []
    for spec in config["inputs"]:
        if spec.get("kind") == "shape":
            result.append(spec["dims"])
            continue

        shape = spec["shape"]
        dtype_str = spec["dtype"].replace("torch.", "")
        dtype = getattr(torch, dtype_str)
        stride = spec.get("stride")
        device = spec.get("device", "cuda")

        if dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
            hi = spec.get("max_val") or 100
            if stride:
                numel = sum((s - 1) * st for s, st in zip(shape, stride) if s > 1) + 1
                storage = torch.randint(0, hi, (numel,), dtype=dtype, device=device)
                result.append(storage.as_strided(shape, stride))
            else:
                result.append(torch.randint(0, hi, shape, dtype=dtype, device=device))
        elif dtype == torch.bool:
            if stride:
                numel = sum((s - 1) * st for s, st in zip(shape, stride) if s > 1) + 1
                storage = torch.randint(0, 2, (numel,), dtype=torch.bool, device=device)
                result.append(storage.as_strided(shape, stride))
            else:
                result.append(torch.randint(0, 2, shape, dtype=torch.bool, device=device))
        else:
            if stride:
                numel = sum((s - 1) * st for s, st in zip(shape, stride) if s > 1) + 1
                storage = torch.randn(numel, dtype=dtype, device=device)
                result.append(storage.as_strided(shape, stride))
            else:
                result.append(torch.randn(shape, dtype=dtype, device=device))
    return result


def count_bytes_naive(inputs, outputs) -> int:
    total = 0
    for t in inputs:
        if isinstance(t, torch.Tensor):
            total += t.nelement() * t.element_size()
    if isinstance(outputs, torch.Tensor):
        total += outputs.nelement() * outputs.element_size()
    elif isinstance(outputs, (tuple, list)):
        for o in outputs:
            if isinstance(o, torch.Tensor):
                total += o.nelement() * o.element_size()
    return total


def count_bytes_adjusted(mod, inputs) -> int:
    """Count actual DRAM bytes via dispatch-mode op tracing.

    Handles view-only pass-throughs (not counted) and index/gather
    (counts output size, not full input).
    """
    from torch.utils._python_dispatch import TorchDispatchMode

    _VIEW_OPS = {
        torch.ops.aten.permute.default, torch.ops.aten.transpose.int,
        torch.ops.aten.t.default, torch.ops.aten.reshape.default,
        torch.ops.aten.view.default, torch.ops.aten.expand.default,
        torch.ops.aten.slice.Tensor, torch.ops.aten.unsqueeze.default,
        torch.ops.aten.squeeze.default, torch.ops.aten.squeeze.dim,
        torch.ops.aten.as_strided.default, torch.ops.aten.select.int,
    }
    _INDEX_OPS = {
        torch.ops.aten.index.Tensor, torch.ops.aten.gather.default,
        torch.ops.aten.index_select.default, torch.ops.aten.embedding.default,
    }

    class _ByteCount(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self._produced = set()
            self._views_of = {}
            self._input_bytes = {}
            self._computed_inputs = set()
            self._gather_adj = {}

        def _root(self, tid):
            seen = set()
            while tid in self._views_of and tid not in seen:
                seen.add(tid)
                tid = self._views_of[tid]
            return tid

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}

            def track(x):
                if isinstance(x, torch.Tensor) and id(x) not in self._produced:
                    if id(x) not in self._input_bytes:
                        self._input_bytes[id(x)] = x.numel() * x.element_size()

            torch.utils._pytree.tree_map(
                lambda x: track(x) if isinstance(x, torch.Tensor) else None, (args, kwargs))
            result = func(*args, **kwargs)

            def mark(x):
                if isinstance(x, torch.Tensor):
                    self._produced.add(id(x))

            torch.utils._pytree.tree_map(
                lambda x: mark(x) if isinstance(x, torch.Tensor) else None, result)

            if func in _VIEW_OPS and args and isinstance(args[0], torch.Tensor):
                if isinstance(result, torch.Tensor):
                    self._views_of[id(result)] = id(args[0])
            else:
                def mark_computed(x):
                    if isinstance(x, torch.Tensor):
                        root = self._root(id(x))
                        if root in self._input_bytes:
                            self._computed_inputs.add(root)

                torch.utils._pytree.tree_map(
                    lambda x: mark_computed(x) if isinstance(x, torch.Tensor) else None, (args, kwargs))
                if func in _INDEX_OPS and args and isinstance(args[0], torch.Tensor):
                    root = self._root(id(args[0]))
                    if root in self._input_bytes and isinstance(result, torch.Tensor):
                        gathered = result.numel() * result.element_size()
                        self._gather_adj[root] = self._gather_adj.get(root, 0) + gathered

            return result

        def total(self, outputs):
            read = sum(
                min(self._gather_adj.get(t, self._input_bytes[t]), self._input_bytes[t])
                for t in self._computed_inputs
            )
            write = 0
            if isinstance(outputs, torch.Tensor):
                write = outputs.numel() * outputs.element_size()
            elif isinstance(outputs, (tuple, list)):
                for o in outputs:
                    if isinstance(o, torch.Tensor):
                        write += o.numel() * o.element_size()
            return read + write

    with _ByteCount() as bc:
        with torch.no_grad():
            out = mod(*inputs)
    return bc.total(out)


def count_kernels(mod, inputs) -> tuple[int, list[str]]:
    """Compile and count how many Triton kernels Inductor generates."""
    from torch._inductor.utils import fresh_inductor_cache
    from torch._inductor.codecache import cache_dir

    torch._dynamo.reset()
    with fresh_inductor_cache():
        compiled = torch.compile(mod)
        with torch.no_grad():
            compiled(*inputs)
            torch.cuda.synchronize()
        cd = cache_dir()
        py_files = sorted(glob.glob(os.path.join(cd, "**", "*.py"), recursive=True), key=os.path.getmtime)
        for f in reversed(py_files):
            with open(f) as fh:
                content = fh.read()
            if 'def call(' in content and '.run(' in content:
                runs = [l for l in content.split('\n') if '.run(' in l and not l.strip().startswith('#')]
                names = []
                for r in runs:
                    name = r.strip().split('.run(')[0]
                    names.append(name)
                return len(names), names
    return 0, []


def load_perf(repro_file: str) -> dict:
    """Load perf.json for a canonical repro. Returns {hardware: {shape: results}}."""
    perf_path = Path(repro_file).parent / "perf.json"
    if not perf_path.exists():
        return {}
    with open(perf_path) as f:
        return json.load(f)


def _save_perf(repro_file: str, hardware: str, shape_label: str, result: dict):
    """Append a benchmark result to perf.json, keyed by hardware + shape."""
    perf_path = Path(repro_file).parent / "perf.json"
    if perf_path.exists():
        with open(perf_path) as f:
            perf = json.load(f)
    else:
        perf = {}

    if hardware not in perf:
        perf[hardware] = {}

    perf[hardware][shape_label] = result

    with open(perf_path, "w") as f:
        json.dump(perf, f, indent=2)


def _detect_hardware() -> str:
    """Auto-detect GPU hardware kind from nvidia-smi."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, timeout=5
        ).strip().split("\n")[0]
        for kind in ("B200", "H200", "H100", "A100", "V100"):
            if kind in out:
                return kind
        return out.replace(" ", "_")[:20]
    except Exception:
        return "unknown"


def benchmark_repro(repro_file: str, repro_cls, make_inputs_fn, args=None):
    """Full benchmark entry point for canonical repros.

    Supports --shape, --all-shapes, --hardware, --no-cd, --output,
    --count-kernels-only, --update-perf, --gpu, --device-kind.
    """
    from triton.testing import do_bench

    parser = argparse.ArgumentParser(description="Benchmark canonical repro")
    parser.add_argument("--shape", type=str, default=None,
                        help="Named shape config from shapes.json")
    parser.add_argument("--all-shapes", action="store_true",
                        help="Benchmark all shape configs")
    parser.add_argument("--hardware", type=str, default=None,
                        help="Hardware label for perf.json (e.g., H100, B200). Auto-detected if not set.")
    parser.add_argument("--update-perf", action="store_true",
                        help="Write results to perf.json (keyed by hardware + shape)")
    parser.add_argument("--output", type=str, default=None,
                        help="Write JSON results to file")
    parser.add_argument("--no-cd", action="store_true",
                        help="Skip coordinate descent tuning")
    parser.add_argument("--count-kernels-only", action="store_true",
                        help="Only count generated kernels, skip timing")
    parser.add_argument("--n-warmup", type=int, default=25)
    parser.add_argument("--n-rep", type=int, default=200)
    parser.add_argument("--gpu", default=None,
                        help="GPU id to lock (default: grab any free GPU)")
    parser.add_argument("--device-kind", default=None,
                        help="GPU kind to lock, e.g. H100 or B200")
    parser.add_argument("--no-gpu-lock", action="store_true",
                        help="Disable the per-GPU benchmark lock")
    parsed = parser.parse_args(args)

    def _run_benchmark():
        configs = load_shape_configs(repro_file)

        if parsed.all_shapes and configs:
            shape_names = list(configs.keys())
        elif parsed.shape and parsed.shape in configs:
            shape_names = [parsed.shape]
        else:
            shape_names = [None]

        all_results = {}

        for name in shape_names:
            if name is not None:
                inputs = make_inputs_from_config(configs[name])
                label = name
                # If shapes.json doesn't include shape params but forward() expects them,
                # merge shape params from _default_make_inputs at the correct positions
                default_inputs = make_inputs_fn()
                if len(inputs) < len(default_inputs):
                    # Build merged list: use config tensors where available,
                    # fill shape params (plain lists/ints) from defaults
                    merged = []
                    config_idx = 0
                    for di in default_inputs:
                        if isinstance(di, (list, int)) and not isinstance(di, torch.Tensor):
                            # Shape param or scalar — take from default
                            merged.append(di)
                        else:
                            # Tensor — take from config if available
                            if config_idx < len(inputs):
                                merged.append(inputs[config_idx])
                                config_idx += 1
                            else:
                                merged.append(di)
                    inputs = merged
            else:
                inputs = make_inputs_fn()
                label = "default"

            mod = repro_cls()

            with torch.no_grad():
                eager_out = mod(*inputs)

            total_bytes_naive = count_bytes_naive(inputs, eager_out)
            total_bytes = count_bytes_adjusted(mod, inputs)
            if total_bytes_naive > total_bytes * 1.1:
                print(f"\n[{label}] Bytes adjusted: {total_bytes_naive/1e6:.1f} MB naive -> {total_bytes/1e6:.1f} MB actual")

            n_kernels, kernel_names = count_kernels(mod, inputs)
            print(f"\n[{label}] Kernels generated: {n_kernels}")
            for kn in kernel_names:
                print(f"  {kn}")

            if parsed.count_kernels_only:
                all_results[label] = {
                    "n_kernels": n_kernels,
                    "kernel_names": kernel_names,
                    "total_bytes": total_bytes,
                }
                continue

            # SOL: memcopy same total bytes
            copy_elems = max(total_bytes // (2 * 4), 256)
            src = torch.empty(copy_elems, dtype=torch.float32, device="cuda")
            dst = torch.empty_like(src)
            sol_ms = do_bench(lambda: dst.copy_(src), warmup=parsed.n_warmup, rep=parsed.n_rep)
            sol_us = sol_ms * 1000
            del src, dst

            # Compiled (default heuristics)
            torch._dynamo.reset()
            compiled = torch.compile(mod)
            with torch.no_grad():
                for _ in range(3):
                    compiled(*inputs)
                torch.cuda.synchronize()
            compiled_ms = do_bench(lambda: compiled(*inputs), warmup=parsed.n_warmup, rep=parsed.n_rep)
            compiled_us = compiled_ms * 1000

            # Compiled with coordinate descent tuning
            cd_us = None
            if not parsed.no_cd:
                inductor_config.coordinate_descent_tuning = True
                torch._dynamo.reset()
                compiled_cd = torch.compile(mod)
                with torch.no_grad():
                    for _ in range(3):
                        compiled_cd(*inputs)
                    torch.cuda.synchronize()
                cd_ms = do_bench(lambda: compiled_cd(*inputs), warmup=parsed.n_warmup, rep=parsed.n_rep)
                cd_us = cd_ms * 1000
                inductor_config.coordinate_descent_tuning = False

            print(f"\n[{label}] Kernel data: {total_bytes / 1024:.1f} KB (read+write)")
            print(f"[{label}] Memcopy SOL (same size): {sol_us:8.1f} us")
            print(f"[{label}] Compiled (default):      {compiled_us:8.1f} us")
            if cd_us is not None:
                print(f"[{label}] Compiled (coord desc):   {cd_us:8.1f} us")
            print(f"[{label}] Gap (default / SOL):     {compiled_us / sol_us:8.2f}x")
            if cd_us is not None:
                print(f"[{label}] Gap (CD / SOL):          {cd_us / sol_us:8.2f}x")

            all_results[label] = {
                "compiled_us": compiled_us,
                "coord_descent_us": cd_us,
                "memcopy_sol_us": sol_us,
                "total_bytes": total_bytes,
                "n_kernels": n_kernels,
                "kernel_names": kernel_names,
                "gap_default": compiled_us / sol_us if sol_us > 0 else None,
                "gap_cd": cd_us / sol_us if (cd_us and sol_us > 0) else None,
            }

        if parsed.output:
            with open(parsed.output, "w") as f:
                json.dump(all_results, f, indent=2)

        if parsed.update_perf:
            hardware = parsed.hardware or _detect_hardware()
            for shape_label, result in all_results.items():
                perf_entry = {k: v for k, v in result.items() if k != "kernel_names"}
                perf_entry["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                _save_perf(repro_file, hardware, shape_label, perf_entry)
            print(f"\n[perf] Saved to perf.json under hardware={hardware}")

        return all_results

    # Acquire GPU lock before any CUDA work. Since `import torch` doesn't
    # initialize CUDA, setting CUDA_VISIBLE_DEVICES here (before the first
    # .cuda() call) is safe.
    if parsed.no_gpu_lock:
        if parsed.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(parsed.gpu)
        return _run_benchmark()

    scripts_dir = Path(__file__).resolve().parent / "scripts"
    sys.path.insert(0, str(scripts_dir))
    from gpu_lock import gpu_lock, gpu_lock_for_kind

    if parsed.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(parsed.gpu)
        with gpu_lock(parsed.gpu, label=f"repro {Path(repro_file).stem}") as lock_path:
            print(f"[gpu-lock] acquired {lock_path}")
            return _run_benchmark()
    else:
        with gpu_lock_for_kind(
            parsed.device_kind,
            label=f"repro {Path(repro_file).stem}",
        ) as device_info:
            os.environ["CUDA_VISIBLE_DEVICES"] = device_info["index"]
            print(
                f"[gpu-lock] acquired {device_info['lock_path']} "
                f"({device_info['kind']} {device_info['name']})"
            )
            return _run_benchmark()
