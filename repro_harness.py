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
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch._inductor.config as inductor_config
import torch._inductor.inductor_prims  # noqa: F401


UNVERSIONED_REPRO_VERSION = 0
CURRENT_REPRO_VERSION = 2
_REPRO_VERSION_RE = re.compile(
    r"^_repro_version\s*=\s*(\d+)\s*(?:#.*)?$",
)
_REPRO_VERSION_ASSIGN_RE = re.compile(r"^_repro_version\s*=.*$", re.MULTILINE)


def parse_repro_version(source: str) -> int:
    """Return the repro format version, or 0 for legacy unversioned source."""
    assignments = _REPRO_VERSION_ASSIGN_RE.findall(source)
    if not assignments:
        return UNVERSIONED_REPRO_VERSION
    if len(assignments) > 1:
        raise ValueError("_repro_version must appear exactly once")

    match = _REPRO_VERSION_RE.match(assignments[0])
    if match is None:
        raise ValueError("_repro_version must be a top-level integer assignment")
    return int(match.group(1))


def read_repro_version(repro_path: str | Path) -> int:
    """Read the repro format version from a repro.py path."""
    return parse_repro_version(Path(repro_path).read_text())


def load_shape_configs(repro_file: str) -> dict:
    """Load shape configs for a canonical repro.

    Preference order:
      1. shapes.json (new format with points array) — used for new captures
      2. shapes.txt (compact T()/S() format) — the existing 1482 repros
    """
    repro_dir = Path(repro_file).parent

    # Prefer shapes.json (new format from wave-1 onward)
    shapes_json = repro_dir / "shapes.json"
    if shapes_json.exists():
        return _parse_shapes_json(shapes_json)

    # Fallback: shapes.txt for existing corpus
    shapes_txt = repro_dir / "shapes.txt"
    if shapes_txt.exists():
        return _parse_shapes_txt(shapes_txt)

    return {}


def _parse_shapes_json(shapes_path: Path) -> dict:
    """Parse shapes.json (new format with points array).

    Each point has a shape_hash, signature (T()/S() string), and models dict.
    Returns the same dict format as _parse_shapes_txt: {label: {"inputs": [specs]}}.
    Label is constructed from shape_hash + first model key for readability.
    """
    with open(shapes_path) as f:
        data = json.load(f)

    # Legacy shapes.json format (pre-points schema)
    if "configs" in data and "points" not in data:
        return data.get("configs", {})

    points = data.get("points", [])
    if not points:
        return {}

    configs = {}
    for point in points:
        signature = point.get("signature", "")
        shape_hash = point.get("shape_hash", "unknown")
        models = point.get("models", {})

        # Build label: shape_hash_first_model_key (sanitized)
        first_model = next(iter(models), "")
        # Use last component of model path for brevity
        model_short = first_model.rsplit("/", 1)[-1] if first_model else ""
        label = f"{model_short}_{shape_hash}" if model_short else shape_hash

        # Prefer the structured compact entries (input_codec) — the data.
        # Signature-eval is the fallback for points written before the
        # compact encoding existed; the string is documentation otherwise.
        compact = point.get("inputs")
        if compact is not None:  # [] is a VALID zero-input config
            # NO silent fallback: data-only points have no signature to
            # fall back to, so a decode failure here must be LOUD. (The
            # old except-pass masked input_codec missing from the editable
            # package — every standalone repro run silently saw 0 configs.)
            from input_codec import spec_from_compact
            specs = [spec_from_compact(e) for e in compact]
            cfg = {"inputs": specs}
            # alias_group_nbytes must travel with the config — specs
            # carrying alias_group crash generation without it
            # (adversarial review bug #1).
            if point.get("alias_group_nbytes"):
                cfg["alias_group_nbytes"] = point["alias_group_nbytes"]
            configs[label] = cfg
            continue
        try:
            inputs = _eval_signature(signature)
            if inputs:
                configs[label] = {"inputs": inputs}
        except Exception:
            continue

    return configs


def _make_shape_eval_ns():
    """Build the eval namespace for T()/S() signature parsing."""
    _DTYPE_MAP = {
        "f32": "torch.float32", "f16": "torch.float16",
        "bf16": "torch.bfloat16", "f64": "torch.float64",
        "i64": "torch.int64", "i32": "torch.int32",
        "i16": "torch.int16", "i8": "torch.int8",
        "b8": "torch.bool", "u8": "torch.uint8",
    }

    def Index(high, low=0):
        return {"kind": "index", "low": int(low), "high": int(high)}

    def Perm(size=None):
        return {"kind": "permutation", "size": None if size is None else int(size)}

    def T(shape, dtype, stride=None, max=None, gen=None, constraint=None, index=None):
        spec = {
            "kind": "tensor",
            "shape": shape,
            "dtype": _DTYPE_MAP.get(dtype, f"torch.{dtype}"),
            "stride": list(stride) if stride else None,
            "device": "cuda",
        }
        if constraint is None:
            constraint = index
        if gen is not None:
            spec["gen"] = gen
        elif constraint == "permutation":
            spec["gen"] = Perm(max)
        elif max is not None:
            spec["gen"] = Index(max)
        if constraint is not None:
            spec["constraint"] = constraint
        if max is not None:
            spec["max_val"] = int(max)
        return spec

    def S(dims):
        return {"kind": "shape", "dims": dims}

    return {"__builtins__": {}, "T": T, "S": S, "Index": Index, "Perm": Perm,
            "f32": "f32", "f16": "f16", "bf16": "bf16", "f64": "f64",
            "i64": "i64", "i32": "i32", "i16": "i16", "i8": "i8",
            "b8": "b8", "u8": "u8"}


def _eval_signature(expr: str) -> list:
    """Eval a T()/S() signature string and return a list of input specs."""
    ns = _make_shape_eval_ns()
    inputs = eval(expr, ns)  # noqa: S307
    if isinstance(inputs, tuple):
        inputs = list(inputs)
    elif isinstance(inputs, dict):
        inputs = [inputs]
    return inputs if inputs else []


def _parse_shapes_txt(shapes_path: Path) -> dict:
    """Parse shapes.txt by eval'ing each line with T() and S() as constructors."""
    _eval_ns = _make_shape_eval_ns()

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

    def Index(high, low=0):
        return {"kind": "index", "low": int(low), "high": int(high)}

    def Perm(size=None):
        return {"kind": "permutation", "size": None if size is None else int(size)}

    def T(shape, dtype, stride=None, max=None, gen=None, constraint=None, index=None):
        spec = {
            "kind": "tensor",
            "shape": shape,
            "dtype": _DTYPE_MAP.get(dtype, f"torch.{dtype}"),
            "stride": list(stride) if stride else None,
            "device": "cuda",
        }
        if constraint is None:
            constraint = index
        if gen is not None:
            spec["gen"] = gen
        elif constraint == "permutation":
            spec["gen"] = Perm(max)
        elif max is not None:
            spec["gen"] = Index(max)
        if constraint is not None:
            spec["constraint"] = constraint
        if max is not None:
            spec["max_val"] = int(max)
        return spec

    def S(dims):
        return {"kind": "shape", "dims": dims}

    _ns = {"__builtins__": {}, "T": T, "S": S, "Index": Index, "Perm": Perm,
            "torch": torch,
            "f32": "f32", "f16": "f16", "bf16": "bf16", "f64": "f64",
            "i64": "i64", "i32": "i32", "i16": "i16", "i8": "i8",
            "b8": "b8", "u8": "u8"}
    inputs = eval(config_str, _ns)
    if isinstance(inputs, dict):
        inputs = [inputs]
    elif isinstance(inputs, tuple):
        inputs = list(inputs)
    if not inputs and config_str.strip() not in ("()", ""):
        raise ValueError(f"parse_shapes_config produced empty result for non-empty config: {config_str[:80]}")
    return make_inputs_from_config({"inputs": inputs})


def _storage_size_for_strided(shape: list[int], stride: list[int]) -> int:
    """Compute minimum storage size needed for a strided tensor."""
    if not shape:
        return 1
    return sum((s - 1) * st for s, st in zip(shape, stride) if s > 1) + 1


def _numel(shape: list[int]) -> int:
    """Compute number of elements from a shape."""
    result = 1
    for dim in shape:
        result *= dim
    return result


def _generation_spec(spec: dict) -> dict | None:
    """Extract generation spec from a tensor spec dict.

    Handles both new-style 'gen' field and legacy 'max_val' field.
    """
    gen = spec.get("gen")
    if gen is not None:
        return gen

    if spec.get("constraint") == "permutation":
        return {"kind": "permutation", "size": spec.get("max_val")}
    if spec.get("max_val") is not None:
        return {"kind": "index", "low": 0, "high": int(spec["max_val"])}
    return None


def _make_permutation_tensor(shape, dtype, device, stride=None, size=None):
    """Create a tensor containing random permutations of [0, size).

    If `size` is smaller than the tensor numel and divides it evenly, generate
    independent row-wise permutations. This covers inverse-permutation index
    tensors with shape like [batch, heads, N] and gen=Perm(N).
    """
    logical_numel = _numel(shape)
    size = int(size if size is not None else logical_numel)
    if size == logical_numel:
        values = torch.randperm(size, dtype=dtype, device=device).reshape(shape)
    elif logical_numel % size == 0:
        rows = logical_numel // size
        values = torch.stack(
            [torch.randperm(size, dtype=dtype, device=device) for _ in range(rows)],
            dim=0,
        ).reshape(shape)
    else:
        raise ValueError(
            "permutation generator needs size == numel or size to divide numel, "
            f"got size={size}, numel={logical_numel}"
        )
    if stride:
        out = torch.empty_strided(shape, stride, dtype=dtype, device=device)
        out.copy_(values)
        return out
    return values


def make_inputs_from_config(config: dict) -> list:
    """Create inputs from a shape config. Returns mix of tensors and shape-param lists.

    Alias groups: specs carrying "alias_group" are views of ONE storage
    (packed-qkv saved views). One buffer is allocated per group — sized by
    config["alias_group_nbytes"][g], the true capture-time allocation —
    and each member is as_strided at its own (shape, stride, offset).
    Footprint and locality then match the model instead of giving every
    view a private storage.
    """
    group_nbytes = config.get("alias_group_nbytes") or []
    group_storage: dict[int, torch.Tensor] = {}

    def _group_buffer(g: int, device) -> torch.Tensor:
        if g not in group_storage:
            nbytes = group_nbytes[g] if g < len(group_nbytes) else 0
            if nbytes <= 0:
                raise ValueError(
                    f"alias_group {g} has no recorded nbytes "
                    f"(alias_group_nbytes={group_nbytes})")
            buf = torch.empty(nbytes, dtype=torch.uint8, device=device)
            buf.random_(0, 255)  # bit-pattern noise; views reinterpret dtype
            group_storage[g] = buf
        return group_storage[g]

    def _contiguous_stride_local(shape):
        st = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            st[i] = st[i + 1] * max(int(shape[i + 1]), 1)
        return st

    result = []
    for spec in config["inputs"]:
        if spec.get("kind") == "shape":
            result.append(spec["dims"])
            continue

        if spec.get("alias_group") is not None:
            shape = spec["shape"]
            dtype = getattr(torch, spec["dtype"].replace("torch.", ""))
            stride = spec.get("stride") or _contiguous_stride_local(shape)
            device = spec.get("device", "cuda")
            offset = int(spec.get("storage_offset", 0))
            buf = _group_buffer(spec["alias_group"], device)
            typed = buf.view(dtype) if dtype != torch.uint8 else buf
            result.append(typed.as_strided(shape, stride, offset))
            continue

        shape = spec["shape"]
        dtype_str = spec["dtype"].replace("torch.", "")
        dtype = getattr(torch, dtype_str)
        stride = spec.get("stride")
        device = spec.get("device", "cuda")
        gen = _generation_spec(spec)

        if dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
            if gen and gen.get("kind") == "constant":
                # Single safe value (maxpool window-center offsets etc).
                if stride:
                    numel = _storage_size_for_strided(shape, stride)
                    storage = torch.full((numel,), int(gen.get("value", 0)),
                                         dtype=dtype, device=device)
                    result.append(storage.as_strided(shape, stride))
                else:
                    result.append(torch.full(shape, int(gen.get("value", 0)),
                                             dtype=dtype, device=device))
                continue
            if gen and gen.get("kind") == "permutation":
                result.append(
                    _make_permutation_tensor(
                        shape,
                        dtype=dtype,
                        device=device,
                        stride=stride,
                        size=gen.get("size"),
                    )
                )
                continue

            if gen and gen.get("kind") == "offsets":
                # Segment offsets (e.g. _embedding_bag): non-decreasing,
                # first element 0, bounded by high (= len(indices)).
                hi = max(int(gen.get("high", 1)), 1)
                vals, _ = torch.sort(
                    torch.randint(0, hi, shape, dtype=dtype, device=device))
                flat = vals.reshape(-1)
                if flat.numel():
                    flat[0] = 0
                result.append(vals)
                continue

            low = int(gen.get("low", 0)) if gen else 0
            hi = int(gen.get("high", 100)) if gen else 100
            if hi <= low:
                hi = low + 1
            if stride:
                numel = _storage_size_for_strided(shape, stride)
                storage = torch.randint(low, hi, (numel,), dtype=dtype, device=device)
                result.append(storage.as_strided(shape, stride))
            else:
                result.append(torch.randint(low, hi, shape, dtype=dtype, device=device))
        elif dtype == torch.bool:
            if stride:
                numel = _storage_size_for_strided(shape, stride)
                storage = torch.randint(0, 2, (numel,), dtype=torch.bool, device=device)
                result.append(storage.as_strided(shape, stride))
            else:
                result.append(torch.randint(0, 2, shape, dtype=torch.bool, device=device))
        else:
            if gen and gen.get("kind") == "index":
                low = int(gen.get("low", 0))
                hi = int(gen.get("high", 100))
                if hi <= low:
                    hi = low + 1
                if stride:
                    numel = _storage_size_for_strided(shape, stride)
                    storage = torch.randint(low, hi, (numel,), dtype=torch.int64, device=device)
                    result.append(storage.to(dtype).as_strided(shape, stride))
                else:
                    result.append(torch.randint(low, hi, shape, dtype=torch.int64, device=device).to(dtype))
            elif stride:
                numel = _storage_size_for_strided(shape, stride)
                storage = torch.randn(numel, dtype=dtype, device=device)
                result.append(storage.as_strided(shape, stride))
            else:
                result.append(torch.randn(shape, dtype=dtype, device=device))
    return result


def _randn_with_bool(old_randn):
    """Wrapper around torch.randn that handles bool dtype (legacy repro compat)."""
    def randn(*args, **kwargs):
        if kwargs.get("dtype") is torch.bool:
            kwargs = dict(kwargs)
            kwargs.pop("dtype")
            if "size" in kwargs:
                size = kwargs.pop("size")
            elif len(args) == 1 and isinstance(args[0], (list, tuple, torch.Size)):
                size = tuple(args[0])
            else:
                size = args
            return torch.randint(0, 2, size, dtype=torch.bool, **kwargs)
        return old_randn(*args, **kwargs)

    return randn


def make_inputs_safely(make_inputs_fn, *args, **kwargs):
    """Run generated make_inputs while tolerating legacy bool randn repros."""
    old_randn = torch.randn
    torch.randn = _randn_with_bool(old_randn)
    try:
        return make_inputs_fn(*args, **kwargs)
    finally:
        torch.randn = old_randn


def count_bytes_naive(inputs, outputs) -> int:
    from byte_accounting import count_bytes_naive as _count_bytes_naive

    return _count_bytes_naive(inputs, outputs)


def count_bytes_adjusted(mod, inputs) -> int:
    from byte_accounting import count_bytes_effective

    return count_bytes_effective(mod, inputs)


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
    parser.add_argument("--n-rep", type=int, default=100)
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
                default_inputs = make_inputs_safely(make_inputs_fn)
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
                inputs = make_inputs_safely(make_inputs_fn)
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
            sol_ms = do_bench(lambda: torch.add(src, 1, out=dst), warmup=parsed.n_warmup, rep=parsed.n_rep, return_mode="min")
            sol_us = sol_ms * 1000
            del src, dst

            # Compiled (default heuristics) with CUDAGraph replay
            torch._dynamo.reset()
            compiled = torch.compile(mod)
            with torch.no_grad():
                for _ in range(3):
                    compiled(*inputs)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    compiled(*inputs)
                torch.cuda.synchronize()
            compiled_ms = do_bench(lambda: g.replay(), warmup=parsed.n_warmup, rep=parsed.n_rep, return_mode="min")
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
                    g_cd = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g_cd):
                        compiled_cd(*inputs)
                    torch.cuda.synchronize()
                cd_ms = do_bench(lambda: g_cd.replay(), warmup=parsed.n_warmup, rep=parsed.n_rep, return_mode="min")
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
