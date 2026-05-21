#!/usr/bin/env python3
"""Validate and repair compact Index(...) bounds for repro shape configs.

The normal random input path can miss an invalid index bound. This tool stress
fills generated index tensors with their current maximum value and runs the
repro in eager mode. If that fails with an index out-of-bounds or CUDA
device-assert class error, it probes each generated index input and suggests a
smaller exclusive Index() high. With --write, only the affected compact shape
line or _shapes_config string is rewritten.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


_DTYPE_TO_TORCH = {
    "f32": "torch.float32",
    "f16": "torch.float16",
    "bf16": "torch.bfloat16",
    "f64": "torch.float64",
    "i64": "torch.int64",
    "i32": "torch.int32",
    "i16": "torch.int16",
    "i8": "torch.int8",
    "b8": "torch.bool",
    "u8": "torch.uint8",
}

_TORCH_TO_DTYPE = {value: key for key, value in _DTYPE_TO_TORCH.items()}


@dataclass(frozen=True)
class ShapePoint:
    repro: Path
    label: str
    expr: str
    source: str
    line_no: int | None = None


def _index_gen(high: int, low: int = 0) -> dict[str, Any]:
    return {"kind": "index", "low": int(low), "high": int(high)}


def _perm_gen(size: int | None = None) -> dict[str, Any]:
    return {"kind": "permutation", "size": None if size is None else int(size)}


def _tensor_spec(
    shape,
    dtype,
    stride=None,
    max=None,
    gen=None,
    constraint=None,
    index=None,
):
    if constraint is None:
        constraint = index
    if isinstance(shape, int):
        shape = [shape]
    spec = {
        "kind": "tensor",
        "shape": list(shape),
        "dtype": _DTYPE_TO_TORCH.get(dtype, f"torch.{dtype}"),
        "stride": list(stride) if stride is not None else None,
        "device": "cuda",
    }
    if gen is not None:
        spec["gen"] = gen
    elif constraint == "permutation":
        spec["gen"] = _perm_gen(max)
    elif max is not None:
        spec["gen"] = _index_gen(max)
    if constraint is not None:
        spec["constraint"] = constraint
    if max is not None:
        spec["max_val"] = int(max)
    return spec


def _shape_spec(dims):
    if isinstance(dims, int):
        dims = [dims]
    return {"kind": "shape", "dims": list(dims)}


def parse_compact_config(expr: str) -> list[dict[str, Any]]:
    """Parse one compact T()/S() shape expression into input specs."""
    ns = {
        "__builtins__": {},
        "T": _tensor_spec,
        "S": _shape_spec,
        "Index": _index_gen,
        "Perm": _perm_gen,
        **{name: name for name in _DTYPE_TO_TORCH},
    }
    value = eval(expr, ns)
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and value[1] == {}
        and isinstance(value[0], (tuple, list, dict))
    ):
        value = value[0]
    if isinstance(value, dict):
        return [value]
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    raise ValueError(f"unsupported compact config value: {type(value).__name__}")


def _generation_spec(spec: dict[str, Any]) -> dict[str, Any] | None:
    gen = spec.get("gen")
    if gen is not None:
        return gen
    if spec.get("constraint") == "permutation":
        return {"kind": "permutation", "size": spec.get("max_val")}
    if spec.get("max_val") is not None:
        return _index_gen(spec["max_val"])
    return None


def _generation_expr(spec: dict[str, Any]) -> str | None:
    gen = _generation_spec(spec)
    if gen is None:
        return None
    if gen.get("kind") == "permutation":
        size = gen.get("size")
        return "Perm()" if size is None else f"Perm({int(size)})"
    if gen.get("kind") == "index":
        low = int(gen.get("low", 0))
        high = int(gen["high"])
        if low == 0:
            return f"Index({high})"
        return f"Index({high}, low={low})"
    return None


def format_compact_config(specs: list[dict[str, Any]]) -> str:
    parts = []
    for spec in specs:
        if spec.get("kind") == "shape":
            parts.append(f"S({spec['dims']})")
            continue

        dtype = _TORCH_TO_DTYPE.get(spec["dtype"], spec["dtype"].replace("torch.", ""))
        kwargs = []
        if spec.get("stride") is not None:
            kwargs.append(f"stride={tuple(spec['stride'])}")
        gen = _generation_expr(spec)
        if gen is not None:
            kwargs.append(f"gen={gen}")
        suffix = f", {', '.join(kwargs)}" if kwargs else ""
        parts.append(f"T({spec['shape']}, {dtype}{suffix})")
    return f"({', '.join(parts)})"


def _default_shapes_config(repro: Path) -> str | None:
    match = re.search(
        r'^_shapes_config\s*=\s*"(.+)"',
        repro.read_text(),
        flags=re.MULTILINE,
    )
    return match.group(1) if match else None


def discover_shape_points(
    repro: Path,
    *,
    all_shapes: bool = False,
    shape_label: str | None = None,
) -> list[ShapePoint]:
    """Return the exact repro/shape points requested by the caller."""
    repro = repro if repro.name == "repro.py" else repro / "repro.py"
    if not repro.exists():
        raise FileNotFoundError(repro)

    points: list[ShapePoint] = []
    default_expr = _default_shapes_config(repro)
    if shape_label in (None, "default") and default_expr is not None:
        points.append(ShapePoint(repro, "default", default_expr, "default"))

    shapes_txt = repro.parent / "shapes.txt"
    if shapes_txt.exists() and (
        (all_shapes and shape_label is None)
        or shape_label not in (None, "default")
    ):
        for line_no, line in enumerate(shapes_txt.read_text().splitlines(), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or ":" not in stripped:
                continue
            label, expr = stripped.split(":", 1)
            label = label.strip()
            if shape_label is not None and label != shape_label:
                continue
            points.append(
                ShapePoint(repro, label, expr.strip(), "shapes.txt", line_no)
            )

    if shape_label is not None and not points:
        raise ValueError(f"{repro}: shape label {shape_label!r} not found")
    if not points:
        raise ValueError(f"{repro}: no shape configs found")
    return points


def _find_repros(paths: list[Path]) -> list[Path]:
    repros: list[Path] = []
    for path in paths:
        if path.is_file():
            repros.append(path)
        elif (path / "repro.py").exists():
            repros.append(path / "repro.py")
        else:
            repros.extend(path.glob("*/repro.py"))
    return sorted(set(repros))


def _index_inputs(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for idx, spec in enumerate(specs):
        if spec.get("kind") != "tensor":
            continue
        gen = _generation_spec(spec)
        if gen is None or gen.get("kind") != "index":
            continue
        result.append({
            "input_index": idx,
            "shape": spec.get("shape"),
            "dtype": spec.get("dtype"),
            "low": int(gen.get("low", 0)),
            "high": int(gen["high"]),
        })
    return result


def _specs_for_device(specs: list[dict[str, Any]], device: str) -> list[dict[str, Any]]:
    copied = copy.deepcopy(specs)
    for spec in copied:
        if spec.get("kind") == "tensor":
            spec["device"] = device
    return copied


def _with_high_overrides(
    specs: list[dict[str, Any]], high_overrides: dict[int, int]
) -> list[dict[str, Any]]:
    copied = copy.deepcopy(specs)
    for idx, high in high_overrides.items():
        spec = copied[idx]
        gen = _generation_spec(spec)
        if gen is None or gen.get("kind") != "index":
            continue
        gen["high"] = int(high)
        spec["gen"] = gen
        if "max_val" in spec:
            spec["max_val"] = int(high)
    return copied


def _fill_for_all_highs(specs: list[dict[str, Any]]) -> dict[int, int]:
    fill: dict[int, int] = {}
    for item in _index_inputs(specs):
        fill[item["input_index"]] = item["high"] - 1
    return fill


def _fill_for_all_lows(specs: list[dict[str, Any]]) -> dict[int, int]:
    fill: dict[int, int] = {}
    for item in _index_inputs(specs):
        fill[item["input_index"]] = item["low"]
    return fill


def _fill_for_one_high(specs: list[dict[str, Any]], input_index: int, high: int) -> dict[int, int]:
    fill = _fill_for_all_lows(specs)
    fill[input_index] = int(high) - 1
    return fill


def _load_repro_module(repro: Path):
    name = f"bound_repro_{os.getpid()}_{abs(hash((str(repro), time.time_ns())))}"
    sys.path.insert(0, str(repro.parent))
    spec = importlib.util.spec_from_file_location(name, repro)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {repro}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_inputs(specs: list[dict[str, Any]], fill_values: dict[int, int]):
    import torch
    from repro_harness import make_inputs_from_config

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    inputs = make_inputs_from_config({"inputs": specs})
    for idx, fill in fill_values.items():
        spec = specs[int(idx)]
        dtype_name = spec["dtype"].replace("torch.", "")
        dtype = getattr(torch, dtype_name)
        shape = spec["shape"]
        stride = spec.get("stride")
        device = spec.get("device", "cuda")
        if stride is not None:
            tensor = torch.empty_strided(shape, stride, dtype=dtype, device=device)
            tensor.fill_(int(fill))
        else:
            tensor = torch.full(shape, int(fill), dtype=dtype, device=device)
        inputs[int(idx)] = tensor
    return inputs


def _run_specs_inprocess(
    repro: Path,
    specs: list[dict[str, Any]],
    *,
    fill_values: dict[int, int],
) -> dict[str, Any]:
    import torch

    start = time.time()
    result: dict[str, Any] = {
        "ok": False,
        "error": None,
        "traceback": None,
        "elapsed_s": None,
    }
    try:
        module = _load_repro_module(repro)
        model = module.Repro()
        inputs = _make_inputs(specs, fill_values)
        with torch.no_grad():
            out = model(*inputs)
            if torch.cuda.is_available() and any(
                spec.get("device", "").startswith("cuda")
                for spec in specs
                if spec.get("kind") == "tensor"
            ):
                torch.cuda.synchronize()
        del out, inputs, model
        result["ok"] = True
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc(limit=8)
    finally:
        result["elapsed_s"] = time.time() - start
    return result


def _run_child(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.child_run).read_text())
    result = _run_specs_inprocess(
        Path(payload["repro"]),
        payload["specs"],
        fill_values={int(k): int(v) for k, v in payload["fill_values"].items()},
    )
    Path(args.child_output).write_text(json.dumps(result))
    return 0 if result["ok"] else 1


def _run_specs_subprocess(
    repro: Path,
    specs: list[dict[str, Any]],
    *,
    fill_values: dict[int, int],
    timeout_s: int,
    cuda_visible_devices: str | None = None,
) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(prefix="bound_run_", suffix=".json", delete=False) as tmp:
        payload_path = Path(tmp.name)
    with tempfile.NamedTemporaryFile(prefix="bound_result_", suffix=".json", delete=False) as tmp:
        result_path = Path(tmp.name)

    payload_path.write_text(json.dumps({
        "repro": str(repro),
        "specs": specs,
        "fill_values": fill_values,
    }))

    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    cmd = [
        sys.executable,
        __file__,
        "--child-run",
        str(payload_path),
        "--child-output",
        str(result_path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
        if result_path.exists() and result_path.stat().st_size:
            result = json.loads(result_path.read_text())
        else:
            result = {
                "ok": False,
                "error": f"child exited {proc.returncode} without result",
                "traceback": None,
            }
        result["returncode"] = proc.returncode
        if proc.stdout:
            result["output_tail"] = proc.stdout[-4000:]
        return result
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "error": f"timeout after {timeout_s}s",
            "traceback": None,
            "output_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
        }
    finally:
        for path in (payload_path, result_path):
            try:
                path.unlink()
            except OSError:
                pass


def _run_specs(
    repro: Path,
    specs: list[dict[str, Any]],
    *,
    fill_values: dict[int, int],
    isolate: bool,
    timeout_s: int,
    cuda_visible_devices: str | None,
) -> dict[str, Any]:
    if isolate:
        return _run_specs_subprocess(
            repro,
            specs,
            fill_values=fill_values,
            timeout_s=timeout_s,
            cuda_visible_devices=cuda_visible_devices,
        )
    return _run_specs_inprocess(repro, specs, fill_values=fill_values)


def is_index_bounds_error(result: dict[str, Any]) -> bool:
    text = "\n".join(
        str(result.get(key) or "")
        for key in ("error", "traceback", "output_tail")
    ).lower()
    needles = (
        "index out of range",
        "out of bounds",
        "out-of-bounds",
        "device-side assert",
        "device side assert",
        "srcindex < srcselectdimsize",
        "indexselectlargeindex",
        "indexselectsmallindex",
    )
    return any(needle in text for needle in needles)


def _candidate_passes(
    repro: Path,
    specs: list[dict[str, Any]],
    *,
    input_index: int,
    high: int,
    isolate: bool,
    timeout_s: int,
    cuda_visible_devices: str | None,
) -> tuple[bool, dict[str, Any]]:
    result = _run_specs(
        repro,
        specs,
        fill_values=_fill_for_one_high(specs, input_index, high),
        isolate=isolate,
        timeout_s=timeout_s,
        cuda_visible_devices=cuda_visible_devices,
    )
    return bool(result["ok"]), result


def _infer_repair_for_input(
    repro: Path,
    specs: list[dict[str, Any]],
    item: dict[str, Any],
    *,
    isolate: bool,
    timeout_s: int,
    cuda_visible_devices: str | None,
) -> dict[str, Any] | None:
    input_index = int(item["input_index"])
    low = int(item["low"])
    old_high = int(item["high"])
    if old_high <= low + 1:
        return None

    passes, result = _candidate_passes(
        repro,
        specs,
        input_index=input_index,
        high=old_high,
        isolate=isolate,
        timeout_s=timeout_s,
        cuda_visible_devices=cuda_visible_devices,
    )
    if passes or not is_index_bounds_error(result):
        return None

    lo = low + 1
    hi = old_high
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        passes, _ = _candidate_passes(
            repro,
            specs,
            input_index=input_index,
            high=mid,
            isolate=isolate,
            timeout_s=timeout_s,
            cuda_visible_devices=cuda_visible_devices,
        )
        if passes:
            lo = mid
        else:
            hi = mid

    return {
        "input_index": input_index,
        "old_high": old_high,
        "new_high": lo,
        "low": low,
        "shape": item["shape"],
        "dtype": item["dtype"],
    }


def analyze_shape_point(
    point: ShapePoint,
    *,
    device: str,
    repair: bool = True,
    isolate: bool = False,
    timeout_s: int = 120,
    cuda_visible_devices: str | None = None,
) -> dict[str, Any]:
    specs = _specs_for_device(parse_compact_config(point.expr), device)
    index_inputs = _index_inputs(specs)
    result = _run_specs(
        point.repro,
        specs,
        fill_values=_fill_for_all_highs(specs),
        isolate=isolate,
        timeout_s=timeout_s,
        cuda_visible_devices=cuda_visible_devices,
    )

    analysis: dict[str, Any] = {
        "repro": str(point.repro),
        "shape": point.label,
        "source": point.source,
        "line_no": point.line_no,
        "original_ok": bool(result["ok"]),
        "error": result.get("error"),
        "failure_class": None,
        "index_inputs": index_inputs,
        "needs_bound_repair": False,
        "repairs": [],
        "repaired_ok": None,
        "updated_expr": None,
    }
    if result["ok"]:
        return analysis
    if not index_inputs:
        analysis["failure_class"] = "eager_failure"
        return analysis

    if not is_index_bounds_error(result):
        analysis["failure_class"] = "eager_failure"
        return analysis

    analysis["failure_class"] = "index_bounds"
    analysis["needs_bound_repair"] = True
    if not repair:
        return analysis

    base = _run_specs(
        point.repro,
        specs,
        fill_values=_fill_for_all_lows(specs),
        isolate=isolate,
        timeout_s=timeout_s,
        cuda_visible_devices=cuda_visible_devices,
    )
    if not base["ok"]:
        analysis["base_low_error"] = base.get("error")
        return analysis

    repairs = []
    for item in index_inputs:
        repair_item = _infer_repair_for_input(
            point.repro,
            specs,
            item,
            isolate=isolate,
            timeout_s=timeout_s,
            cuda_visible_devices=cuda_visible_devices,
        )
        if repair_item is not None:
            repairs.append(repair_item)

    analysis["repairs"] = repairs
    if not repairs:
        return analysis

    repaired_specs = _with_high_overrides(
        specs,
        {int(item["input_index"]): int(item["new_high"]) for item in repairs},
    )
    verify = _run_specs(
        point.repro,
        repaired_specs,
        fill_values=_fill_for_all_highs(repaired_specs),
        isolate=isolate,
        timeout_s=timeout_s,
        cuda_visible_devices=cuda_visible_devices,
    )
    analysis["repaired_ok"] = bool(verify["ok"])
    if verify["ok"]:
        analysis["updated_expr"] = format_compact_config(repaired_specs)
    else:
        analysis["repair_error"] = verify.get("error")
    return analysis


def _write_default_expr(repro: Path, updated_expr: str) -> None:
    content = repro.read_text()
    replacement = f"_shapes_config = {json.dumps(updated_expr)}"
    new_content, count = re.subn(
        r'^_shapes_config\s*=\s*".+"',
        replacement,
        content,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError(f"{repro}: could not update _shapes_config")
    repro.write_text(new_content)


def _write_shapes_txt_expr(repro: Path, line_no: int, updated_expr: str) -> None:
    path = repro.parent / "shapes.txt"
    lines = path.read_text().splitlines()
    idx = line_no - 1
    if idx < 0 or idx >= len(lines) or ":" not in lines[idx]:
        raise RuntimeError(f"{path}: could not update line {line_no}")
    prefix = lines[idx].split(":", 1)[0]
    lines[idx] = f"{prefix}: {updated_expr}"
    path.write_text("\n".join(lines) + "\n")


def write_repairs(analyses: list[dict[str, Any]]) -> int:
    updated = 0
    for analysis in analyses:
        updated_expr = analysis.get("updated_expr")
        if not updated_expr:
            continue
        repro = Path(analysis["repro"])
        if analysis["source"] == "default":
            _write_default_expr(repro, updated_expr)
        elif analysis["source"] == "shapes.txt":
            _write_shapes_txt_expr(repro, int(analysis["line_no"]), updated_expr)
        else:
            raise RuntimeError(f"unknown source {analysis['source']!r}")
        updated += 1
    return updated


def validate_repro(
    repro: Path,
    *,
    all_shapes: bool = False,
    shape_label: str | None = None,
    device: str = "cuda",
    repair: bool = True,
    write: bool = False,
    isolate: bool = False,
    timeout_s: int = 120,
    cuda_visible_devices: str | None = None,
) -> dict[str, Any]:
    points = discover_shape_points(repro, all_shapes=all_shapes, shape_label=shape_label)
    analyses = [
        analyze_shape_point(
            point,
            device=device,
            repair=repair,
            isolate=isolate,
            timeout_s=timeout_s,
            cuda_visible_devices=cuda_visible_devices,
        )
        for point in points
    ]
    updated = write_repairs(analyses) if write else 0
    return {
        "repro": str(repro),
        "points": len(points),
        "updated": updated,
        "results": analyses,
    }


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cuda"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate eager index bounds and optionally rewrite compact Index() highs"
    )
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--all-shapes", action="store_true")
    parser.add_argument("--shape", dest="shape_label")
    parser.add_argument("--device", default=None)
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--no-repair", dest="repair", action="store_false")
    parser.add_argument("--isolate", action="store_true", default=None)
    parser.add_argument("--no-isolate", dest="isolate", action="store_false")
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child-run")
    parser.add_argument("--child-output")
    args = parser.parse_args()

    if args.child_run:
        return _run_child(args)

    device = args.device or _default_device()
    isolate = args.isolate
    if isolate is None:
        isolate = device.startswith("cuda")

    repros = _find_repros(args.paths or [Path("repros/canonical")])
    all_results = []
    for repro in repros:
        summary = validate_repro(
            repro,
            all_shapes=args.all_shapes,
            shape_label=args.shape_label,
            device=device,
            repair=args.repair,
            write=args.write,
            isolate=isolate,
            timeout_s=args.timeout_s,
            cuda_visible_devices=args.cuda_visible_devices,
        )
        all_results.extend(summary["results"])

    failed = [r for r in all_results if not r["original_ok"]]
    repairable = [r for r in failed if r.get("updated_expr")]
    updated = sum(1 for r in all_results if r.get("updated_expr")) if args.write else 0

    for result in all_results:
        name = Path(result["repro"]).parent.name
        if result["original_ok"]:
            print(f"OK   {name} [{result['shape']}]")
            continue
        if result.get("updated_expr"):
            repairs = ", ".join(
                f"arg{r['input_index']}: Index({r['old_high']}) -> Index({r['new_high']})"
                for r in result["repairs"]
            )
            action = "UPDATED" if args.write else "REPAIR"
            print(f"{action} {name} [{result['shape']}] {repairs}")
        else:
            print(
                f"FAIL {name} [{result['shape']}] "
                f"{result.get('failure_class')}: {result.get('error')}"
            )

    summary = {
        "ok": len(all_results) - len(failed),
        "failed": len(failed),
        "repairable": len(repairable),
        "updated": updated,
        "results": all_results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2))
        print(f"[output] Wrote {args.output}")

    print(
        f"Done: {summary['ok']} ok, {summary['failed']} failed, "
        f"{summary['repairable']} repairable, {summary['updated']} updated"
    )
    unrepairable = len(failed) - len(repairable)
    if failed and not (args.write and unrepairable == 0 and updated == len(repairable)):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
