"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full `arg0_1 > 0`, metadata-only flatten view, and scalar `aten.any` reduction as one single-program Triton reduction, whereas Inductor currently emits a generic fused scalar-reduction template with an artificial size-1 X dimension, a reduction loop, a boolean accumulator tensor, and the helper `any` epilogue; Inductor cannot do this today because its scheduler/codegen lacks a zero-dimensional any/all specialization for small flattened reductions whose only producer is an inlineable predicate; the fix is SCHEDULER_FUSION: teach Inductor's reduction scheduler to lower such scalar any/all patterns to a compact one-block predicate reduction without the generic X/R loop scaffold."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - keeps py_compile useful.
    triton = None
    tl = None

from repro_harness import load_shape_configs, make_inputs_from_config


REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"
REPRO_ID = REPRO_DIR.name
HISTORICAL_BEST_COMPILE_US = 5.7


def _load_repro_module() -> Any:
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shape_product(shape: Any) -> int:
    if isinstance(shape, int):
        return int(shape)
    result = 1
    for dim in shape:
        result *= int(dim)
    return result


def _default_inputs() -> list[Any]:
    module = _load_repro_module()
    return module.make_inputs()


def _named_inputs(shape_name: str | None) -> tuple[str, list[Any]]:
    if shape_name is None:
        return "default", _default_inputs()

    configs = load_shape_configs(str(REPRO_PATH))
    if shape_name not in configs:
        available = ", ".join(configs)
        raise ValueError(f"unknown shape {shape_name!r}; available shapes: {available}")
    return shape_name, make_inputs_from_config(configs[shape_name])


if triton is not None:

    @triton.jit
    def _gt_zero_any_kernel(
        x_ptr,
        out_ptr,
        n_elements: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_elements
        values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        flags = (values > 0.0).to(tl.int32)
        any_positive = tl.max(flags, axis=0)
        tl.store(out_ptr, any_positive != 0)


def oracle_forward(inputs: tuple[Any, ...] | list[Any]) -> torch.Tensor:
    """Compute exactly Repro()(*make_inputs()) for this captured graph."""
    if triton is None:
        raise RuntimeError("triton is required for oracle_any_reduce.py")
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects two inputs, got {len(inputs)}")

    arg0_1, shape_param = inputs
    if not isinstance(arg0_1, torch.Tensor):
        raise TypeError(f"{REPRO_ID} first input must be a tensor")
    if arg0_1.dtype not in (torch.float16, torch.float32):
        raise ValueError(f"{REPRO_ID} expects f16 or f32 input, got {arg0_1.dtype}")
    if not arg0_1.is_cuda:
        raise ValueError(f"{REPRO_ID} expects CUDA inputs")
    if not arg0_1.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects the captured contiguous input layout")
    if _shape_product(shape_param) != arg0_1.numel():
        raise ValueError(
            f"{REPRO_ID} shape parameter {shape_param!r} does not match "
            f"input numel {arg0_1.numel()}"
        )

    out = torch.empty((), device=arg0_1.device, dtype=torch.bool)
    block = triton.next_power_of_2(arg0_1.numel())
    _gt_zero_any_kernel[(1,)](
        arg0_1,
        out,
        n_elements=arg0_1.numel(),
        BLOCK=block,
        num_warps=16,
        num_stages=1,
    )
    return out


def _repro_forward(inputs: list[Any]) -> torch.Tensor:
    module = _load_repro_module()
    model = module.Repro()
    with torch.no_grad():
        return model(*inputs)


def _check_one(shape_name: str | None) -> bool:
    label, inputs = _named_inputs(shape_name)
    with torch.no_grad():
        eager = _repro_forward(inputs)
        oracle = oracle_forward(inputs)
    if eager.shape != oracle.shape or eager.dtype != oracle.dtype:
        print(
            f"{label}: FAIL shape/dtype eager={tuple(eager.shape)} {eager.dtype} "
            f"oracle={tuple(oracle.shape)} {oracle.dtype}"
        )
        return False
    ok = torch.equal(eager, oracle)
    print(f"{label}: {'PASS' if ok else 'FAIL'} value eager={bool(eager.item())} oracle={bool(oracle.item())}")
    return ok


def _do_bench(fn: Any, warmup: int, rep: int) -> float:
    if triton is not None:
        from triton.testing import do_bench

        return do_bench(fn, warmup=warmup, rep=rep, return_mode="min") * 1000.0

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = math.inf
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) * 1000.0)
    return best


def _compile_repro(inputs: list[Any], warmup: int) -> Any:
    import torch._inductor.config as inductor_config

    module = _load_repro_module()
    inductor_config.coordinate_descent_tuning = True
    compiled = torch.compile(module.Repro())
    with torch.no_grad():
        for _ in range(max(1, warmup)):
            compiled(*inputs)
    torch.cuda.synchronize()
    return compiled


def _bench_one(shape_name: str | None, warmup: int, rep: int) -> dict[str, Any]:
    label, inputs = _named_inputs(shape_name)
    with torch.no_grad():
        oracle_forward(inputs)
    torch.cuda.synchronize()
    oracle_us = _do_bench(lambda: oracle_forward(inputs), warmup=warmup, rep=rep)

    compiled = _compile_repro(inputs, warmup)
    compile_us = _do_bench(lambda: compiled(*inputs), warmup=warmup, rep=rep)
    best_compare_us = min(compile_us, HISTORICAL_BEST_COMPILE_US)
    result = {
        "repro_id": REPRO_ID,
        "shape": label,
        "oracle_us": round(oracle_us, 3),
        "compile_us": round(compile_us, 3),
        "historical_best_compile_us": HISTORICAL_BEST_COMPILE_US,
        "best_compare_us": round(best_compare_us, 3),
        "ratio": round(best_compare_us / oracle_us, 3) if oracle_us > 0 else None,
    }
    print(json.dumps(result))
    return result


def _shape_names(all_shapes: bool, shape: str | None) -> list[str | None]:
    if shape is not None:
        return [shape]
    if not all_shapes:
        return [None]
    return list(load_shape_configs(str(REPRO_PATH)).keys())


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Oracle for {REPRO_ID}")
    parser.add_argument("--check", action="store_true", help="verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="benchmark the full oracle against torch.compile")
    parser.add_argument("--shape", default=None, help="shape name from shapes.txt")
    parser.add_argument("--all-shapes", action="store_true", help="run all named shapes from shapes.txt")
    parser.add_argument("--warmup", type=int, default=25, help="benchmark warmup iterations")
    parser.add_argument("--rep", type=int, default=200, help="benchmark repetitions")
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = args.bench = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Triton oracle")

    names = _shape_names(args.all_shapes, args.shape)
    if args.check:
        ok = True
        for name in names:
            ok = _check_one(name) and ok
        if not ok:
            raise SystemExit(1)
    if args.bench:
        for name in names:
            _bench_one(name, args.warmup, args.rep)


if __name__ == "__main__":
    main()
