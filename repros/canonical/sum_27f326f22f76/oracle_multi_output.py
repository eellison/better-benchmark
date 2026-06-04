"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full Repro.forward result by fusing the broadcast multiply, square/subtract epilogue, materialized [1000,16] producer for the returned [16,1000] permute view, and sibling dim-0 sum into one Triton kernel, whereas Inductor currently schedules the shared pointwise producer and the reduction/layout consumers as generic fused pointwise plus reduction work instead of one multi-output producer/reduction kernel; Inductor cannot do this today because its scheduler does not fuse a materialized layout output and a sibling reduction over the same pointwise expression when the layout output must preserve a non-contiguous view stride; the fix is SCHEDULER_FUSION: teach Inductor to form a full-scope multi-output schedule that writes the required layout base while accumulating the compatible column reduction in the same generated kernel."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
    triton = None
    tl = None

from repro_harness import load_shape_configs, make_inputs_from_config


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"
HISTORICAL_BEST_COMPILE_US = 9.1

ROWS = 1000
COLS = 16
BLOCK_ROWS = 1024
BLOCK_COLS = 16


def _load_repro_module() -> Any:
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _default_inputs() -> list[Any]:
    return _load_repro_module().make_inputs()


def _named_inputs(shape_name: str | None) -> tuple[str, list[Any]]:
    if shape_name is None:
        return "default", _default_inputs()

    configs = load_shape_configs(str(REPRO_PATH))
    if shape_name not in configs:
        available = ", ".join(configs)
        raise ValueError(f"unknown shape {shape_name!r}; available shapes: {available}")
    return shape_name, make_inputs_from_config(configs[shape_name])


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _require_triton_cuda() -> None:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_multi_output.py")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for oracle_multi_output.py")


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _require_triton_cuda()
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects four inputs, got {len(inputs)}")

    arg4_1, arg9_1, arg8_1, shape_param = inputs
    tensors = (("arg4_1", arg4_1), ("arg9_1", arg9_1), ("arg8_1", arg8_1))
    for name, value in tensors:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
        if value.dtype != torch.float32:
            raise TypeError(f"{name} must be torch.float32, got {value.dtype}")
        if not value.is_cuda:
            raise ValueError(f"{name} must be on CUDA")

    if tuple(arg4_1.shape) != (1, COLS) or tuple(arg4_1.stride()) != (COLS, 1):
        raise ValueError(f"expected arg4_1 f32[1,{COLS}] with stride ({COLS},1)")
    if tuple(arg9_1.shape) != (ROWS, 1) or tuple(arg9_1.stride()) != (1, 1):
        raise ValueError(f"expected arg9_1 f32[{ROWS},1] with stride (1,1)")
    if tuple(arg8_1.shape) != (ROWS, COLS) or tuple(arg8_1.stride()) != (COLS, 1):
        raise ValueError(f"expected arg8_1 f32[{ROWS},{COLS}] with stride ({COLS},1)")
    if arg4_1.device != arg9_1.device or arg4_1.device != arg8_1.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")
    if _shape_tuple(shape_param) != (COLS,):
        raise ValueError(f"expected shape parameter [{COLS}], got {shape_param!r}")

    return arg4_1, arg9_1, arg8_1


if triton is not None:

    @triton.jit
    def _mul_sub_transpose_sum_kernel(
        arg4_ptr,
        arg9_ptr,
        arg8_ptr,
        val_ptr,
        sum_ptr,
        n_rows: tl.constexpr,
        n_cols: tl.constexpr,
        block_rows: tl.constexpr,
        block_cols: tl.constexpr,
    ):
        rows = tl.arange(0, block_rows)
        cols = tl.arange(0, block_cols)
        mask = (rows[:, None] < n_rows) & (cols[None, :] < n_cols)

        arg4 = tl.load(arg4_ptr + cols, mask=cols < n_cols, other=0.0).to(tl.float32)
        arg9 = tl.load(arg9_ptr + rows, mask=rows < n_rows, other=0.0).to(tl.float32)
        arg8 = tl.load(
            arg8_ptr + rows[:, None] * n_cols + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        mul = arg9[:, None] * arg4[None, :]
        sub = 1.0 - arg8 * arg8
        value = tl.where(mask, mul * sub, 0.0)

        tl.store(val_ptr + rows[:, None] * n_cols + cols[None, :], value, mask=mask)
        sums = tl.sum(value, axis=0)
        tl.store(sum_ptr + cols, sums, mask=cols < n_cols)


def oracle_forward(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exactly Repro()(*make_inputs()) for the captured full graph."""
    arg4_1, arg9_1, arg8_1 = _validate_inputs(inputs)

    val = torch.empty((ROWS, COLS), device=arg8_1.device, dtype=torch.float32)
    sum_out = torch.empty((COLS,), device=arg8_1.device, dtype=torch.float32)
    _mul_sub_transpose_sum_kernel[(1,)](
        arg4_1,
        arg9_1,
        arg8_1,
        val,
        sum_out,
        n_rows=ROWS,
        n_cols=COLS,
        block_rows=BLOCK_ROWS,
        block_cols=BLOCK_COLS,
        num_warps=8,
        num_stages=1,
    )
    return val.permute(1, 0), sum_out


def _repro_forward(inputs: list[Any]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro()
    with torch.no_grad():
        result = model(*inputs)
    return _normalize_outputs(result)


def _normalize_outputs(outputs: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(outputs, torch.Tensor):
        return (outputs,)
    if isinstance(outputs, (tuple, list)):
        flattened = []
        for output in outputs:
            if isinstance(output, torch.Tensor):
                flattened.append(output)
            elif isinstance(output, (tuple, list)):
                flattened.extend(_normalize_outputs(output))
        return tuple(flattened)
    return ()


def _check_one(shape_name: str | None, atol: float, rtol: float) -> bool:
    label, inputs = _named_inputs(shape_name)
    with torch.no_grad():
        eager_outputs = _repro_forward(inputs)
        oracle_outputs = _normalize_outputs(oracle_forward(inputs))

    if len(eager_outputs) != len(oracle_outputs):
        print(
            f"{label}: FAIL output count eager={len(eager_outputs)} "
            f"oracle={len(oracle_outputs)}"
        )
        return False

    ok_all = True
    for index, (eager, oracle) in enumerate(zip(eager_outputs, oracle_outputs)):
        metadata_ok = True
        if eager.shape != oracle.shape:
            print(
                f"{label}: output {index} FAIL shape eager={tuple(eager.shape)} "
                f"oracle={tuple(oracle.shape)}"
            )
            metadata_ok = False
        if eager.dtype != oracle.dtype:
            print(
                f"{label}: output {index} FAIL dtype eager={eager.dtype} "
                f"oracle={oracle.dtype}"
            )
            metadata_ok = False
        if eager.stride() != oracle.stride():
            print(
                f"{label}: output {index} FAIL stride eager={tuple(eager.stride())} "
                f"oracle={tuple(oracle.stride())}"
            )
            metadata_ok = False

        if not metadata_ok:
            ok_all = False
            continue

        if eager.is_floating_point():
            diff = (eager.float() - oracle.float()).abs()
            max_diff = diff.max().item() if diff.numel() else 0.0
            values_ok = torch.allclose(eager.float(), oracle.float(), atol=atol, rtol=rtol)
            print(
                f"{label}: output {index} {'PASS' if values_ok else 'FAIL'} "
                f"shape={list(eager.shape)} dtype={eager.dtype} "
                f"stride={tuple(eager.stride())} max_diff={max_diff:.3e}"
            )
        else:
            values_ok = torch.equal(eager, oracle)
            print(
                f"{label}: output {index} {'PASS' if values_ok else 'FAIL'} "
                f"shape={list(eager.shape)} dtype={eager.dtype} stride={tuple(eager.stride())}"
            )

        ok_all = values_ok and ok_all

    return ok_all


def _do_bench(fn: Any, warmup: int, rep: int) -> float:
    if triton is not None:
        from triton.testing import do_bench

        return do_bench(fn, warmup=warmup, rep=rep, return_mode="min") * 1000.0

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best_us = math.inf
    for _ in range(rep):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best_us = min(best_us, (time.perf_counter() - start) * 1_000_000.0)
    return best_us


def _compile_repro(inputs: list[Any], warmup: int) -> Any:
    import torch._inductor.config as inductor_config

    inductor_config.coordinate_descent_tuning = True
    module = _load_repro_module()
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
    best_compile_us = min(compile_us, HISTORICAL_BEST_COMPILE_US)
    result = {
        "repro_id": REPRO_ID,
        "shape": label,
        "oracle_us": round(oracle_us, 3),
        "compile_us": round(compile_us, 3),
        "historical_best_compile_us": HISTORICAL_BEST_COMPILE_US,
        "best_compile_us": round(best_compile_us, 3),
        "ratio": round(best_compile_us / oracle_us, 3) if oracle_us > 0 else None,
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
    parser = argparse.ArgumentParser(description=f"Full-scope oracle for {REPRO_ID}")
    parser.add_argument("--check", action="store_true", help="verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="benchmark the Triton oracle")
    parser.add_argument("--shape", default=None, help="shape name from shapes.txt")
    parser.add_argument("--all-shapes", action="store_true", help="run all named shapes from shapes.txt")
    parser.add_argument("--atol", type=float, default=1e-4, help="absolute tolerance for checks")
    parser.add_argument("--rtol", type=float, default=1e-4, help="relative tolerance for checks")
    parser.add_argument("--warmup", type=int, default=25, help="benchmark warmup iterations")
    parser.add_argument("--rep", type=int, default=200, help="benchmark repetitions")
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = args.bench = True

    _require_triton_cuda()
    names = _shape_names(args.all_shapes, args.shape)

    if args.check:
        ok = True
        for name in names:
            ok = _check_one(name, args.atol, args.rtol) and ok
        if not ok:
            raise SystemExit(1)

    if args.bench:
        for name in names:
            _bench_one(name, args.warmup, args.rep)


if __name__ == "__main__":
    main()
