"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full Repro.forward result by fusing the broadcast multiply, square/subtract epilogue, materialized [1000,16] producer for the returned [16,1000] permute view, and sibling dim-0 sum into one Triton kernel, whereas Inductor currently schedules the shared pointwise producer and the reduction/layout consumers as generic fused pointwise plus reduction work instead of one multi-output producer/reduction kernel; Inductor cannot do this today because its scheduler does not fuse a materialized layout output and a sibling reduction over the same pointwise expression when the layout output must preserve a non-contiguous view stride; the fix is SCHEDULER_FUSION: teach Inductor to form a full-scope multi-output schedule that writes the required layout base while accumulating the compatible column reduction in the same generated kernel."""
from __future__ import annotations

import argparse
import sys
import importlib.util
import json
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

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true",
                        help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true",
                        help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2,
                        help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2,
                        help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25,
                        help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200,
                        help="Repetitions for benchmark")
    parser.add_argument("--no-skip-stochastic", action="store_true",
                        help="Disable auto-detection and skipping of stochastic outputs")
    parser.add_argument("--all-shapes", action="store_true",
                        help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true",
                        help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = _harness_get_inputs(REPRO_DIR)
    instance = _harness_get_repro_instance(REPRO_DIR)

    if has_stochastic_ops(REPRO_PATH):
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = check_oracle(
            oracle_forward,
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
            skip_stochastic=not args.no_skip_stochastic,
        )
        status = "PASS" if ok else "FAIL"
        print(f"Correctness: {status}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
        else:
            result = bench_oracle(
                oracle_forward,
                instance,
                inputs,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile "
                      f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
