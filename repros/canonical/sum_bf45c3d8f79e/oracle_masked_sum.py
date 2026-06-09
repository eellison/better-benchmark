"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete masked column sum from `Repro.forward` in one Triton reduction kernel, folding the bool-to-f32 cast, `1.25` scale, elementwise multiply, singleton view, bool `where` mask, and `[0,2,3]` reduction into direct stores of the final f32 `[1280]` result, whereas tuned Inductor is expected to emit equivalent fused reduction work for this small fixed-shape graph; if compile matches or beats this kernel, the fix is BANDWIDTH_BOUND: record it as not_true_floor because the workload is already dominated by reading the two bool inputs and one f32 input plus launch/reduction overhead rather than a missing semantic fusion."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps import useful without Triton.
    triton = None
    tl = None

from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

M = 512
N = 1280
OUT_SHAPE = (N,)
OUT_STRIDE = (1,)
ARG_SHAPE = (M, N)
ARG_STRIDE = (N, 1)
MASK_SHAPE = (M, N, 1, 1)
MASK_STRIDE = (N, 1, 1, 1)
VIEW_SHAPE = MASK_SHAPE

BLOCK_M = 512
BLOCK_N = 16
NUM_WARPS = 4


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _masked_column_sum_kernel(
        arg476_ptr,
        mm_ptr,
        arg478_ptr,
        out_ptr,
        BLOCK_M_: tl.constexpr,
        BLOCK_N_: tl.constexpr,
        N_: tl.constexpr,
    ):
        cols = tl.program_id(0) * BLOCK_N_ + tl.arange(0, BLOCK_N_)
        rows = tl.arange(0, BLOCK_M_)
        offsets = rows[:, None] * N_ + cols[None, :]
        col_mask = cols < N_

        active = tl.load(arg476_ptr + offsets, mask=col_mask[None, :], other=0).to(tl.float32)
        mm = tl.load(mm_ptr + offsets, mask=col_mask[None, :], other=0.0).to(tl.float32)
        masked = tl.load(arg478_ptr + offsets, mask=col_mask[None, :], other=1) != 0

        values = active * mm * 1.25
        values = tl.where(masked, 0.0, values)
        reduced = tl.sum(values, axis=0)
        tl.store(out_ptr + cols, reduced, mask=col_mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"shape parameter must be iterable, got {type(value)!r}") from exc


def _require_tensor(
    name: str,
    value: Any,
    *,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    arg476_1, mm, arg478_1, shape_param_0 = inputs
    arg476_1 = _require_tensor(
        "arg476_1",
        arg476_1,
        shape=ARG_SHAPE,
        stride=ARG_STRIDE,
        dtype=torch.bool,
    )
    mm = _require_tensor(
        "mm",
        mm,
        shape=ARG_SHAPE,
        stride=ARG_STRIDE,
        dtype=torch.float32,
    )
    arg478_1 = _require_tensor(
        "arg478_1",
        arg478_1,
        shape=MASK_SHAPE,
        stride=MASK_STRIDE,
        dtype=torch.bool,
    )

    if _shape_tuple(shape_param_0) != VIEW_SHAPE:
        raise ValueError(f"_shape_param_0 mismatch: expected {VIEW_SHAPE}, got {shape_param_0!r}")
    if mm.device != arg476_1.device or arg478_1.device != arg476_1.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return arg476_1, mm, arg478_1


@oracle_impl(hardware="H100", shapes="(T([512, 1280], b8), T([512, 1280], f32), T([512, 1280, 1, 1], b8), S([512, 1280, 1, 1]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full `Repro.forward` computation with a Triton reduction kernel."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_masked_sum.py")

    arg476_1, mm, arg478_1 = _validate_inputs(inputs)
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=mm.device,
        dtype=torch.float32,
    )
    grid = (triton.cdiv(N, BLOCK_N),)
    _masked_column_sum_kernel[grid](
        arg476_1,
        mm,
        arg478_1,
        out,
        BLOCK_M_=BLOCK_M,
        BLOCK_N_=BLOCK_N,
        N_=N,
        num_warps=NUM_WARPS,
        num_stages=3,
    )
    return out


def _check_layout() -> bool:
    out = oracle_forward(get_inputs())
    torch.cuda.synchronize()
    ok = tuple(out.shape) == OUT_SHAPE and tuple(out.stride()) == OUT_STRIDE
    print(
        f"  output 0 layout: {'PASS' if ok else 'FAIL'} "
        f"(shape={list(out.shape)} stride={out.stride()})"
    )
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations")
    parser.add_argument("--rep", type=int, default=200, help="Benchmark repetitions")
    parser.add_argument("--no-skip-stochastic", action="store_true")
    parser.add_argument("--all-shapes", action="store_true")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        import json

        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

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
        ok = _check_layout() and ok
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
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
                    print(
                        f"WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
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
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
