"""
Oracle for sum_5e60905fccb8

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle folds the
`full -> select_scatter(dim=1,index=0) -> reshape` chain into the original
f32[128,768] input, then streams that input once in Triton to write the returned
non-contiguous transpose view backing storage and accumulate the sibling
f32[768] column sum, whereas tuned Inductor already lowers this full scope to
the same effective one-kernel read/store/reduce shape; Inductor cannot
materially do less today because output 0 must materialize the transpose backing
storage for the returned view and output 1 must read all 128 rows, so the
remaining cost is memory traffic plus launch overhead rather than a missing
scheduler-fusion, scatter-reduce, split-K, algebraic-elimination,
recompute-fusion, or new-pattern opportunity.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
    triton = None
    tl = None


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

ROWS = 128
COLS = 768


from oracle_harness import (  # noqa: E402
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_ROWS": 128, "BLOCK_COLS": 1}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_ROWS": 128, "BLOCK_COLS": 2}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_ROWS": 128, "BLOCK_COLS": 4}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_ROWS": 128, "BLOCK_COLS": 8}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_ROWS": 128, "BLOCK_COLS": 16}, num_warps=8, num_stages=4),
        ],
        key=["ROWS", "COLS"],
    )
    @triton.jit
    def _sum_permute_kernel(
        x_ptr,
        out_transpose_ptr,
        out_sum_ptr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        col_block = tl.program_id(0)
        rows = tl.arange(0, BLOCK_ROWS)
        cols = col_block * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        offsets = rows[:, None] * COLS + cols[None, :]
        mask = (rows[:, None] < ROWS) & (cols[None, :] < COLS)

        values = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        # out_transpose has shape [768,128] and stride (1,768), so its backing
        # storage is laid out exactly like the original [128,768] input.
        tl.store(out_transpose_ptr + offsets, values, mask=mask)
        sums = tl.sum(tl.where(mask, values, 0.0), axis=0)
        tl.store(out_sum_ptr + cols, sums, mask=cols < COLS)


def _validate_inputs(inputs):
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects three inputs, got {len(inputs)}")

    x, shape0, shape1 = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected tensor input 0, got {type(x)!r}")
    if x.device.type != "cuda":
        raise RuntimeError("this Triton oracle requires CUDA inputs")
    if x.dtype != torch.float32:
        raise ValueError(f"expected float32 input, got {x.dtype}")
    if tuple(x.shape) != (ROWS, COLS):
        raise ValueError(f"expected input shape {(ROWS, COLS)}, got {tuple(x.shape)}")
    if not x.is_contiguous():
        raise ValueError(f"expected contiguous input, got stride={x.stride()}")
    if tuple(shape0) != (ROWS, COLS):
        raise ValueError(f"unexpected reshape parameter 0: {shape0!r}")
    if tuple(shape1) != (COLS,):
        raise ValueError(f"unexpected reshape parameter 1: {shape1!r}")
    return x


def oracle_forward(inputs):
    """Run the full-scope oracle for both Repro.forward outputs."""
    if triton is None:
        raise RuntimeError("triton is required for oracle_sum_permute.py")

    x = _validate_inputs(inputs)
    out_transpose = torch.empty_strided(
        (COLS, ROWS),
        (1, COLS),
        device=x.device,
        dtype=torch.float32,
    )
    out_sum = torch.empty((COLS,), device=x.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(COLS, meta["BLOCK_COLS"]),)
    _sum_permute_kernel[grid](
        x,
        out_transpose,
        out_sum,
        ROWS=ROWS,
        COLS=COLS,
    )
    return out_transpose, out_sum


def _normalize_outputs(outputs):
    if isinstance(outputs, torch.Tensor):
        return [outputs]
    if isinstance(outputs, (tuple, list)):
        return list(outputs)
    return []


def _sync_from_inputs(inputs) -> None:
    for value in inputs:
        if isinstance(value, torch.Tensor) and value.device.type == "cuda":
            torch.cuda.synchronize(value.device)
            return


def run_check(rtol: float, atol: float) -> bool:
    inputs = get_inputs()
    instance = get_repro_instance()

    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        _sync_from_inputs(inputs)

    expected_list = _normalize_outputs(expected)
    actual_list = _normalize_outputs(actual)
    if len(expected_list) != len(actual_list):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(actual_list)} outputs, "
            f"eager produces {len(expected_list)}"
        )
        return False

    ok = True
    for idx, (actual_tensor, expected_tensor) in enumerate(zip(actual_list, expected_list)):
        shape_ok = actual_tensor.shape == expected_tensor.shape
        dtype_ok = actual_tensor.dtype == expected_tensor.dtype
        stride_ok = actual_tensor.stride() == expected_tensor.stride()
        value_ok = torch.allclose(
            actual_tensor.float(),
            expected_tensor.float(),
            rtol=rtol,
            atol=atol,
        )
        diff = (actual_tensor.float() - expected_tensor.float()).abs()
        max_diff = diff.max().item() if diff.numel() else 0.0
        output_ok = shape_ok and dtype_ok and stride_ok and value_ok
        ok = ok and output_ok
        print(
            f"  output {idx}: {'PASS' if output_ok else 'FAIL'} "
            f"(shape={list(actual_tensor.shape)} dtype={actual_tensor.dtype} "
            f"stride={actual_tensor.stride()} max_diff={max_diff:.2e} "
            f"allclose={value_ok})"
        )

    expected_transpose_stride = (1, COLS)
    actual_stride = actual_list[0].stride() if actual_list else None
    stride_ok = actual_stride == expected_transpose_stride
    ok = ok and stride_ok
    print(
        f"  output 0 expected transpose stride: {'PASS' if stride_ok else 'FAIL'} "
        f"(stride={actual_stride})"
    )
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


COMPILE_CONFIGS = [
    (
        "coordinate_descent_tuning=True",
        {"coordinate_descent_tuning": True},
    ),
    (
        "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,"
        "coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3",
        {
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": True,
            "coordinate_descent_tuning": True,
            "benchmark_combo_kernel": True,
            "triton.multi_kernel": 3,
        },
    ),
]


@contextlib.contextmanager
def _patched_inductor_config(config):
    import torch._inductor.config as inductor_config

    saved = []
    for key, value in config.items():
        obj = inductor_config
        parts = key.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        leaf = parts[-1]
        saved.append((obj, leaf, getattr(obj, leaf)))
        setattr(obj, leaf, value)
    try:
        yield
    finally:
        for obj, leaf, value in reversed(saved):
            setattr(obj, leaf, value)


def _compile_with_config(instance, inputs, config):
    import torch._dynamo

    torch._dynamo.reset()
    with _patched_inductor_config(config):
        compiled = torch.compile(instance)
        with torch.no_grad():
            for _ in range(5):
                compiled(*inputs)
            _sync_from_inputs(inputs)
    return compiled


def _bench_cuda(fn, warmup: int, rep: int) -> float:
    if triton is None:
        raise RuntimeError("triton is required for benchmarking")
    return triton.testing.do_bench(
        fn,
        warmup=warmup,
        rep=rep,
        return_mode="min",
    ) * 1000.0


def run_bench(warmup: int, rep: int, no_compile: bool) -> dict[str, object]:
    inputs = get_inputs()
    instance = get_repro_instance()

    with torch.no_grad():
        oracle_forward(inputs)
        _sync_from_inputs(inputs)
        oracle_us = _bench_cuda(lambda: oracle_forward(inputs), warmup=warmup, rep=rep)

    compile_times: dict[str, float] = {}
    if not no_compile:
        for label, config in COMPILE_CONFIGS:
            compiled = _compile_with_config(get_repro_instance(), inputs, config)
            with torch.no_grad():
                compile_times[label] = _bench_cuda(
                    lambda compiled=compiled: compiled(*inputs),
                    warmup=warmup,
                    rep=rep,
                )

    best_compile = min(compile_times.values()) if compile_times else math.nan
    ratio = best_compile / oracle_us if compile_times and oracle_us > 0 else math.nan
    if not compile_times:
        status = "UNKNOWN"
    elif ratio > 1.05:
        status = "GOOD"
    elif ratio < 0.95:
        status = "BAD_ORACLE"
    else:
        status = "AT_FLOOR"
    true_floor = status == "GOOD"

    result = {
        "repro_id": REPRO_ID,
        "oracle_us": round(oracle_us, 3),
        "compile_us": round(best_compile, 3) if compile_times else None,
        "ratio": round(ratio, 3) if compile_times else None,
        "status": status,
        "true_floor": true_floor,
        "compile_times": {k: round(v, 3) for k, v in compile_times.items()},
    }
    print(json.dumps(result))
    print(f"oracle full-scope transpose materialization + sum: {oracle_us:.3f} us")
    for label, value in compile_times.items():
        print(f"torch.compile {label}: {value:.3f} us")
    print(f"true_floor: {'yes' if true_floor else 'no'}")
    return result


def main():
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
    parser.add_argument("--no-compile", action="store_true", help="Only benchmark the oracle")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if has_stochastic_ops(REPRO_PATH):
        print(f"NOTE: {REPRO_ID} contains stochastic ops")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        if not run_check(rtol=args.rtol, atol=args.atol):
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        run_bench(warmup=args.warmup, rep=args.rep, no_compile=args.no_compile)


if __name__ == "__main__":
    main()
