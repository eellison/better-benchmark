"""
Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle covers the full
f32 softmax materialization from repro.py, including the metadata view from
`[256,32,33]` to `[32,8,32,33]`, the multiply by one, stable max/exp/sum
normalization with scale `1/8`, the same-shape expand, and the final metadata
view back to contiguous `[256,32,33]`. It differs from Inductor only by using a
shape-specialized Triton multi-row softmax template for the 33-wide rows, while
the required Inductor configs already lower this simple view/pointwise/reduce
chain to a compact fused softmax kernel. Inductor cannot be materially improved
here by another scheduler fusion because the remaining cost is one launch plus
the unavoidable f32 read, exp/reduce, and f32 write traffic rather than a missed
intermediate materialization; the fix class is BANDWIDTH_BOUND: treat this as a
diagnosis-only oracle unless a broader small-K softmax template beats the
required compiled configs under interleaved timing.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl

from oracle_harness import (
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
)


REPRO_ID = "amax_sum_351ca37dddb9"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

INPUT_SHAPE = (256, 32, 33)
VIEW_SHAPE = (32, 8, 32, 33)
OUT_SHAPE = INPUT_SHAPE
OUT_STRIDE = (32 * 33, 33, 1)
ROWS = 256 * 32
COLS = 33
SCALE = 0.125
CLASSIFICATION = "BANDWIDTH_BOUND"

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning=True", {"coordinate_descent_tuning": True}),
    (
        "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,"
        "coordinate_descent_tuning=True,benchmark_combo_kernel=True,"
        "triton.multi_kernel=3",
        {
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": True,
            "coordinate_descent_tuning": True,
            "benchmark_combo_kernel": True,
            "triton.multi_kernel": 3,
        },
    ),
]


@triton.jit
def _scaled_softmax_rows_kernel(
    x_ptr,
    out_ptr,
    x_s0: tl.constexpr,
    x_s1: tl.constexpr,
    x_s2: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    out_s2: tl.constexpr,
    block_rows: tl.constexpr,
    block_cols: tl.constexpr,
):
    row_offsets = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    col_offsets = tl.arange(0, block_cols)
    mask = (row_offsets[:, None] < 8192) & (col_offsets[None, :] < 33)

    i0 = row_offsets // 32
    i1 = row_offsets - i0 * 32
    x_offsets = (
        i0[:, None] * x_s0
        + i1[:, None] * x_s1
        + col_offsets[None, :] * x_s2
    )
    vals = tl.load(x_ptr + x_offsets, mask=mask, other=-float("inf")).to(tl.float32)

    row_max = tl.max(vals, axis=1)
    numer = tl.exp2((vals - row_max[:, None]) * 0.18033688011112042)
    denom = tl.sum(numer, axis=1)
    out_vals = numer / denom[:, None]

    out_offsets = (
        i0[:, None] * out_s0
        + i1[:, None] * out_s1
        + col_offsets[None, :] * out_s2
    )
    tl.store(out_ptr + out_offsets, out_vals, mask=mask)


def get_inputs() -> tuple[Any, ...]:
    """Load inputs from the repro's make_inputs."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _load_repro_module() -> Any:
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_inputs(seed: int) -> tuple[Any, ...]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return tuple(
        item.cuda() if isinstance(item, torch.Tensor) and not item.is_cuda else item
        for item in get_inputs()
    )


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _validate_shape_param(name: str, actual: Any, expected: tuple[int, ...]) -> None:
    if actual is None:
        return
    got = _shape_tuple(actual)
    if got != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {got}")


def _validate_inputs(
    bmm_14: torch.Tensor,
    _shape_param_0: Any = None,
    _shape_param_1: Any = None,
    _shape_param_2: Any = None,
) -> None:
    if not bmm_14.is_cuda:
        raise RuntimeError("CUDA tensor input is required")
    if bmm_14.dtype != torch.float32:
        raise TypeError(f"expected bmm_14 fp32, got {bmm_14.dtype}")
    if tuple(bmm_14.shape) != INPUT_SHAPE:
        raise ValueError(f"expected bmm_14 shape {INPUT_SHAPE}, got {tuple(bmm_14.shape)}")
    _validate_shape_param("_shape_param_0", _shape_param_0, VIEW_SHAPE)
    _validate_shape_param("_shape_param_1", _shape_param_1, VIEW_SHAPE)
    _validate_shape_param("_shape_param_2", _shape_param_2, OUT_SHAPE)


def _make_output(device: torch.device) -> torch.Tensor:
    return torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=device,
        dtype=torch.float32,
    )


def _launch_oracle(
    bmm_14: torch.Tensor,
    out: torch.Tensor,
    *,
    block_rows: int,
    block_cols: int,
    num_warps: int,
) -> torch.Tensor:
    if block_rows <= 0 or block_rows & (block_rows - 1):
        raise ValueError(f"block_rows must be a positive power of two, got {block_rows}")
    if block_cols < COLS or block_cols & (block_cols - 1):
        raise ValueError(f"block_cols must be a power of two >= {COLS}, got {block_cols}")
    if out.shape != OUT_SHAPE or out.dtype != torch.float32 or not out.is_cuda:
        raise ValueError("preallocated output must be CUDA fp32 with repro output shape")
    if out.stride() != OUT_STRIDE:
        raise ValueError(f"preallocated output stride mismatch: expected {OUT_STRIDE}, got {out.stride()}")

    grid = (triton.cdiv(ROWS, block_rows),)
    _scaled_softmax_rows_kernel[grid](
        bmm_14,
        out,
        x_s0=bmm_14.stride(0),
        x_s1=bmm_14.stride(1),
        x_s2=bmm_14.stride(2),
        out_s0=out.stride(0),
        out_s1=out.stride(1),
        out_s2=out.stride(2),
        block_rows=block_rows,
        block_cols=block_cols,
        num_warps=num_warps,
    )
    return out


def oracle_online_softmax(
    bmm_14: torch.Tensor,
    _shape_param_0: Any = None,
    _shape_param_1: Any = None,
    _shape_param_2: Any = None,
    *,
    block_rows: int = 16,
    block_cols: int | None = None,
    num_warps: int = 1,
) -> torch.Tensor:
    _validate_inputs(bmm_14, _shape_param_0, _shape_param_1, _shape_param_2)
    out = _make_output(bmm_14.device)
    actual_block_cols = block_cols if block_cols is not None else triton.next_power_of_2(COLS)
    return _launch_oracle(
        bmm_14,
        out,
        block_rows=block_rows,
        block_cols=actual_block_cols,
        num_warps=num_warps,
    )


def oracle_forward(inputs: tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward scope on the exact input tuple."""
    bmm_14, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    return oracle_online_softmax(
        bmm_14,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
    )


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    diff = (actual_f32 - expected_f32).abs()
    max_abs = torch.nan_to_num(diff, nan=0.0).max().item()
    rel = diff / expected_f32.abs().clamp_min(1e-8)
    max_rel = torch.nan_to_num(rel, nan=0.0).max().item()
    return max_abs, max_rel


def _bench_cuda_graph(fn: Any, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times: list[float] = []
    for _ in range(rep):
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def _compile_with_config(
    module: Any,
    inputs: tuple[Any, ...],
    config: dict[str, object],
    warmup: int,
) -> Any:
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    model = module.Repro().cuda()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        for _ in range(max(1, warmup)):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def run_check(
    block_rows: int,
    block_cols: int | None,
    num_warps: int,
    rtol: float,
    atol: float,
) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    inputs = _make_inputs(seed=1234)
    _validate_inputs(*inputs)
    model = get_repro_instance().cuda()

    with torch.no_grad():
        expected = _as_tuple(model(*inputs))
        actual = _as_tuple(
            oracle_online_softmax(
                *inputs,
                block_rows=block_rows,
                block_cols=block_cols,
                num_warps=num_warps,
            )
        )
        torch.cuda.synchronize()

    if len(actual) != len(expected):
        print(f"  SCOPE_MISMATCH: expected {len(expected)} outputs, got {len(actual)}")
        print("Correctness: FAIL")
        return False

    ok = True
    for idx, (got_item, ref_item) in enumerate(zip(actual, expected)):
        if not isinstance(got_item, torch.Tensor) or not isinstance(ref_item, torch.Tensor):
            item_ok = got_item == ref_item
            print(f"  output {idx}: {'PASS' if item_ok else 'FAIL'} non-tensor")
            ok = ok and bool(item_ok)
            continue

        shape_ok = got_item.shape == ref_item.shape
        dtype_ok = got_item.dtype == ref_item.dtype
        stride_ok = got_item.stride() == ref_item.stride()
        value_ok = torch.allclose(
            got_item.float(),
            ref_item.float(),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
        max_abs, max_rel = _max_diff(got_item, ref_item)
        item_ok = shape_ok and dtype_ok and stride_ok and value_ok
        ok = ok and bool(item_ok)
        print(
            f"  output {idx}: {'PASS' if item_ok else 'FAIL'} "
            f"shape={list(got_item.shape)} dtype={got_item.dtype} "
            f"stride={list(got_item.stride())} max_abs={max_abs:.6e} "
            f"max_rel={max_rel:.6e} shape_match={shape_ok} "
            f"dtype_match={dtype_ok} stride_match={stride_ok} allclose={value_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def run_bench(
    block_rows: int,
    block_cols: int | None,
    num_warps: int,
    warmup: int,
    rep: int,
    no_compile: bool,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    module = _load_repro_module()
    inputs = _make_inputs(seed=4321)
    _validate_inputs(*inputs)
    bmm_14 = inputs[0]
    out = _make_output(bmm_14.device)
    actual_block_cols = block_cols if block_cols is not None else triton.next_power_of_2(COLS)
    logical_bytes = bmm_14.numel() * bmm_14.element_size() * 2

    print(
        "oracle shape: "
        f"bmm_14=f32{tuple(bmm_14.shape)} stride={tuple(bmm_14.stride())} "
        f"view=f32{VIEW_SHAPE} out=f32{OUT_SHAPE} stride={OUT_STRIDE}"
    )
    print(
        "oracle tiling: "
        f"block_rows={block_rows} block_cols={actual_block_cols} "
        f"num_warps={num_warps} logical_bytes={logical_bytes / 1e6:.3f} MB"
    )

    with torch.no_grad():
        oracle_us = _bench_cuda_graph(
            lambda: _launch_oracle(
                bmm_14,
                out,
                block_rows=block_rows,
                block_cols=actual_block_cols,
                num_warps=num_warps,
            ),
            warmup=warmup,
            rep=rep,
        )

    oracle_bw = logical_bytes / (oracle_us * 1e-6) / 1e12
    print(
        "oracle full-scope scaled softmax: "
        f"{oracle_us:.3f} us ({oracle_bw:.3f} TB/s logical bytes)"
    )
    print(f"oracle_us={oracle_us:.3f}")

    compile_times: list[tuple[str, float]] = []
    if not no_compile:
        print("CUDA graph replay timings cover the same repro.py view, mul, amax, scaled exp/sum/div, expand, and view")
        for label, config in COMPILE_CONFIGS:
            try:
                compiled = _compile_with_config(module, inputs, config, warmup)
                us = _bench_cuda_graph(lambda: compiled(*inputs), warmup=warmup, rep=rep)
                compile_times.append((label, us))
                print(f"torch.compile {label}: {us:.3f} us")
            except Exception as exc:
                print(f"torch.compile {label}: FAILED ({exc})")

    best_compile = min((us for _, us in compile_times), default=float("nan"))
    ratio = best_compile / oracle_us if compile_times and oracle_us > 0 else float("nan")
    valid_floor = bool(compile_times) and len(compile_times) == len(COMPILE_CONFIGS) and oracle_us < best_compile
    status = (
        "GOOD"
        if compile_times and ratio > 1.05
        else "AT_FLOOR"
        if compile_times and ratio >= 0.95
        else "BAD_ORACLE"
        if compile_times
        else "NO_COMPILE"
    )
    result: dict[str, Any] = {
        "repro_id": REPRO_ID,
        "oracle_us": round(oracle_us, 3),
        "compile_us": round(best_compile, 3) if compile_times else None,
        "ratio": round(ratio, 3) if compile_times else None,
        "status": status,
        "classification": CLASSIFICATION,
        "valid_floor": valid_floor,
        "compile_configs_us": {label: round(us, 3) for label, us in compile_times},
    }
    if compile_times:
        print(f"best_required_compile_us={best_compile:.3f}")
        print(f"valid_floor={valid_floor}")
        if not valid_floor:
            print("diagnosis_only: oracle is not a true floor because a required compile config is as fast or faster")
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Oracle for {REPRO_ID}")
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=200)
    parser.add_argument("--block-rows", type=int, default=16, help="number of rows per Triton program")
    parser.add_argument("--block-cols", type=int, default=None, help="Triton row tile size")
    parser.add_argument("--num-warps", type=int, default=1, help="Triton warps per program")
    parser.add_argument("--no-compile", action="store_true", help="skip torch.compile baselines")
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = args.bench = True

    if args.check:
        print(f"Checking {REPRO_ID}...")
        if not run_check(
            block_rows=args.block_rows,
            block_cols=args.block_cols,
            num_warps=args.num_warps,
            rtol=args.rtol,
            atol=args.atol,
        ):
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        run_bench(
            block_rows=args.block_rows,
            block_cols=args.block_cols,
            num_warps=args.num_warps,
            warmup=args.warmup,
            rep=args.rep,
            no_compile=args.no_compile,
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
