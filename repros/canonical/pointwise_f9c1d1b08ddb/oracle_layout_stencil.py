"""Full-scope Triton oracle for pointwise_f9c1d1b08ddb.

Gap diagnosis:
  Classification: SCHEDULER_FUSION
  What oracle does differently: The oracle fuses each input branch's ReLU into
    the 3x3 stride-2 max-pool stencil and writes directly into the final
    concatenated [N, 2C, Hout, Wout] value/offset outputs, avoiding an explicit
    relu/cat materialization.
  What Inductor change would fix: Teach scheduler/codegen to sink cat branch
    producers into a following max-pool-with-offsets stencil and write the final
    concatenated output layout directly.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl

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


REPRO_ID = "pointwise_f9c1d1b08ddb"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

CLASSIFICATION = "SCHEDULER_FUSION"
HISTORICAL_BEST_COMPILE_US = 312.1280074119568

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


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create Repro() for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


@triton.jit
def _relu_maxpool_branch_kernel(
    inp_ptr,
    out_val_ptr,
    out_idx_ptr,
    rows: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    out_height: tl.constexpr,
    out_width: tl.constexpr,
    out_channel_offset: tl.constexpr,
    block_m: tl.constexpr,
    block_w: tl.constexpr,
):
    row = tl.program_id(0) * block_m + tl.arange(0, block_m)
    ow = tl.arange(0, block_w)

    row_mask = row < rows
    ow_mask = ow < out_width
    mask = row_mask[:, None] & ow_mask[None, :]

    oh = row % out_height
    tmp = row // out_height
    channel = tmp % channels
    batch = tmp // channels

    input_base = batch * channels * height * width + channel * height * width
    ih = oh * 2
    iw = ow * 2

    best_val = tl.full((block_m, block_w), -float("inf"), tl.float32)
    best_idx = tl.zeros((block_m, block_w), tl.int8)

    for kh in tl.static_range(3):
        for kw in tl.static_range(3):
            in_offsets = input_base[:, None] + (ih[:, None] + kh) * width + (iw[None, :] + kw)
            vals = tl.load(inp_ptr + in_offsets, mask=mask, other=0.0).to(tl.float32)
            vals = tl.maximum(vals, 0.0)
            better = vals > best_val
            best_val = tl.where(better, vals, best_val)
            candidate_idx = tl.full((block_m, block_w), kh * 3 + kw, tl.int8)
            best_idx = tl.where(better, candidate_idx, best_idx)

    out_channel = channel + out_channel_offset
    out_rows = batch * (channels * 2) * out_height + out_channel * out_height + oh
    out_offsets = out_rows[:, None] * out_width + ow[None, :]
    tl.store(out_val_ptr + out_offsets, best_val, mask=mask)
    tl.store(out_idx_ptr + out_offsets, best_idx, mask=mask)


def _pool_output_size(size: int) -> int:
    return math.ceil((size - 3) / 2) + 1


def _block_w_for(out_width: int) -> int:
    return triton.next_power_of_2(out_width)


def _validate_inputs(lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    if not lhs.is_cuda or not rhs.is_cuda:
        raise RuntimeError("CUDA inputs are required for the Triton oracle")
    if lhs.shape != rhs.shape:
        raise ValueError(f"input shape mismatch: {tuple(lhs.shape)} vs {tuple(rhs.shape)}")
    if lhs.dtype != rhs.dtype:
        raise TypeError(f"input dtype mismatch: {lhs.dtype} vs {rhs.dtype}")
    if lhs.dim() != 4:
        raise ValueError(f"expected rank-4 NCHW inputs, got {tuple(lhs.shape)}")
    if not lhs.is_contiguous() or not rhs.is_contiguous():
        raise ValueError("expected contiguous NCHW inputs")
    if lhs.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise TypeError(f"unsupported dtype for this oracle: {lhs.dtype}")


def _make_outputs(lhs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, channels, height, width = lhs.shape
    out_height = _pool_output_size(int(height))
    out_width = _pool_output_size(int(width))
    out_shape = (batch, channels * 2, out_height, out_width)
    values = torch.empty(out_shape, device=lhs.device, dtype=lhs.dtype)
    offsets = torch.empty(out_shape, device=lhs.device, dtype=torch.int8)
    return values, offsets


def _launch_oracle(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    values: torch.Tensor,
    offsets: torch.Tensor,
    *,
    block_m: int,
    num_warps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_inputs(lhs, rhs)
    batch, channels, height, width = [int(dim) for dim in lhs.shape]
    out_height = _pool_output_size(height)
    out_width = _pool_output_size(width)
    expected_shape = (batch, channels * 2, out_height, out_width)
    if tuple(values.shape) != expected_shape or tuple(offsets.shape) != expected_shape:
        raise ValueError(
            f"output shape mismatch: expected {expected_shape}, "
            f"got {tuple(values.shape)} and {tuple(offsets.shape)}"
        )
    if values.dtype != lhs.dtype or offsets.dtype != torch.int8:
        raise TypeError("output dtypes must be input dtype and torch.int8")
    if not values.is_contiguous() or not offsets.is_contiguous():
        raise ValueError("outputs must be contiguous")

    rows = batch * channels * out_height
    block_w = _block_w_for(out_width)
    grid = (triton.cdiv(rows, block_m),)
    _relu_maxpool_branch_kernel[grid](
        lhs,
        values,
        offsets,
        rows,
        channels,
        height,
        width,
        out_height,
        out_width,
        0,
        block_m,
        block_w,
        num_warps=num_warps,
    )
    _relu_maxpool_branch_kernel[grid](
        rhs,
        values,
        offsets,
        rows,
        channels,
        height,
        width,
        out_height,
        out_width,
        channels,
        block_m,
        block_w,
        num_warps=num_warps,
    )
    return values, offsets


def oracle_layout_stencil(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    block_m: int = 16,
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    values, offsets = _make_outputs(lhs)
    return _launch_oracle(lhs, rhs, values, offsets, block_m=block_m, num_warps=num_warps)


@oracle_impl(hardware="H100", shapes="(T([512, 128, 27, 27], f16), T([512, 128, 27, 27], f16))")
def oracle_forward(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full-scope oracle on the exact Repro.forward input tuple."""
    if len(inputs) != 2:
        raise ValueError(f"expected two inputs, got {len(inputs)}")
    lhs, rhs = inputs
    if not isinstance(lhs, torch.Tensor) or not isinstance(rhs, torch.Tensor):
        raise TypeError("expected tensor inputs")
    return oracle_layout_stencil(lhs, rhs)


def _load_repro_module() -> Any:
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _as_tuple(value: object) -> tuple[object, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual.float() - expected.float()).abs().max().item()


def run_check(block_m: int, num_warps: int, rtol: float, atol: float) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    inputs = tuple(get_inputs())
    instance = get_repro_instance().cuda()
    with torch.no_grad():
        expected = _as_tuple(instance(*inputs))
        actual = _as_tuple(oracle_layout_stencil(*inputs, block_m=block_m, num_warps=num_warps))
        torch.cuda.synchronize()

    if len(actual) != len(expected):
        print(f"  SCOPE_MISMATCH output count oracle={len(actual)} eager={len(expected)}")
        print("Correctness: FAIL")
        return False

    ok = True
    for idx, (got, ref) in enumerate(zip(actual, expected)):
        if not isinstance(got, torch.Tensor) or not isinstance(ref, torch.Tensor):
            item_ok = got == ref
            print(f"  output {idx}: {'PASS' if item_ok else 'FAIL'} (non-tensor)")
            ok = ok and bool(item_ok)
            continue

        metadata_ok = (
            got.shape == ref.shape
            and got.dtype == ref.dtype
            and got.stride() == ref.stride()
        )
        if ref.is_floating_point():
            value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
            diff = _max_diff(got, ref)
            print(
                f"  output {idx}: {'PASS' if metadata_ok and value_ok else 'FAIL'} "
                f"(shape={list(got.shape)} dtype={got.dtype} "
                f"stride={list(got.stride())} max_diff={diff:.2e})"
            )
        else:
            value_ok = torch.equal(got, ref)
            print(
                f"  output {idx}: {'PASS' if metadata_ok and value_ok else 'FAIL'} "
                f"(exact, dtype={got.dtype} stride={list(got.stride())})"
            )
        ok = ok and metadata_ok and value_ok

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def _bench_cuda_graph(fn: object, warmup: int, rep: int) -> float:
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
) -> object:
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


def _bench_one_oracle(
    inputs: tuple[Any, ...],
    *,
    block_m: int,
    num_warps: int,
    warmup: int,
    rep: int,
) -> float:
    lhs, rhs = inputs
    values, offsets = _make_outputs(lhs)
    return _bench_cuda_graph(
        lambda: _launch_oracle(
            lhs,
            rhs,
            values,
            offsets,
            block_m=block_m,
            num_warps=num_warps,
        ),
        warmup=warmup,
        rep=rep,
    )


def run_bench(block_m: int, num_warps: int, warmup: int, rep: int, no_compile: bool) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    module = _load_repro_module()
    inputs = tuple(module.make_inputs())
    lhs, rhs = inputs
    _validate_inputs(lhs, rhs)
    batch, channels, height, width = [int(dim) for dim in lhs.shape]
    out_height = _pool_output_size(height)
    out_width = _pool_output_size(width)

    input_bytes = lhs.numel() * lhs.element_size() + rhs.numel() * rhs.element_size()
    output_bytes = (
        batch * channels * 2 * out_height * out_width * lhs.element_size()
        + batch * channels * 2 * out_height * out_width
    )
    stencil_read_bytes = 9 * batch * channels * 2 * out_height * out_width * lhs.element_size()
    print(
        "oracle shape: "
        f"lhs={lhs.dtype}{tuple(lhs.shape)} rhs={rhs.dtype}{tuple(rhs.shape)} "
        f"out=({batch}, {channels * 2}, {out_height}, {out_width})"
    )
    print(
        "oracle tiling: "
        f"block_m={block_m} block_w={_block_w_for(out_width)} num_warps={num_warps} "
        f"input_bytes={input_bytes / 1e6:.1f} MB output_bytes={output_bytes / 1e6:.1f} MB "
        f"stencil_read_bytes={stencil_read_bytes / 1e6:.1f} MB"
    )

    with torch.no_grad():
        oracle_us = _bench_one_oracle(
            inputs,
            block_m=block_m,
            num_warps=num_warps,
            warmup=warmup,
            rep=rep,
        )
    print(f"oracle full-scope relu/cat/maxpool_with_offsets: {oracle_us:.3f} us")
    print(f"oracle_us={oracle_us:.3f}")

    compile_results: list[dict[str, object]] = []
    if not no_compile:
        holder: list[Any] = [None]
        print("torch.compile timings cover the same full Repro.forward return tuple")
        for label, config in COMPILE_CONFIGS:
            try:
                compiled = _compile_with_config(module, inputs, config, warmup)
                us = _bench_cuda_graph(
                    lambda: holder.__setitem__(0, compiled(*inputs)),
                    warmup=warmup,
                    rep=rep,
                )
                compile_results.append({"label": label, "us": us})
                print(f"torch.compile {label}: {us:.3f} us")
            except Exception as exc:
                compile_results.append({"label": label, "error": str(exc)})
                print(f"torch.compile {label}: FAILED ({exc})")

    successful_compile_us = [float(result["us"]) for result in compile_results if "us" in result]
    best_required_compile_us = min(successful_compile_us) if successful_compile_us else None
    beats_required_compile = (
        len(successful_compile_us) == len(COMPILE_CONFIGS)
        and best_required_compile_us is not None
        and oracle_us < best_required_compile_us
    )
    beats_historical_best = oracle_us < HISTORICAL_BEST_COMPILE_US
    true_floor = bool(beats_required_compile and beats_historical_best)

    if best_required_compile_us is not None:
        print(f"best_required_compile_us={best_required_compile_us:.3f}")
    print(f"historical_best_compile_us={HISTORICAL_BEST_COMPILE_US:.3f}")
    print(f"beats_required_compile={beats_required_compile}")
    print(f"beats_historical_best={beats_historical_best}")
    print(f"true_floor={true_floor}")
    if not true_floor:
        print("diagnosis_only: oracle did not beat both required configs and historical best")

    result = {
        "repro_id": REPRO_ID,
        "classification": CLASSIFICATION,
        "oracle_us": oracle_us,
        "compile_results": compile_results,
        "best_required_compile_us": best_required_compile_us,
        "historical_best_compile_us": HISTORICAL_BEST_COMPILE_US,
        "beats_required_compile": beats_required_compile,
        "beats_historical_best": beats_historical_best,
        "true_floor": true_floor,
        "status": "GOOD" if true_floor else "DIAGNOSIS_ONLY",
    }
    print(json.dumps(result, sort_keys=True))
    return result


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
