"""
Full-scope Triton oracle for amax_sum_sum_b942094d64da.

Gap diagnosis (classification: SCHEDULER_FUSION): the oracle computes the
two-branch softmax weights, branch-weighted spatial tensor, branch sum, and
3x3 stride-2 avg-pool epilogue in one Triton kernel for the exact repro
inputs and output. Inductor lowers the decomposed view/permute/amax/exp/sum/div
softmax, weighted branch sum, and avg_pool2d as generic scheduled work, so it
does not currently fuse the branch reduction with the following spatial pooling
reduction into this one-pass layout-specific kernel.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "amax_sum_sum_b942094d64da"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
QUEUE_PATH = REPO_ROOT / "investigation_results" / "oracle_gap_closure_queue.csv"
HISTORICAL_BEST_COMPILE_US = 17.535999417304993

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning", {"coordinate_descent_tuning": True}),
    (
        "combo_looped_cd",
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
def _softmax_weighted_avgpool_kernel(
    conv_ptr,
    view_ptr,
    out_ptr,
    total: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    out_height: tl.constexpr,
    out_width: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < total

    ow = offsets % out_width
    tmp = offsets // out_width
    oh = tmp % out_height
    tmp = tmp // out_height
    c = tmp % channels
    n = tmp // channels

    conv_base = n * (2 * channels) + c
    x0 = tl.load(conv_ptr + conv_base, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(conv_ptr + conv_base + channels, mask=mask, other=0.0).to(tl.float32)
    m = tl.maximum(x0, x1)
    e0 = tl.exp(x0 - m)
    e1 = tl.exp(x1 - m)
    denom = e0 + e1
    w0 = e0 / denom
    w1 = e1 / denom

    acc = tl.full((block,), 0.0, tl.float32)
    branch_stride = channels * height * width

    for kh in tl.static_range(0, 3):
        ih = oh * 2 + kh - 1
        valid_h = (ih >= 0) & (ih < height)
        for kw in tl.static_range(0, 3):
            iw = ow * 2 + kw - 1
            valid = mask & valid_h & (iw >= 0) & (iw < width)
            view0_offset = ((n * 2 * channels + c) * height + ih) * width + iw
            v0 = tl.load(view_ptr + view0_offset, mask=valid, other=0.0).to(tl.float32)
            v1 = tl.load(view_ptr + view0_offset + branch_stride, mask=valid, other=0.0).to(tl.float32)
            acc += v0 * w0 + v1 * w1

    tl.store(out_ptr + offsets, acc * (1.0 / 9.0), mask=mask)


def _launch_oracle(
    convolution_24: torch.Tensor,
    view_15: torch.Tensor,
    out: torch.Tensor,
    *,
    block: int,
) -> torch.Tensor:
    if not convolution_24.is_cuda or not view_15.is_cuda:
        raise RuntimeError("CUDA tensors are required for the Triton oracle")
    if convolution_24.dtype != torch.float32 or view_15.dtype != torch.float32:
        raise TypeError("expected f32 convolution_24 and view_15 inputs")
    if convolution_24.ndim != 4 or view_15.ndim != 5:
        raise ValueError("unexpected input rank for full-scope oracle")

    batch, branches, channels, height, width = view_15.shape
    if branches != 2:
        raise ValueError(f"expected two branches, got {branches}")
    if convolution_24.shape != (batch, 2 * channels, 1, 1):
        raise ValueError(
            "convolution_24 shape must be [batch, 2 * channels, 1, 1], "
            f"got {tuple(convolution_24.shape)} for view_15 {tuple(view_15.shape)}"
        )
    out_height = (height + 2 - 3) // 2 + 1
    out_width = (width + 2 - 3) // 2 + 1
    if out.shape != (batch, channels, out_height, out_width):
        raise ValueError(f"unexpected output shape {tuple(out.shape)}")
    if not convolution_24.is_contiguous() or not view_15.is_contiguous() or not out.is_contiguous():
        raise ValueError("oracle expects the canonical contiguous repro inputs and output")

    total = out.numel()
    grid = (triton.cdiv(total, block),)
    _softmax_weighted_avgpool_kernel[grid](
        convolution_24,
        view_15,
        out,
        total=total,
        channels=channels,
        height=height,
        width=width,
        out_height=out_height,
        out_width=out_width,
        block=block,
        num_warps=4,
    )
    return out


def oracle_forward(inputs: tuple, *, block: int = 128) -> torch.Tensor:
    """Compute exactly Repro()(*inputs): same inputs, one f32 NCHW output."""
    convolution_24, view_15 = inputs[:2]
    batch, _, channels, height, width = view_15.shape
    out_height = (height + 2 - 3) // 2 + 1
    out_width = (width + 2 - 3) // 2 + 1
    out = torch.empty(
        (batch, channels, out_height, out_width),
        device=view_15.device,
        dtype=view_15.dtype,
    )
    return _launch_oracle(convolution_24, view_15, out, block=block)


def _load_repro_module():
    sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def get_inputs() -> tuple:
    module = _load_repro_module()
    return tuple(module.make_inputs())


def get_repro_instance() -> torch.nn.Module:
    module = _load_repro_module()
    return module.Repro()


def _bench_cuda_graph(fn, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(rep):
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def _compile_with_config(module, inputs: tuple, config: dict[str, object], warmup: int):
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


def _historical_best_compile_us() -> float:
    if not QUEUE_PATH.exists():
        return HISTORICAL_BEST_COMPILE_US
    try:
        with QUEUE_PATH.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("repro_id") == REPRO_ID:
                    value = row.get("best_compile_us") or ""
                    return float(value) if value else HISTORICAL_BEST_COMPILE_US
    except Exception:
        return HISTORICAL_BEST_COMPILE_US
    return HISTORICAL_BEST_COMPILE_US


def run_check(*, block: int, rtol: float, atol: float) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    inputs = get_inputs()
    model = get_repro_instance().cuda()
    with torch.no_grad():
        ref = model(*inputs)
        got = oracle_forward(inputs, block=block)
        torch.cuda.synchronize()

    same_shape = got.shape == ref.shape
    same_dtype = got.dtype == ref.dtype
    same_stride = got.stride() == ref.stride()
    max_abs = (got.float() - ref.float()).abs().max().item()
    ok_values = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
    ok = bool(same_shape and same_dtype and same_stride and ok_values)

    print(
        "check full-scope softmax-weighted avgpool: "
        f"shape={list(got.shape)} dtype={got.dtype} stride={got.stride()} "
        f"ref_stride={ref.stride()} max_abs={max_abs:.6e} allclose={ok_values}"
    )
    if not same_shape:
        print(f"  SCOPE_MISMATCH shape oracle={list(got.shape)} eager={list(ref.shape)}")
    if not same_dtype:
        print(f"  SCOPE_MISMATCH dtype oracle={got.dtype} eager={ref.dtype}")
    if not same_stride:
        print(f"  SCOPE_MISMATCH stride oracle={got.stride()} eager={ref.stride()}")
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def run_bench(*, block: int, warmup: int, rep: int, no_compile: bool) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    module = _load_repro_module()
    inputs = tuple(module.make_inputs())
    convolution_24, view_15 = inputs[:2]
    batch, _, channels, height, width = view_15.shape
    out_height = (height + 2 - 3) // 2 + 1
    out_width = (width + 2 - 3) // 2 + 1
    out = torch.empty(
        (batch, channels, out_height, out_width),
        device=view_15.device,
        dtype=view_15.dtype,
    )

    read_bytes = out.numel() * (2 + 18) * 4
    write_bytes = out.numel() * 4
    total_bytes = read_bytes + write_bytes

    with torch.no_grad():
        oracle_us = _bench_cuda_graph(
            lambda: _launch_oracle(convolution_24, view_15, out, block=block),
            warmup=warmup,
            rep=rep,
        )

    historical_best = _historical_best_compile_us()
    compile_results: dict[str, float] = {}
    if not no_compile:
        for label, config in COMPILE_CONFIGS:
            compiled = _compile_with_config(module, inputs, config, warmup)
            compile_results[label] = _bench_cuda_graph(
                lambda compiled=compiled: compiled(*inputs),
                warmup=warmup,
                rep=rep,
            )

    measured_best = min(compile_results.values()) if compile_results else math.nan
    comparison_best = historical_best
    if compile_results:
        comparison_best = min(historical_best, measured_best)
    speedup_vs_best = comparison_best / oracle_us if oracle_us > 0 else math.nan
    valid_floor = bool(
        compile_results
        and all(oracle_us < value for value in compile_results.values())
        and oracle_us < historical_best
    )

    result = {
        "repro_id": REPRO_ID,
        "oracle_us": round(oracle_us, 3),
        "compile_us": round(measured_best, 3) if compile_results else None,
        "compile_configs_us": {k: round(v, 3) for k, v in compile_results.items()},
        "historical_best_compile_us": round(historical_best, 3),
        "comparison_best_compile_us": round(comparison_best, 3),
        "ratio": round(speedup_vs_best, 3),
        "valid_floor": valid_floor,
        "status": "GOOD" if valid_floor else "DIAGNOSIS_ONLY",
        "total_logical_bytes": total_bytes,
    }
    print(json.dumps(result))
    print(f"oracle full-scope softmax-weighted avgpool: {oracle_us:.3f} us")
    for label, value in compile_results.items():
        print(f"torch.compile {label}: {value:.3f} us")
    print(f"historical best_compile_us: {historical_best:.3f} us")
    print(f"valid floor: {'yes' if valid_floor else 'no'}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="benchmark warmup iterations")
    parser.add_argument("--rep", type=int, default=50, help="benchmark repetitions")
    parser.add_argument("--block", type=int, default=128, help="Triton output elements per program")
    parser.add_argument("--rtol", type=float, default=1e-4, help="correctness relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-4, help="correctness absolute tolerance")
    parser.add_argument("--no-compile", action="store_true", help="skip torch.compile baselines")
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check:
        ok = run_check(block=args.block, rtol=args.rtol, atol=args.atol)
        if not ok:
            sys.exit(1)

    if args.bench:
        run_bench(
            block=args.block,
            warmup=args.warmup,
            rep=args.rep,
            no_compile=args.no_compile,
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
