"""Oracle for pointwise_edf2e0dc44ea.

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete captured `aten.embedding.default` as a shape-specialized Triton gather/copy from the f32 `[128, 2560]` table through the i64 `[128]` ids into a fresh contiguous f32 `[128, 2560]` output, whereas tuned Inductor already lowers the same one-op repro to the same mandatory gather and materialization work; Inductor cannot materially improve this case through scheduler fusion or scatter-reduce because the semantic output is a dense fresh tensor and the remaining cost is dominated by required table reads and output writes; the fix is BANDWIDTH_BOUND: record the full-scope gather/copy floor and only revisit for broad launch/allocation or copy-codegen improvements.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None

from oracle_harness import (
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    load_repro_module,
)


REPRO_ID = "pointwise_edf2e0dc44ea"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

ROWS = 128
HIDDEN = 2560
TABLE_SHAPE = (ROWS, HIDDEN)
IDS_SHAPE = (ROWS,)
OUT_SHAPE = (ROWS, HIDDEN)
OUT_STRIDE = (HIDDEN, 1)
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


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _embedding_gather_kernel(
        table_ptr,
        ids_ptr,
        out_ptr,
        HIDDEN_: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        col_block = tl.program_id(1)
        cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = cols < HIDDEN_

        src_row = tl.load(ids_ptr + row)
        values = tl.load(table_ptr + src_row * HIDDEN_ + cols, mask=mask, other=0.0)
        tl.store(out_ptr + row * HIDDEN_ + cols, values, mask=mask)


def _load_repro_module() -> Any:
    return load_repro_module(REPRO_DIR)


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _make_inputs(module: Any, seed: int) -> tuple[Any, ...]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return tuple(
        item.cuda() if isinstance(item, torch.Tensor) and not item.is_cuda else item
        for item in module.make_inputs()
    )


def _validate_inputs(arg1_1: torch.Tensor, arg0_1: torch.Tensor) -> None:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if arg1_1.device.type != "cuda" or arg0_1.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if arg1_1.dtype != torch.float32:
        raise TypeError(f"expected fp32 embedding table, got {arg1_1.dtype}")
    if arg0_1.dtype != torch.int64:
        raise TypeError(f"expected int64 embedding ids, got {arg0_1.dtype}")
    if tuple(arg1_1.shape) != TABLE_SHAPE:
        raise ValueError(f"unexpected embedding table shape: {tuple(arg1_1.shape)}")
    if tuple(arg0_1.shape) != IDS_SHAPE:
        raise ValueError(f"unexpected embedding ids shape: {tuple(arg0_1.shape)}")
    if not arg1_1.is_contiguous():
        raise ValueError("oracle expects the captured embedding table to be contiguous")
    if not arg0_1.is_contiguous():
        raise ValueError("oracle expects the captured embedding ids to be contiguous")


def _make_output(device: torch.device) -> torch.Tensor:
    return torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=device, dtype=torch.float32)


def _launch_oracle(
    arg1_1: torch.Tensor,
    arg0_1: torch.Tensor,
    out: torch.Tensor,
    *,
    block_n: int,
    num_warps: int,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if block_n <= 0 or block_n & (block_n - 1):
        raise ValueError(f"block_n must be a positive power of two, got {block_n}")
    if tuple(out.shape) != OUT_SHAPE or tuple(out.stride()) != OUT_STRIDE:
        raise ValueError(f"output must have shape {OUT_SHAPE} and stride {OUT_STRIDE}")
    if out.dtype != torch.float32 or out.device.type != "cuda":
        raise ValueError("output must be CUDA fp32")

    _embedding_gather_kernel[(ROWS, triton.cdiv(HIDDEN, block_n))](
        arg1_1,
        arg0_1,
        out,
        HIDDEN_=HIDDEN,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return out


def oracle_embedding(
    arg1_1: torch.Tensor,
    arg0_1: torch.Tensor,
    *,
    block_n: int = 2048,
    num_warps: int = 4,
) -> torch.Tensor:
    """Compute the complete Repro.forward embedding result with Triton."""
    _validate_inputs(arg1_1, arg0_1)
    out = _make_output(arg1_1.device)
    return _launch_oracle(
        arg1_1,
        arg0_1,
        out,
        block_n=block_n,
        num_warps=num_warps,
    )


def oracle_forward(inputs: tuple[Any, ...] | list[Any]) -> torch.Tensor:
    """Run the full-scope oracle for Repro.forward."""
    arg1_1, arg0_1 = inputs
    return oracle_embedding(arg1_1, arg0_1)


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


def run_check(block_n: int, num_warps: int, rtol: float, atol: float) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    module = _load_repro_module()
    inputs = _make_inputs(module, seed=1234)
    _validate_inputs(*inputs)
    model = module.Repro().cuda()

    with torch.no_grad():
        expected = _as_tuple(model(*inputs))
        actual = _as_tuple(
            oracle_embedding(*inputs, block_n=block_n, num_warps=num_warps)
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
    block_n: int,
    num_warps: int,
    warmup: int,
    rep: int,
    no_compile: bool,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    module = _load_repro_module()
    inputs = _make_inputs(module, seed=4321)
    arg1_1, arg0_1 = inputs
    _validate_inputs(arg1_1, arg0_1)
    out = _make_output(arg1_1.device)
    logical_bytes = ROWS * HIDDEN * 4 * 2 + ROWS * 8

    print(
        "oracle shape: "
        f"arg1_1=f32{TABLE_SHAPE} stride={tuple(arg1_1.stride())} "
        f"arg0_1=i64{IDS_SHAPE} stride={tuple(arg0_1.stride())} "
        f"out=f32{OUT_SHAPE} stride={OUT_STRIDE}"
    )
    print(
        "oracle tiling: "
        f"rows={ROWS} hidden={HIDDEN} block_n={block_n} num_warps={num_warps} "
        f"logical_read_write_bytes={logical_bytes / 1e6:.3f} MB"
    )

    with torch.no_grad():
        oracle_us = _bench_cuda_graph(
            lambda: _launch_oracle(
                arg1_1,
                arg0_1,
                out,
                block_n=block_n,
                num_warps=num_warps,
            ),
            warmup=warmup,
            rep=rep,
        )

    oracle_bw = logical_bytes / (oracle_us * 1e-6) / 1e12
    print(
        "oracle full-scope embedding gather/copy: "
        f"{oracle_us:.3f} us ({oracle_bw:.3f} TB/s logical bytes)"
    )
    print(f"oracle_us={oracle_us:.3f}")

    compile_results: list[dict[str, Any]] = []
    if not no_compile:
        print("torch.compile timings cover the same repro.py aten.embedding.default")
        for label, config in COMPILE_CONFIGS:
            try:
                compiled = _compile_with_config(module, inputs, config, warmup)
                holder: list[Any] = [None]
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

    successful_compile = [float(item["us"]) for item in compile_results if "us" in item]
    best_compile_us = min(successful_compile) if successful_compile else math.nan
    ratio = best_compile_us / oracle_us if successful_compile and oracle_us > 0 else math.nan
    if successful_compile and ratio > 1.05:
        status = "GOOD"
    elif successful_compile and ratio < 0.95:
        status = "BAD_ORACLE"
    elif successful_compile:
        status = "AT_FLOOR"
    else:
        status = "NO_COMPILE"

    result = {
        "repro_id": REPRO_ID,
        "oracle_us": round(oracle_us, 3),
        "compile_us": round(best_compile_us, 3) if successful_compile else None,
        "compile_results": [
            {
                "label": item["label"],
                **(
                    {"us": round(float(item["us"]), 3)}
                    if "us" in item
                    else {"error": item["error"]}
                ),
            }
            for item in compile_results
        ],
        "ratio": round(ratio, 3) if successful_compile else None,
        "status": status,
        "classification": CLASSIFICATION,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument("--block-n", type=int, default=2048, help="hidden columns per Triton program")
    parser.add_argument("--num-warps", type=int, default=4, help="Triton warps per program")
    parser.add_argument("--no-compile", action="store_true", help="skip torch.compile config baselines")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    if args.check:
        print(f"Checking {REPRO_ID}...")
        if not run_check(args.block_n, args.num_warps, args.rtol, args.atol):
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        run_bench(args.block_n, args.num_warps, args.warmup, args.rep, args.no_compile)


if __name__ == "__main__":
    with torch.no_grad():
        main()
