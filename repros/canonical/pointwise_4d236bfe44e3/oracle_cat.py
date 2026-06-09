"""Gap diagnosis (classification: NEW_PATTERN): this oracle materializes the full three-input f32[768] cat returned by Repro.forward with one Triton kernel that copies each contiguous input segment directly into the f32[2304] output, whereas Inductor currently lowers aten.cat as a generic pointwise fused_cat kernel over the concatenated index space with three predicated loads and nested tl.where source selection; Inductor cannot do this today because its cat lowering goes through pointwise index codegen and has no segment-aware small-list cat template that emits straight-line contiguous copies for fixed equal-size inputs; the fix is NEW_PATTERN: add an Inductor cat materialization template that splits fixed contiguous cat inputs into direct segment loads/stores inside one kernel without per-element source muxing."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful.
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

SEGMENT_N = 768
OUT_N = SEGMENT_N * 3
BLOCK_N = 1024


def get_inputs() -> tuple[object, ...]:
    """Load the exact repro inputs via repro.py::make_inputs()."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _cat3_f32_768_kernel(
        in0,
        in1,
        in2,
        out,
        SEGMENT: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK)
        mask = offsets < SEGMENT

        values0 = tl.load(in0 + offsets, mask=mask, other=0.0)
        values1 = tl.load(in1 + offsets, mask=mask, other=0.0)
        values2 = tl.load(in2 + offsets, mask=mask, other=0.0)

        tl.store(out + offsets, values0, mask=mask)
        tl.store(out + SEGMENT + offsets, values1, mask=mask)
        tl.store(out + 2 * SEGMENT + offsets, values2, mask=mask)


def _validate_input(name: str, value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != (SEGMENT_N,):
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected ({SEGMENT_N},)")
    if value.dtype is not torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous for this fixed-shape cat oracle")
    return value


@oracle_impl(hardware="H100", shapes="(T([768], f32), T([768], f32), T([768], f32))")
def oracle_forward(inputs: tuple[object, ...] | list[object]) -> torch.Tensor:
    """Compute exactly Repro()(*make_inputs()) for pointwise_4d236bfe44e3."""
    if triton is None:
        raise RuntimeError("triton is required for oracle_cat.py")
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects 3 inputs, got {len(inputs)}")

    in0 = _validate_input("input 0", inputs[0])
    in1 = _validate_input("input 1", inputs[1])
    in2 = _validate_input("input 2", inputs[2])
    if in0.device != in1.device or in0.device != in2.device:
        raise ValueError("all inputs must be on the same CUDA device")

    out = torch.empty_strided((OUT_N,), (1,), device=in0.device, dtype=torch.float32)
    _cat3_f32_768_kernel[(1,)](
        in0,
        in1,
        in2,
        out,
        SEGMENT=SEGMENT_N,
        BLOCK=BLOCK_N,
        num_warps=4,
    )
    return out


def _device_from_inputs(inputs: tuple[object, ...]) -> torch.device:
    for value in inputs:
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _status(compile_us: float, oracle_us: float) -> str:
    ratio = compile_us / oracle_us if oracle_us > 0 else 0.0
    if ratio > 1.05:
        return "GOOD"
    if ratio < 0.95:
        return "BAD_ORACLE"
    return "AT_FLOOR"


def bench_oracle_cudagraph(
    instance: torch.nn.Module,
    inputs: tuple[object, ...],
    *,
    warmup: int,
    rep: int,
) -> dict[str, object]:
    """Benchmark full oracle_forward scope with the same CUDA graph style as repro.py."""
    device = _device_from_inputs(inputs)
    if device.type != "cuda":
        return bench_oracle(oracle_forward, instance, inputs, REPRO_ID, warmup=warmup, rep=rep)

    from triton.testing import do_bench
    import torch._inductor.config as cfg

    with torch.no_grad():
        oracle_forward(inputs)
        torch.cuda.synchronize()

        oracle_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(oracle_graph):
            oracle_forward(inputs)
        cfg.coordinate_descent_tuning = True
        compiled = torch.compile(instance)
        for _ in range(5):
            compiled(*inputs)
        torch.cuda.synchronize()

        compile_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(compile_graph):
            compiled(*inputs)
        torch.cuda.synchronize()

        for _ in range(max(5, warmup)):
            oracle_graph.replay()
            compile_graph.replay()
        torch.cuda.synchronize()

        # Prime do_bench/event state so launch-floor timings are not order-biased.
        do_bench(
            lambda: oracle_graph.replay(),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        )
        do_bench(
            lambda: compile_graph.replay(),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        )

        oracle_us = do_bench(
            lambda: oracle_graph.replay(),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0
        compile_us = do_bench(
            lambda: compile_graph.replay(),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0

    ratio = compile_us / oracle_us if oracle_us > 0 else 0.0
    result: dict[str, object] = {
        "repro_id": REPRO_ID,
        "oracle_us": round(oracle_us, 2),
        "compile_us": round(compile_us, 2),
        "ratio": round(ratio, 3),
        "status": _status(compile_us, oracle_us),
    }
    print(json.dumps(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt")
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
            result = bench_oracle_cudagraph(
                instance,
                inputs,
                warmup=args.warmup,
                rep=args.rep,
            )
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
