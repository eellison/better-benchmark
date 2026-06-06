"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full 12-input float32 sum as one flat Triton streaming kernel with the minimal required twelve loads and one store per element, whereas Inductor already lowers the add chain to a fused pointwise kernel with the same irreducible memory traffic; Inductor cannot do meaningfully less work today because there is no reusable producer, reduction, scatter, or algebraic cancellation to exploit and the operation is dominated by HBM bandwidth; the fix is BANDWIDTH_BOUND: keep the single-kernel pointwise lowering and classify this repro as an at-floor memory-bandwidth case rather than adding a compiler optimization."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
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

    @triton.jit
    def _sum12_kernel(
        x0,
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7,
        x8,
        x9,
        x10,
        x11,
        out,
        n_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        acc = tl.load(x0 + offsets, mask=mask, other=0.0).to(tl.float32)
        acc = acc + tl.load(x1 + offsets, mask=mask, other=0.0).to(tl.float32)
        acc = acc + tl.load(x2 + offsets, mask=mask, other=0.0).to(tl.float32)
        acc = acc + tl.load(x3 + offsets, mask=mask, other=0.0).to(tl.float32)
        acc = acc + tl.load(x4 + offsets, mask=mask, other=0.0).to(tl.float32)
        acc = acc + tl.load(x5 + offsets, mask=mask, other=0.0).to(tl.float32)
        acc = acc + tl.load(x6 + offsets, mask=mask, other=0.0).to(tl.float32)
        acc = acc + tl.load(x7 + offsets, mask=mask, other=0.0).to(tl.float32)
        acc = acc + tl.load(x8 + offsets, mask=mask, other=0.0).to(tl.float32)
        acc = acc + tl.load(x9 + offsets, mask=mask, other=0.0).to(tl.float32)
        acc = acc + tl.load(x10 + offsets, mask=mask, other=0.0).to(tl.float32)
        acc = acc + tl.load(x11 + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(out + offsets, acc, mask=mask)


def _validate_inputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    if triton is None:
        raise RuntimeError("triton is not available")
    if len(inputs) != 12:
        raise ValueError(f"expected 12 inputs, got {len(inputs)}")

    tensors = tuple(inputs)
    if not all(isinstance(x, torch.Tensor) for x in tensors):
        raise TypeError("all repro inputs must be tensors")

    first = tensors[0]
    if first.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if first.dtype != torch.float32:
        raise TypeError(f"expected float32 inputs, got {first.dtype}")
    if not first.is_contiguous():
        raise ValueError("oracle expects the captured contiguous inputs")

    for index, tensor in enumerate(tensors[1:], start=1):
        if tensor.shape != first.shape:
            raise ValueError(
                f"input {index} shape {tuple(tensor.shape)} does not match {tuple(first.shape)}"
            )
        if tensor.dtype != torch.float32:
            raise TypeError(f"input {index} has dtype {tensor.dtype}, expected float32")
        if tensor.device != first.device:
            raise ValueError(f"input {index} is on {tensor.device}, expected {first.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"input {index} is not contiguous")

    return tensors  # type: ignore[return-value]


def oracle_forward(inputs):
    """Run the full Repro.forward computation for the same input tuple."""
    tensors = _validate_inputs(tuple(inputs))
    out = torch.empty_strided(
        tensors[0].shape,
        tensors[0].stride(),
        device=tensors[0].device,
        dtype=tensors[0].dtype,
    )
    n_elements = out.numel()
    block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)
    _sum12_kernel[grid](
        *tensors,
        out,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return out


def main():
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify correctness against eager Repro",
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="Benchmark oracle vs torch.compile",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-2,
        help="Relative tolerance for correctness check",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-2,
        help="Absolute tolerance for correctness check",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=25,
        help="Warmup iterations for benchmark",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=200,
        help="Repetitions for benchmark",
    )
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument(
        "--all-shapes",
        action="store_true",
        help="Benchmark across all shapes from shapes.txt",
    )
    parser.add_argument(
        "--show-hw",
        action="store_true",
        help="Print GPU hardware info and exit",
    )
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
