"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle fuses the complete add/div, Longformer sliding-window as_strided clone, and final returned layout view into one Triton materialization kernel, whereas Inductor treats the pointwise producer and overlapping layout clone as separate scheduling regions; Inductor cannot do this today because clone/as_strided materialization is a scheduler fusion barrier even when the overlapping stencil is affine and the consumer is only a view; the fix is SCHEDULER_FUSION: allow pointwise producers to fuse into affine overlapping layout materialization kernels and preserve the requested output stride as a view of that materialized buffer."""
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


SEQ = 1024
BATCH = 2
HIDDEN = 768
HEADS = 12
HEAD_DIM = 64
HEAD_BATCH = BATCH * HEADS
WINDOWS = 3
WINDOW_SIZE = 512
WINDOW_STEP = 256
OUT_N = HEAD_BATCH * WINDOWS
BASE_SHAPE = (OUT_N, WINDOW_SIZE, HEAD_DIM)
BASE_STRIDE = (WINDOW_SIZE * HEAD_DIM, HEAD_DIM, 1)
RETURN_SHAPE = (OUT_N, HEAD_DIM, WINDOW_SIZE)
RETURN_STRIDE = (WINDOW_SIZE * HEAD_DIM, 1, HEAD_DIM)
NUMEL = OUT_N * WINDOW_SIZE * HEAD_DIM


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _layout_stencil_kernel(
        mm_ptr,
        bias_ptr,
        out_ptr,
        total: tl.constexpr,
        block_size: tl.constexpr,
        head_dim: tl.constexpr,
        window_size: tl.constexpr,
        windows: tl.constexpr,
        window_step: tl.constexpr,
        heads: tl.constexpr,
        batch_size: tl.constexpr,
        hidden: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < total

        dim = offsets % head_dim
        tmp = offsets // head_dim
        pos = tmp % window_size
        out_n = tmp // window_size

        window = out_n % windows
        head_batch = out_n // windows
        batch = head_batch // heads
        head = head_batch - batch * heads

        source_seq = pos + window * window_step
        source_feature = head * head_dim + dim
        source_row = source_seq * batch_size + batch
        source_offset = source_row * hidden + source_feature

        mm = tl.load(mm_ptr + source_offset, mask=mask, other=0.0)
        bias = tl.load(bias_ptr + source_feature, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, (mm + bias) * 0.125, mask=mask)


def _expect_layout_inputs(mm_44: torch.Tensor, arg180_1: torch.Tensor) -> None:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if not mm_44.is_cuda or not arg180_1.is_cuda:
        raise RuntimeError("This oracle is a CUDA/Triton oracle and requires CUDA inputs")
    if mm_44.dtype != torch.float32 or arg180_1.dtype != torch.float32:
        raise ValueError(f"expected f32 inputs, got {mm_44.dtype=} {arg180_1.dtype=}")
    if tuple(mm_44.shape) != (SEQ * BATCH, HIDDEN):
        raise ValueError(f"unexpected mm_44 shape: {tuple(mm_44.shape)}")
    if tuple(arg180_1.shape) != (HIDDEN,):
        raise ValueError(f"unexpected arg180_1 shape: {tuple(arg180_1.shape)}")
    if tuple(mm_44.stride()) != (HIDDEN, 1):
        raise ValueError(f"mm_44 must be contiguous, got stride={tuple(mm_44.stride())}")
    if tuple(arg180_1.stride()) != (1,):
        raise ValueError(f"arg180_1 must be contiguous, got stride={tuple(arg180_1.stride())}")


def oracle_forward(inputs):
    """Compute the full Repro.forward scope with one fused Triton materialization."""
    mm_44, arg180_1 = inputs[0], inputs[1]
    _expect_layout_inputs(mm_44, arg180_1)

    base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=mm_44.device,
        dtype=torch.float32,
    )
    grid = (triton.cdiv(NUMEL, 256),)
    _layout_stencil_kernel[grid](
        mm_44,
        arg180_1,
        base,
        total=NUMEL,
        block_size=256,
        head_dim=HEAD_DIM,
        window_size=WINDOW_SIZE,
        windows=WINDOWS,
        window_step=WINDOW_STEP,
        heads=HEADS,
        batch_size=BATCH,
        hidden=HIDDEN,
    )
    out = base.permute(0, 2, 1)
    if tuple(out.shape) != RETURN_SHAPE or tuple(out.stride()) != RETURN_STRIDE:
        raise RuntimeError(
            f"unexpected output layout: shape={tuple(out.shape)} stride={tuple(out.stride())}"
        )
    return out


def main():
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
