"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle materializes the complete Longformer f16 head/window layout transform returned by Repro.forward, including the view/permute-derived head indexing, constant -1 padding, overlapping as_strided windows, clone materialization, and final contiguous [192, 768, 64] output view, whereas Inductor currently lowers the pad/as_strided/clone chain as generic layout work around an intermediate padded tensor; Inductor cannot do this today because clone/as_strided materialization with constant-pad indexing is a scheduler fusion barrier even when the indexing is affine and the consumer is only a view; the fix is SCHEDULER_FUSION: fuse constant padding into affine overlapping layout materialization kernels and write the requested materialized output layout directly."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
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
CLASSIFICATION = "SCHEDULER_FUSION"

SEQ = 4096
HIDDEN = 768
HEADS = 12
HEAD_DIM = 64
WINDOWS = 16
WINDOW_SIZE = 768
WINDOW_STEP = 256
PAD_BEFORE = 256
OUT_SHAPE = (HEADS * WINDOWS, WINDOW_SIZE, HEAD_DIM)
OUT_STRIDE = (WINDOW_SIZE * HEAD_DIM, HEAD_DIM, 1)
BLOCK_M = 32


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _padded_window_layout_kernel(
        in_ptr,
        out_ptr,
        block_m: tl.constexpr,
        head_dim: tl.constexpr,
        window_size: tl.constexpr,
        window_step: tl.constexpr,
        pad_before: tl.constexpr,
        seq_len: tl.constexpr,
        hidden: tl.constexpr,
        windows: tl.constexpr,
    ):
        out_n = tl.program_id(0)
        pos_base = tl.program_id(1) * block_m

        rows = pos_base + tl.arange(0, block_m)
        dims = tl.arange(0, head_dim)
        window = out_n % windows
        head = out_n // windows

        source_seq = rows.to(tl.int64) + window.to(tl.int64) * window_step - pad_before
        source_feature = head * head_dim + dims
        source_offset = source_seq[:, None] * hidden + source_feature[None, :]
        valid = (source_seq[:, None] >= 0) & (source_seq[:, None] < seq_len)
        out_offset = out_n * window_size * head_dim + rows[:, None] * head_dim + dims[None, :]

        value = tl.load(in_ptr + source_offset, mask=valid, other=-1.0)
        tl.store(out_ptr + out_offset, value)


def _shape_tuple(value: Any, name: str) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable shape, got {type(value)!r}") from exc


def _expect_inputs(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_longformer_padded_window_layout.py")
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    addmm_68, shape0, shape1, shape2, shape3 = inputs
    if not isinstance(addmm_68, torch.Tensor):
        raise TypeError(f"expected tensor input 0, got {type(addmm_68)!r}")
    if addmm_68.device.type != "cuda":
        raise RuntimeError("This oracle is a CUDA/Triton oracle and requires CUDA inputs")
    if addmm_68.dtype != torch.float16:
        raise ValueError(f"expected f16 input, got {addmm_68.dtype}")
    if tuple(addmm_68.shape) != (SEQ, HIDDEN):
        raise ValueError(f"unexpected addmm_68 shape: {tuple(addmm_68.shape)}")
    if tuple(addmm_68.stride()) != (HIDDEN, 1):
        raise ValueError(f"addmm_68 must be contiguous, got stride={tuple(addmm_68.stride())}")

    expected_shapes = (
        (SEQ, 1, HIDDEN),
        (SEQ, 1, HEADS, HEAD_DIM),
        (HEADS, SEQ, HEAD_DIM),
        OUT_SHAPE,
    )
    actual_shapes = (
        _shape_tuple(shape0, "_shape_param_0"),
        _shape_tuple(shape1, "_shape_param_1"),
        _shape_tuple(shape2, "_shape_param_2"),
        _shape_tuple(shape3, "_shape_param_3"),
    )
    if actual_shapes != expected_shapes:
        raise ValueError(f"unexpected shape parameters: {actual_shapes}")

    return addmm_68


@oracle_impl(hardware="H100", shapes="(T([4096, 768], f16), S([4096, 1, 768]), S([4096, 1, 12, 64]), S([12, 4096, 64]), S([192, 768, 64]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Compute the full Repro.forward padded sliding-window materialization."""
    addmm_68 = _expect_inputs(inputs)
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=addmm_68.device,
        dtype=addmm_68.dtype,
    )

    grid = (HEADS * WINDOWS, triton.cdiv(WINDOW_SIZE, BLOCK_M))
    _padded_window_layout_kernel[grid](
        addmm_68,
        out,
        block_m=BLOCK_M,
        head_dim=HEAD_DIM,
        window_size=WINDOW_SIZE,
        window_step=WINDOW_STEP,
        pad_before=PAD_BEFORE,
        seq_len=SEQ,
        hidden=HIDDEN,
        windows=WINDOWS,
        num_warps=4,
        num_stages=1,
    )

    if tuple(out.shape) != OUT_SHAPE or tuple(out.stride()) != OUT_STRIDE:
        raise RuntimeError(f"unexpected output layout: shape={tuple(out.shape)} stride={tuple(out.stride())}")
    return out


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
