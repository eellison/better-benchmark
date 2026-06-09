"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete Llama RoPE rotate-half, full_3 slice_scatter base materialization, bf16 pointwise rounding boundaries, and final `[2048, 2048]` transposed view by writing the contiguous `[4, 512, 32, 64]` backing layout directly with one signed affine rotated load per element, whereas Inductor lowers the decomposed mul/slice/neg/slice_scatter/add/mul/add/permute/clone/view/permute graph to a comparable one-kernel pointwise/layout materialization but preserves the rotate-half scatter expression as separate predicated high-half and low-half load paths; Inductor cannot emit this leaner form today because its algebraic simplifier does not canonicalize paired slice_scatter rotate-half reconstruction through the final layout materialization into a single affine rotated index plus sign; the fix is ALGEBRAIC_ELIMINATION: add a guarded rotate-half slice_scatter canonicalization that preserves bf16 materialization semantics and writes the requested transposed backing storage directly."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 4
HEADS = 32
SEQ = 512
HEAD_DIM = 64
HALF_DIM = 32
LOGICAL_ROWS = BATCH * SEQ * HEADS
TOTAL_NUMEL = LOGICAL_ROWS * HEAD_DIM
OUT_SHAPE = (HEADS * HEAD_DIM, BATCH * SEQ)
OUT_STRIDE = (1, HEADS * HEAD_DIM)
SHAPE_PARAM_0 = (BATCH, SEQ, HEADS * HEAD_DIM)
SHAPE_PARAM_1 = OUT_SHAPE
CLASSIFICATION = "ALGEBRAIC_ELIMINATION"

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    oracle_impl,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
# Replace this section with your optimized Triton kernel(s).
#
# Recommended pattern: use @triton.autotune so the kernel auto-selects
# the best config for each shape encountered via --all-shapes.

if triton is not None:

    @triton.jit
    def _round_bf16_to_f32(value):
        bits = value.to(tl.int32, bitcast=True)
        lsb = (bits >> 16) & 1
        rounded = (bits + 0x7FFF + lsb) & -65536
        return rounded.to(tl.float32, bitcast=True)

    @triton.jit
    def _rope_layout_stencil_kernel(
        x_ptr,
        cos_ptr,
        full_ptr,
        sin_ptr,
        out_ptr,
        N: tl.constexpr,
        HEADS_: tl.constexpr,
        SEQ_: tl.constexpr,
        HEAD_DIM_: tl.constexpr,
        HALF_DIM_: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N

        dim = offsets % HEAD_DIM_
        seq = (offsets // HEAD_DIM_) % SEQ_
        head = (offsets // (HEAD_DIM_ * SEQ_)) % HEADS_
        batch = offsets // (HEAD_DIM_ * SEQ_ * HEADS_)

        is_hi = dim >= HALF_DIM_
        rot_offset = tl.where(is_hi, offsets - HALF_DIM_, offsets + HALF_DIM_)
        coeff_offset = seq * HEAD_DIM_ + dim
        rot_coeff_offset = tl.where(is_hi, coeff_offset - HALF_DIM_, coeff_offset + HALF_DIM_)

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x_rot = tl.load(x_ptr + rot_offset, mask=mask, other=0.0).to(tl.float32)
        full_value = tl.load(full_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        cos_rot = tl.load(cos_ptr + rot_coeff_offset, mask=mask, other=0.0).to(tl.float32)
        sin_value = tl.load(sin_ptr + coeff_offset, mask=mask, other=0.0).to(tl.float32)

        rotated_product = _round_bf16_to_f32(x_rot * cos_rot)
        signed_rotated = tl.where(is_hi, -rotated_product, rotated_product)
        first_add = _round_bf16_to_f32(full_value + signed_rotated)
        sin_product = _round_bf16_to_f32(x * sin_value)
        output = first_add + sin_product

        out_offset = dim + head * HEAD_DIM_ + seq * (HEADS_ * HEAD_DIM_) + batch * (SEQ_ * HEADS_ * HEAD_DIM_)
        tl.store(out_ptr + out_offset, output, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple, torch.Size)):
        raise TypeError(f"expected shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _validate_tensor(value: Any, shape: tuple[int, ...], dtype: torch.dtype, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{REPRO_ID} {name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{REPRO_ID} {name} expects shape {shape}, got {tuple(value.shape)}")
    if value.dtype != dtype:
        raise TypeError(f"{REPRO_ID} {name} expects {dtype}, got {value.dtype}")
    if not value.is_cuda:
        raise ValueError(f"{REPRO_ID} {name} must be a CUDA tensor")
    if not value.is_contiguous():
        raise ValueError(f"{REPRO_ID} {name} must be contiguous, got stride={value.stride()}")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    tensor_shape = (BATCH, HEADS, SEQ, HEAD_DIM)
    coeff_shape = (1, 1, SEQ, HEAD_DIM)
    x = _validate_tensor(inputs[0], tensor_shape, torch.bfloat16, "getitem_45")
    cos = _validate_tensor(inputs[1], coeff_shape, torch.bfloat16, "unsqueeze_6")
    full = _validate_tensor(inputs[2], tensor_shape, torch.bfloat16, "full_3")
    sin = _validate_tensor(inputs[3], coeff_shape, torch.bfloat16, "unsqueeze_7")

    if x.device != cos.device or x.device != full.device or x.device != sin.device:
        raise ValueError(f"{REPRO_ID} tensor inputs must be on the same CUDA device")
    if _shape_tuple(inputs[4]) != SHAPE_PARAM_0:
        raise ValueError(f"{REPRO_ID} _shape_param_0 expects {SHAPE_PARAM_0}, got {_shape_tuple(inputs[4])}")
    if _shape_tuple(inputs[5]) != SHAPE_PARAM_1:
        raise ValueError(f"{REPRO_ID} _shape_param_1 expects {SHAPE_PARAM_1}, got {_shape_tuple(inputs[5])}")

    return x, cos, full, sin


@oracle_impl(hardware="H100", shapes="(T([4, 32, 512, 64], bf16), T([1, 1, 512, 64], bf16), T([4, 32, 512, 64], bf16), T([1, 1, 512, 64], bf16), S([4, 512, 2048]), S([2048, 2048]))")
def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_rope_layout_stencil.py")

    x, cos, full, sin = _validate_inputs(inputs)
    output = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=x.device, dtype=torch.bfloat16)

    block = 1024
    _rope_layout_stencil_kernel[(triton.cdiv(TOTAL_NUMEL, block),)](
        x,
        cos,
        full,
        sin,
        output,
        N=TOTAL_NUMEL,
        HEADS_=HEADS,
        SEQ_=SEQ,
        HEAD_DIM_=HEAD_DIM,
        HALF_DIM_=HALF_DIM,
        BLOCK_N=block,
        num_warps=4,
        num_stages=3,
    )
    return output


def _check_layout(instance: torch.nn.Module, inputs: list[Any] | tuple[Any, ...]) -> bool:
    with torch.no_grad():
        eager = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    layout_ok = (
        tuple(actual.shape) == tuple(eager.shape)
        and tuple(actual.stride()) == tuple(eager.stride())
        and actual.dtype == eager.dtype
        and actual.storage_offset() == eager.storage_offset()
    )
    print(
        f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
        f"(shape={list(actual.shape)} stride={actual.stride()} "
        f"dtype={actual.dtype} storage_offset={actual.storage_offset()})"
    )
    return layout_ok


# --- CLI entry point ---
def main():
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

    # Handle --show-hw early
    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    # Default: run both --check and --bench
    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    # Report if stochastic ops detected in source
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
        ok = _check_layout(instance, inputs) and ok
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward, REPRO_DIR, REPRO_ID,
                warmup=args.warmup, rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile for "
                          f"{result['repro_id']} (ratio={result['ratio']:.3f}x)")
        else:
            # The shared harness owns timing so graph capture, GPU locking, and
            # interleaved oracle/compile measurement stay intact.
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
