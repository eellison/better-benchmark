"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle directly materializes the complete Visformer Q/K/V layout split from the captured channels-last source tensor into the three eager-compatible contiguous outputs in one Triton multi-output layout kernel, whereas Inductor lowers the sibling unbind/permute/clone/reshape materializations as separate generic layout-copy work; Inductor cannot do this today because its scheduler does not fuse multiple users of an unbound strided view when each user needs a different final output index map; the fix is SCHEDULER_FUSION: teach layout-copy scheduling to emit a multi-output fused producer for sibling Q/K/V materializations with per-output affine store maps."""
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

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
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

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_P": 16, "BLOCK_D": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_P": 16, "BLOCK_D": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_P": 32, "BLOCK_D": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_P": 32, "BLOCK_D": 64}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_P": 64, "BLOCK_D": 32}, num_warps=8, num_stages=3),
        ],
        key=["ROWS", "HEADS", "D", "P"],
    )
    @triton.jit
    def _qkv_channels_last_layout_kernel(
        input_ptr,
        q_out_ptr,
        k_out_ptr,
        v_out_ptr,
        ROWS: tl.constexpr,
        HEADS: tl.constexpr,
        D: tl.constexpr,
        P: tl.constexpr,
        BATCH_STRIDE: tl.constexpr,
        POS_STRIDE: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        d_block = tl.program_id(1)
        p_block = tl.program_id(2)

        p = p_block * BLOCK_P + tl.arange(0, BLOCK_P)[:, None]
        d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)[None, :]
        mask = (p < P) & (d < D)

        batch = row // HEADS
        head = row - batch * HEADS
        qkv_stride = HEADS * D

        input_base = batch * BATCH_STRIDE + p * POS_STRIDE + head * D + d
        q_values = tl.load(input_ptr + input_base, mask=mask, other=0.0)
        k_values = tl.load(input_ptr + input_base + qkv_stride, mask=mask, other=0.0)
        v_values = tl.load(input_ptr + input_base + 2 * qkv_stride, mask=mask, other=0.0)

        qv_offsets = row * P * D + p * D + d
        k_offsets = row * D * P + tl.trans(d) * P + tl.trans(p)
        tl.store(q_out_ptr + qv_offsets, q_values, mask=mask)
        tl.store(k_out_ptr + k_offsets, tl.trans(k_values), mask=tl.trans(mask))
        tl.store(v_out_ptr + qv_offsets, v_values, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"expected shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


def _resolve_shape(value: Any, expected_numel: int) -> tuple[int, ...]:
    dims = list(_shape_tuple(value))
    neg_one_count = dims.count(-1)
    if neg_one_count > 1:
        raise ValueError(f"only one inferred dimension is valid, got {dims}")
    if neg_one_count == 1:
        known = 1
        for dim in dims:
            if dim != -1:
                known *= dim
        if known == 0 or expected_numel % known != 0:
            raise ValueError(f"cannot infer shape {dims} for numel={expected_numel}")
        dims[dims.index(-1)] = expected_numel // known
    resolved = tuple(dims)
    if _numel(resolved) != expected_numel:
        raise ValueError(
            f"shape {resolved} has {_numel(resolved)} elements, expected {expected_numel}"
        )
    return resolved


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], int, int, int, int, int, int]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects eight inputs, got {len(inputs)}")

    convolution_53, shape0, shape1, shape2, shape3, shape4, shape5, shape6 = inputs
    if not isinstance(convolution_53, torch.Tensor):
        raise TypeError(f"{REPRO_ID} first input must be a tensor")
    if not convolution_53.is_cuda:
        raise ValueError(f"{REPRO_ID} expects a CUDA input")
    if convolution_53.dtype is not torch.float32:
        raise ValueError(f"{REPRO_ID} expects torch.float32 input, got {convolution_53.dtype}")
    if convolution_53.ndim != 4:
        raise ValueError(f"{REPRO_ID} expects rank-4 input, got shape={tuple(convolution_53.shape)}")
    if convolution_53.stride(1) != 1:
        raise ValueError(f"{REPRO_ID} expects channel-contiguous input, got stride={tuple(convolution_53.stride())}")

    b, qkv, heads, d_model, p_flat = _resolve_shape(shape0, int(convolution_53.numel()))
    if qkv != 3:
        raise ValueError(f"{REPRO_ID} expects a QKV dimension of 3, got {qkv}")
    if tuple(convolution_53.shape) != (b, qkv * heads * d_model, convolution_53.shape[2], convolution_53.shape[3]):
        raise ValueError(
            f"{REPRO_ID} input channel count does not match view shape: "
            f"input={tuple(convolution_53.shape)}, view={(b, qkv, heads, d_model, p_flat)}"
        )
    if convolution_53.shape[2] * convolution_53.shape[3] != p_flat:
        raise ValueError(
            f"{REPRO_ID} input spatial numel {convolution_53.shape[2] * convolution_53.shape[3]} "
            f"does not match P={p_flat}"
        )
    if convolution_53.stride(2) != convolution_53.shape[3] * convolution_53.stride(3):
        raise ValueError(f"{REPRO_ID} expects a flattenable spatial layout, got stride={tuple(convolution_53.stride())}")

    slice_numel = b * heads * d_model * p_flat
    q_expand_shape = _resolve_shape(shape1, slice_numel)
    q_shape = _resolve_shape(shape2, slice_numel)
    k_expand_shape = _resolve_shape(shape3, slice_numel)
    k_shape = _resolve_shape(shape4, slice_numel)
    v_expand_shape = _resolve_shape(shape5, slice_numel)
    v_shape = _resolve_shape(shape6, slice_numel)

    expected_qv_expand = (b, heads, p_flat, d_model)
    expected_k_expand = (b, heads, d_model, p_flat)
    expected_qv_shape = (b * heads, p_flat, d_model)
    expected_k_shape = (b * heads, d_model, p_flat)
    if q_expand_shape != expected_qv_expand or v_expand_shape != expected_qv_expand:
        raise ValueError(f"unexpected Q/V expand shapes {q_expand_shape} and {v_expand_shape}")
    if k_expand_shape != expected_k_expand:
        raise ValueError(f"unexpected K expand shape {k_expand_shape}")
    if q_shape != expected_qv_shape or v_shape != expected_qv_shape:
        raise ValueError(f"unexpected Q/V output shapes {q_shape} and {v_shape}")
    if k_shape != expected_k_shape:
        raise ValueError(f"unexpected K output shape {k_shape}")

    return (
        convolution_53,
        q_shape,
        k_shape,
        v_shape,
        b * heads,
        heads,
        d_model,
        p_flat,
        int(convolution_53.stride(0)),
        int(convolution_53.stride(3)),
    )


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
        raise RuntimeError("Triton is required for this oracle")

    (
        convolution_53,
        q_shape,
        k_shape,
        v_shape,
        rows,
        heads,
        d_model,
        p_flat,
        batch_stride,
        pos_stride,
    ) = _validate_inputs(inputs)

    q_out = torch.empty(q_shape, device=convolution_53.device, dtype=convolution_53.dtype)
    k_out = torch.empty(k_shape, device=convolution_53.device, dtype=convolution_53.dtype)
    v_out = torch.empty(v_shape, device=convolution_53.device, dtype=convolution_53.dtype)

    grid = lambda meta: (
        rows,
        triton.cdiv(d_model, meta["BLOCK_D"]),
        triton.cdiv(p_flat, meta["BLOCK_P"]),
    )
    _qkv_channels_last_layout_kernel[grid](
        convolution_53,
        q_out,
        k_out,
        v_out,
        ROWS=rows,
        HEADS=heads,
        D=d_model,
        P=p_flat,
        BATCH_STRIDE=batch_stride,
        POS_STRIDE=pos_stride,
    )
    return (q_out, k_out, v_out)


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
