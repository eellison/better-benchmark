"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Swin attention output layout scope by directly materializing the required fresh contiguous `[B*S, H*D]` tensor from the contiguous `[B*H, S, D]` input with one Triton head/sequence transpose copy, whereas Inductor already lowers the captured reshape/permute/clone/reshape chain to a single fused layout-copy kernel; Inductor cannot remove the dominant read/write traffic today because the repro contract requires a newly materialized contiguous clone with this transposed layout; the fix is BANDWIDTH_BOUND: there is no algorithmic Inductor fix for this isolated scope beyond minor index-code or autotuning polish."""
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

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
        ],
        key=["N", "S", "H", "D"],
    )
    @triton.jit
    def _swin_attention_layout_kernel(
        input_ptr,
        output_ptr,
        N: tl.constexpr,
        S: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

        dim = offsets % D
        tmp = offsets // D
        head = tmp % H
        tmp = tmp // H
        seq = tmp % S
        batch = tmp // S

        input_offsets = ((batch * H + head) * S + seq) * D + dim
        values = tl.load(input_ptr + input_offsets)
        tl.store(output_ptr + offsets, values)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple, torch.Size)):
        raise TypeError(f"expected shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _resolve_view_shape(value: Any, numel: int) -> tuple[int, ...]:
    dims = list(_shape_tuple(value))
    neg_one_count = dims.count(-1)
    if neg_one_count > 1:
        raise ValueError(f"only one inferred dimension is valid, got {dims}")
    if neg_one_count == 1:
        known = 1
        for dim in dims:
            if dim != -1:
                known *= dim
        if known == 0 or numel % known != 0:
            raise ValueError(f"cannot infer shape {dims} for numel={numel}")
        dims[dims.index(-1)] = numel // known

    product = 1
    for dim in dims:
        product *= dim
    if product != numel:
        raise ValueError(f"shape {tuple(dims)} has {product} elements, expected {numel}")
    return tuple(dims)


def _validate_layout(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int], int, int, int, int]:
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects four inputs, got {len(inputs)}")

    bmm_47, shape0, shape1, shape2 = inputs
    if not isinstance(bmm_47, torch.Tensor):
        raise TypeError(f"{REPRO_ID} first input must be a tensor")
    if not bmm_47.is_cuda:
        raise ValueError(f"{REPRO_ID} expects a CUDA input")
    if bmm_47.dtype is not torch.float32:
        raise ValueError(f"{REPRO_ID} expects torch.float32 input, got {bmm_47.dtype}")
    if bmm_47.ndim != 3:
        raise ValueError(f"{REPRO_ID} expects rank-3 input, got shape={tuple(bmm_47.shape)}")
    if not bmm_47.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects contiguous input, got stride={tuple(bmm_47.stride())}")

    numel = int(bmm_47.numel())
    batch, heads, seq, dim = _resolve_view_shape(shape0, numel)
    if tuple(bmm_47.shape) != (batch * heads, seq, dim):
        raise ValueError(
            f"first view shape {(batch, heads, seq, dim)} is incompatible "
            f"with input shape={tuple(bmm_47.shape)}"
        )

    middle_shape = _resolve_view_shape(shape1, numel)
    expected_middle = (batch, seq, heads * dim)
    if middle_shape != expected_middle:
        raise ValueError(f"unexpected middle shape {middle_shape}, expected {expected_middle}")

    output_shape = _resolve_view_shape(shape2, numel)
    expected_output = (batch * seq, heads * dim)
    if output_shape != expected_output:
        raise ValueError(f"unexpected output shape {output_shape}, expected {expected_output}")

    output_stride = (heads * dim, 1)
    total_vectors = batch * seq * heads
    if (total_vectors * dim) % 256 != 0:
        raise ValueError(f"{REPRO_ID} expects full-scope element count divisible by 256")
    return bmm_47, output_shape, output_stride, total_vectors, seq, heads, dim


@oracle_impl(hardware="H100", shapes="(T([4096, 49, 32], f32), S([128, 32, 49, 32]), S([128, 49, 1024]), S([6272, 1024]))")
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

    bmm_47, output_shape, output_stride, total_vectors, seq, heads, dim = _validate_layout(inputs)
    output = torch.empty_strided(
        output_shape,
        output_stride,
        device=bmm_47.device,
        dtype=bmm_47.dtype,
    )

    n_elements = total_vectors * dim
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _swin_attention_layout_kernel[grid](
        bmm_47,
        output,
        N=n_elements,
        S=seq,
        H=heads,
        D=dim,
    )
    return output


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
