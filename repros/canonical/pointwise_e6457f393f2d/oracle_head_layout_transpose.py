"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Swin attention projection layout scope by directly materializing the required fresh contiguous `[B, H, N, D]` clone buffer from the contiguous `[B*N, H*D]` input with one Triton head-layout transpose copy and returning the final `[B*H, N, D]` view, whereas Inductor lowers the captured reshape/permute/clone/reshape chain through generic layout materialization; Inductor cannot materially improve this isolated repro today because the output contract requires a non-aliasing contiguous clone, so the dominant full read plus full write traffic is mandatory and the compiled path already measures at the same CUDAGraph floor; the fix is BANDWIDTH_BOUND: record this as an at-floor layout-copy case unless broader layout-copy throughput or launch overhead work moves both paths together."""
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
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
        ],
        key=["N", "S", "H", "D"],
    )
    @triton.jit
    def _head_layout_transpose_flat_kernel(
        input_ptr,
        output_ptr,
        N: tl.constexpr,
        S: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        dim = offsets % D
        tmp = offsets // D
        head = tmp % H
        tmp = tmp // H
        seq = tmp % S
        batch = tmp // S

        output_offsets = ((batch * H + head) * S + seq) * D + dim
        values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        tl.store(output_ptr + output_offsets, values, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"expected shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


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
    resolved = tuple(dims)
    product = _numel(resolved)
    if product != numel:
        raise ValueError(f"shape {resolved} has {product} elements, expected {numel}")
    return resolved


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride: list[int] = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


def _validate_and_layout(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, tuple[int, int, int, int], tuple[int, int, int], int, int, int, int]:
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects four inputs, got {len(inputs)}")

    mm_199, shape0, shape1, shape2 = inputs
    if not isinstance(mm_199, torch.Tensor):
        raise TypeError(f"{REPRO_ID} first input must be a tensor")
    if not mm_199.is_cuda:
        raise ValueError(f"{REPRO_ID} expects a CUDA input")
    if mm_199.dtype is not torch.float32:
        raise ValueError(f"{REPRO_ID} expects torch.float32 input, got {mm_199.dtype}")
    if mm_199.ndim != 2:
        raise ValueError(f"{REPRO_ID} expects a rank-2 input, got shape={tuple(mm_199.shape)}")
    if not mm_199.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects contiguous input, got stride={tuple(mm_199.stride())}")

    input_rows = int(mm_199.shape[0])
    hidden = int(mm_199.shape[1])
    numel = int(mm_199.numel())

    batch, seq, shape_hidden = _resolve_view_shape(shape0, numel)
    if (batch * seq, shape_hidden) != (input_rows, hidden):
        raise ValueError(
            f"first reshape {(batch, seq, shape_hidden)} does not match input "
            f"shape={tuple(mm_199.shape)}"
        )

    batch1, seq1, heads, head_dim = _resolve_view_shape(shape1, numel)
    if (batch1, seq1) != (batch, seq) or heads * head_dim != hidden:
        raise ValueError(
            f"head reshape {(batch1, seq1, heads, head_dim)} is incompatible "
            f"with first reshape {(batch, seq, hidden)}"
        )

    output_shape = _resolve_view_shape(shape2, numel)
    expected_output_shape = (batch * heads, seq, head_dim)
    if output_shape != expected_output_shape:
        raise ValueError(f"unexpected output shape {output_shape}, expected {expected_output_shape}")

    clone_shape = (batch, heads, seq, head_dim)
    total_elements = batch * seq * heads * head_dim
    return mm_199, clone_shape, output_shape, total_elements, seq, heads, head_dim


@oracle_impl(hardware="H100", shapes="(T([401408, 128], f32), S([8192, 49, 128]), S([8192, 49, 4, 32]), S([32768, 49, 32]))")
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

    mm_199, clone_shape, output_shape, total_elements, seq, heads, head_dim = _validate_and_layout(inputs)
    clone_buffer = torch.empty_strided(
        clone_shape,
        _contiguous_stride(clone_shape),
        device=mm_199.device,
        dtype=mm_199.dtype,
    )

    grid = lambda meta: (triton.cdiv(total_elements, meta["BLOCK_SIZE"]),)
    _head_layout_transpose_flat_kernel[grid](
        mm_199,
        clone_buffer,
        N=total_elements,
        S=seq,
        H=heads,
        D=head_dim,
    )
    return clone_buffer.reshape(output_shape)


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
