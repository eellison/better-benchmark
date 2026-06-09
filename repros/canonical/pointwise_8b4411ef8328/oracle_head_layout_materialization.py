"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete GPT-Neo attention-key layout materialization by writing the fresh contiguous `[B*H, D, S]` clone/view output directly from the contiguous `[B*S, H*D]` projection with one tiled Triton transpose-copy, whereas Inductor lowers the captured view/view/permute/permute/expand/clone/view chain through generic layout materialization; Inductor cannot do this today because its pointwise/layout codegen does not recognize the double-permute attention head split as a specialized transpose over `[S, D]` tiles, so the fix is NEW_PATTERN: add a guarded head-layout materialization template for `view(B,S,H,D).permute(0,2,3,1).clone().view(B*H,D,S)`."""
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
            triton.Config({"SBLOCK": 16, "DBLOCK": 32}, num_warps=4, num_stages=3),
            triton.Config({"SBLOCK": 32, "DBLOCK": 32}, num_warps=4, num_stages=3),
            triton.Config({"SBLOCK": 16, "DBLOCK": 64}, num_warps=4, num_stages=3),
            triton.Config({"SBLOCK": 64, "DBLOCK": 16}, num_warps=4, num_stages=3),
            triton.Config({"SBLOCK": 32, "DBLOCK": 64}, num_warps=8, num_stages=3),
            triton.Config({"SBLOCK": 64, "DBLOCK": 32}, num_warps=8, num_stages=3),
        ],
        key=["B", "S", "H", "D"],
    )
    @triton.jit
    def _head_layout_materialization_kernel(
        input_ptr,
        output_ptr,
        B: tl.constexpr,
        S: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        SBLOCK: tl.constexpr,
        DBLOCK: tl.constexpr,
    ):
        bh = tl.program_id(0)
        s_offsets = tl.program_id(1) * SBLOCK + tl.arange(0, SBLOCK)[:, None]
        d_offsets = tl.program_id(2) * DBLOCK + tl.arange(0, DBLOCK)[None, :]
        load_mask = (s_offsets < S) & (d_offsets < D)

        batch = bh // H
        head = bh - batch * H
        values = tl.load(
            input_ptr + (batch * S + s_offsets) * (H * D) + head * D + d_offsets,
            mask=load_mask,
            other=0.0,
        )

        d_store = tl.program_id(2) * DBLOCK + tl.arange(0, DBLOCK)[:, None]
        s_store = tl.program_id(1) * SBLOCK + tl.arange(0, SBLOCK)[None, :]
        store_mask = (d_store < D) & (s_store < S)
        tl.store(
            output_ptr + bh * (D * S) + d_store * S + s_store,
            tl.trans(values),
            mask=store_mask,
        )


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple, torch.Size)):
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
    if _numel(resolved) != numel:
        raise ValueError(f"shape {resolved} has {_numel(resolved)} elements, expected {numel}")
    return resolved


def _validate_layout(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, tuple[int, int, int], tuple[int, int, int], int, int, int, int]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    mm_70, shape0, shape1, shape2, shape3 = inputs
    if not isinstance(mm_70, torch.Tensor):
        raise TypeError(f"{REPRO_ID} first input must be a tensor")
    if not mm_70.is_cuda:
        raise ValueError(f"{REPRO_ID} expects a CUDA input")
    if mm_70.dtype is not torch.float32:
        raise ValueError(f"{REPRO_ID} expects torch.float32 input, got {mm_70.dtype}")
    if mm_70.ndim != 2:
        raise ValueError(f"{REPRO_ID} expects a rank-2 input, got shape={tuple(mm_70.shape)}")
    if not mm_70.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects contiguous input, got stride={tuple(mm_70.stride())}")

    rows = int(mm_70.shape[0])
    hidden = int(mm_70.shape[1])
    total = int(mm_70.numel())

    batch, seq, view_hidden = _resolve_view_shape(shape0, total)
    if (batch * seq, view_hidden) != (rows, hidden):
        raise ValueError(
            f"first view shape {(batch, seq, view_hidden)} does not match input "
            f"shape={tuple(mm_70.shape)}"
        )

    batch1, seq1, heads, head_dim = _resolve_view_shape(shape1, total)
    if (batch1, seq1) != (batch, seq) or heads * head_dim != hidden:
        raise ValueError(
            f"head view shape {(batch1, seq1, heads, head_dim)} is incompatible "
            f"with first view {(batch, seq, hidden)}"
        )

    clone_shape = _resolve_view_shape(shape2, total)
    expected_clone_shape = (batch, heads, head_dim, seq)
    if clone_shape != expected_clone_shape:
        raise ValueError(f"unexpected clone shape {clone_shape}, expected {expected_clone_shape}")

    output_shape = _resolve_view_shape(shape3, total)
    expected_output_shape = (batch * heads, head_dim, seq)
    if output_shape != expected_output_shape:
        raise ValueError(f"unexpected output shape {output_shape}, expected {expected_output_shape}")

    output_stride = (head_dim * seq, seq, 1)
    return mm_70, output_shape, output_stride, batch, seq, heads, head_dim


@oracle_impl(hardware="H100", shapes="(T([4096, 2048], f32), S([32, 128, 2048]), S([32, 128, 16, 128]), S([32, 16, 128, 128]), S([512, 128, 128]))")
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

    mm_70, output_shape, output_stride, batch, seq, heads, head_dim = _validate_layout(inputs)
    output = torch.empty_strided(
        output_shape,
        output_stride,
        device=mm_70.device,
        dtype=mm_70.dtype,
    )
    grid = lambda meta: (
        batch * heads,
        triton.cdiv(seq, meta["SBLOCK"]),
        triton.cdiv(head_dim, meta["DBLOCK"]),
    )
    _head_layout_materialization_kernel[grid](
        mm_70,
        output,
        B=batch,
        S=seq,
        H=heads,
        D=head_dim,
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
