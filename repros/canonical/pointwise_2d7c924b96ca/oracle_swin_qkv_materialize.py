"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Swin QKV layout materialization returned by Repro.forward, including the `[128,49,3,32,32]` view, unbind, Q scale, Q/K/V clone materializations, K transpose, and final contiguous `[4096,49,32]`, `[4096,32,49]`, `[4096,49,32]` outputs in one tiled Triton transpose/copy kernel, whereas Inductor lowers the decomposed reshape/permute/unbind/scale/expand/clone/reshape graph as separate generic layout-copy regions around the pointwise scale; Inductor cannot do this today because the scheduler does not fuse sibling clone/materialization outputs from one QKV producer when one branch includes a pointwise scale and another branch needs a tiled transpose; the fix is SCHEDULER_FUSION: teach layout-copy scheduling to group QKV split materializations and sink simple per-branch pointwise epilogues into one multi-output tiled copy."""
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
            triton.Config({"NBLOCK": 8, "DBLOCK": 32}, num_warps=4, num_stages=3),
            triton.Config({"NBLOCK": 16, "DBLOCK": 32}, num_warps=4, num_stages=3),
            triton.Config({"NBLOCK": 32, "DBLOCK": 32}, num_warps=4, num_stages=3),
            triton.Config({"NBLOCK": 64, "DBLOCK": 32}, num_warps=8, num_stages=3),
        ],
        key=["ROWS", "N", "D", "H"],
    )
    @triton.jit
    def _swin_qkv_materialize_kernel(
        input_ptr,
        q_output_ptr,
        k_output_ptr,
        v_output_ptr,
        ROWS: tl.constexpr,
        N: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        NBLOCK: tl.constexpr,
        DBLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        n_start = tl.program_id(1) * NBLOCK
        d_start = tl.program_id(2) * DBLOCK

        n_offsets = n_start + tl.arange(0, NBLOCK)[:, None]
        d_offsets = d_start + tl.arange(0, DBLOCK)[None, :]
        mask = (row < ROWS) & (n_offsets < N) & (d_offsets < D)

        batch = row // H
        head = row - batch * H
        source_base = (((batch * N + n_offsets) * 3) * H + head) * D + d_offsets
        q_values = tl.load(input_ptr + source_base, mask=mask, other=0.0)
        k_values = tl.load(input_ptr + source_base + H * D, mask=mask, other=0.0)
        v_values = tl.load(input_ptr + source_base + 2 * H * D, mask=mask, other=0.0)

        qv_offsets = row * N * D + n_offsets * D + d_offsets
        tl.store(q_output_ptr + qv_offsets, q_values * 0.1767766952966369, mask=mask)
        tl.store(v_output_ptr + qv_offsets, v_values, mask=mask)

        k_d_offsets = d_start + tl.arange(0, DBLOCK)[:, None]
        k_n_offsets = n_start + tl.arange(0, NBLOCK)[None, :]
        k_mask = (row < ROWS) & (k_d_offsets < D) & (k_n_offsets < N)
        k_offsets = row * D * N + k_d_offsets * N + k_n_offsets
        tl.store(k_output_ptr + k_offsets, tl.trans(k_values), mask=k_mask)


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
) -> tuple[
    torch.Tensor,
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int, int],
    int,
    int,
    int,
    int,
]:
    if len(inputs) != 9:
        raise ValueError(f"{REPRO_ID} expects nine inputs, got {len(inputs)}")

    (
        addmm_92,
        shape0,
        shape1,
        q_expand_shape,
        q_output_shape,
        k_expand_shape,
        k_output_shape,
        v_expand_shape,
        v_output_shape,
    ) = inputs
    if not isinstance(addmm_92, torch.Tensor):
        raise TypeError(f"{REPRO_ID} first input must be a tensor")
    if not addmm_92.is_cuda:
        raise ValueError(f"{REPRO_ID} expects a CUDA input")
    if addmm_92.dtype is not torch.float32:
        raise TypeError(f"{REPRO_ID} expects torch.float32 input, got {addmm_92.dtype}")
    if addmm_92.ndim != 2:
        raise ValueError(f"{REPRO_ID} expects rank-2 input, got shape={tuple(addmm_92.shape)}")
    if not addmm_92.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects contiguous input, got stride={tuple(addmm_92.stride())}")

    input_numel = int(addmm_92.numel())
    batch, seq, hidden = _resolve_view_shape(shape0, input_numel)
    if (batch * seq, hidden) != tuple(addmm_92.shape):
        raise ValueError(
            f"first view shape {(batch, seq, hidden)} is incompatible with "
            f"input shape={tuple(addmm_92.shape)}"
        )

    batch1, seq1, qkv, heads, head_dim = _resolve_view_shape(shape1, input_numel)
    if (batch1, seq1) != (batch, seq) or qkv != 3 or heads * head_dim * 3 != hidden:
        raise ValueError(
            f"QKV view shape {(batch1, seq1, qkv, heads, head_dim)} is incompatible "
            f"with first view {(batch, seq, hidden)}"
        )

    slice_numel = batch * heads * seq * head_dim
    expected_qv_expand = (batch, heads, seq, head_dim)
    expected_qv_output = (batch * heads, seq, head_dim)
    expected_k_expand = (batch, heads, head_dim, seq)
    expected_k_output = (batch * heads, head_dim, seq)

    q_expand = _resolve_view_shape(q_expand_shape, slice_numel)
    q_output = _resolve_view_shape(q_output_shape, slice_numel)
    k_expand = _resolve_view_shape(k_expand_shape, slice_numel)
    k_output = _resolve_view_shape(k_output_shape, slice_numel)
    v_expand = _resolve_view_shape(v_expand_shape, slice_numel)
    v_output = _resolve_view_shape(v_output_shape, slice_numel)
    if q_expand != expected_qv_expand or v_expand != expected_qv_expand:
        raise ValueError(f"unexpected Q/V expand shapes {q_expand} and {v_expand}")
    if q_output != expected_qv_output or v_output != expected_qv_output:
        raise ValueError(f"unexpected Q/V output shapes {q_output} and {v_output}")
    if k_expand != expected_k_expand:
        raise ValueError(f"unexpected K expand shape {k_expand}")
    if k_output != expected_k_output:
        raise ValueError(f"unexpected K output shape {k_output}")

    return addmm_92, q_output, k_output, v_output, batch * heads, seq, heads, head_dim


@oracle_impl(hardware="H100", shapes="(T([6272, 3072], f32), S([128, 49, 3072]), S([128, 49, 3, 32, -1]), S([128, 32, 49, 32]), S([4096, 49, 32]), S([128, 32, 32, 49]), S([4096, 32, 49]), S([128, 32, 49, 32]), S([4096, 49, 32]))")
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

    addmm_92, q_shape, k_shape, v_shape, rows, seq, heads, head_dim = _validate_layout(inputs)
    q_output = torch.empty(q_shape, device=addmm_92.device, dtype=addmm_92.dtype)
    k_output = torch.empty(k_shape, device=addmm_92.device, dtype=addmm_92.dtype)
    v_output = torch.empty(v_shape, device=addmm_92.device, dtype=addmm_92.dtype)

    grid = lambda meta: (
        rows,
        triton.cdiv(seq, meta["NBLOCK"]),
        triton.cdiv(head_dim, meta["DBLOCK"]),
    )
    _swin_qkv_materialize_kernel[grid](
        addmm_92,
        q_output,
        k_output,
        v_output,
        ROWS=rows,
        N=seq,
        H=heads,
        D=head_dim,
    )
    return (q_output, k_output, v_output)


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
