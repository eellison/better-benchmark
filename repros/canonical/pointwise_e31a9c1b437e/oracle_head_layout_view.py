"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle directly fills the fresh contiguous clone storage for the `view(B,S,H,D).permute(0,2,1,3).clone().view(B*H,S,D).permute(0,2,1)` attention-head layout and returns the same final non-contiguous view, whereas Inductor already lowers the captured view/permute/expand/clone/view/permute scope to a fused layout materialization; Inductor cannot materially improve this isolated repro today because the full-scope contract requires the dense f32 input read, the dense clone-storage write, and the returned view metadata, so the fix is BANDWIDTH_BOUND: record this case as at floor unless lower-level layout-copy codegen, launch overhead, or allocation behavior improves both paths."""
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


CLASSIFICATION = "BANDWIDTH_BOUND"


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
            triton.Config({"BLOCK_S": 4}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_S": 8}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_S": 16}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_S": 32}, num_warps=8, num_stages=3),
        ],
        key=["S", "H", "D"],
    )
    @triton.jit
    def _head_layout_clone_kernel(
        input_ptr,
        clone_ptr,
        S: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        head_batch = tl.program_id(0)
        seq_block = tl.program_id(1)

        batch = head_batch // H
        head = head_batch - batch * H
        seq = seq_block * BLOCK_S + tl.arange(0, BLOCK_S)[:, None]
        dim = tl.arange(0, D)[None, :]
        mask = seq < S

        input_offsets = ((batch * S + seq) * (H * D)) + head * D + dim
        clone_offsets = (head_batch * S + seq) * D + dim
        values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        tl.store(clone_ptr + clone_offsets, values, mask=mask)


def _shape_tuple(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple, torch.Size)):
        raise TypeError(f"{name} must be a shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _resolve_view_shape(value: Any, numel: int, name: str) -> tuple[int, ...]:
    dims = list(_shape_tuple(value, name))
    neg_one_count = dims.count(-1)
    if neg_one_count > 1:
        raise ValueError(f"{name} has more than one inferred dimension: {dims}")
    if neg_one_count == 1:
        known = 1
        for dim in dims:
            if dim != -1:
                known *= dim
        if known == 0 or numel % known != 0:
            raise ValueError(f"{name}={dims} cannot be inferred for numel={numel}")
        dims[dims.index(-1)] = numel // known

    resolved = tuple(dims)
    product = 1
    for dim in resolved:
        product *= dim
    if product != numel:
        raise ValueError(f"{name}={resolved} has {product} elements, expected {numel}")
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
) -> tuple[torch.Tensor, tuple[int, int, int, int], tuple[int, int, int], int, int, int]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    addmm_69, shape0, shape1, expand_shape_arg, view_shape_arg = inputs
    if not isinstance(addmm_69, torch.Tensor):
        raise TypeError(f"{REPRO_ID} first input must be a tensor")
    if not addmm_69.is_cuda:
        raise ValueError(f"{REPRO_ID} expects a CUDA input")
    if addmm_69.dtype is not torch.float32:
        raise TypeError(f"{REPRO_ID} expects torch.float32 input, got {addmm_69.dtype}")
    if addmm_69.ndim != 2:
        raise ValueError(f"{REPRO_ID} expects a rank-2 input, got {tuple(addmm_69.shape)}")
    if not addmm_69.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects contiguous input, got stride={tuple(addmm_69.stride())}")

    numel = int(addmm_69.numel())
    rows = int(addmm_69.shape[0])
    hidden = int(addmm_69.shape[1])

    batch, seq, shape_hidden = _resolve_view_shape(shape0, numel, "shape0")
    if (batch * seq, shape_hidden) != (rows, hidden):
        raise ValueError(
            f"shape0={(batch, seq, shape_hidden)} is incompatible with input "
            f"shape={tuple(addmm_69.shape)}"
        )

    batch1, seq1, heads, head_dim = _resolve_view_shape(shape1, numel, "shape1")
    if (batch1, seq1, heads * head_dim) != (batch, seq, hidden):
        raise ValueError(
            f"shape1={(batch1, seq1, heads, head_dim)} is incompatible with "
            f"shape0={(batch, seq, hidden)}"
        )

    clone_shape = _resolve_view_shape(expand_shape_arg, numel, "expand_shape")
    expected_clone_shape = (batch, heads, seq, head_dim)
    if clone_shape != expected_clone_shape:
        raise ValueError(f"expand_shape={clone_shape} does not match {expected_clone_shape}")

    view_shape = _resolve_view_shape(view_shape_arg, numel, "view_shape")
    expected_view_shape = (batch * heads, seq, head_dim)
    if view_shape != expected_view_shape:
        raise ValueError(f"view_shape={view_shape} does not match {expected_view_shape}")

    if head_dim != 64:
        raise ValueError(f"{REPRO_ID} Triton tile expects head_dim=64, got {head_dim}")

    return addmm_69, clone_shape, view_shape, seq, heads, head_dim


def oracle_forward(inputs):
    """Run the exact Repro.forward scope with direct clone-storage materialization.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_head_layout_view.py")

    addmm_69, clone_shape, view_shape, seq, heads, head_dim = _validate_and_layout(inputs)
    clone_base = torch.empty_strided(
        clone_shape,
        _contiguous_stride(clone_shape),
        device=addmm_69.device,
        dtype=addmm_69.dtype,
    )

    grid = lambda meta: (clone_shape[0] * heads, triton.cdiv(seq, meta["BLOCK_S"]))
    _head_layout_clone_kernel[grid](
        addmm_69,
        clone_base,
        S=seq,
        H=heads,
        D=head_dim,
    )
    return clone_base.view(view_shape).permute(0, 2, 1)


def _check_layout(instance: torch.nn.Module, inputs: list[Any] | tuple[Any, ...]) -> bool:
    with torch.no_grad():
        eager_out = instance(*inputs)
        oracle_out = oracle_forward(inputs)
        if oracle_out.is_cuda:
            torch.cuda.synchronize()

    layout_ok = (
        tuple(oracle_out.shape) == tuple(eager_out.shape)
        and oracle_out.dtype == eager_out.dtype
        and tuple(oracle_out.stride()) == tuple(eager_out.stride())
        and oracle_out.storage_offset() == eager_out.storage_offset()
        and oracle_out._base is not None
        and eager_out._base is not None
        and oracle_out.untyped_storage().data_ptr() != inputs[0].untyped_storage().data_ptr()
    )
    print(
        f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
        f"(shape={list(oracle_out.shape)} dtype={oracle_out.dtype} "
        f"stride={tuple(oracle_out.stride())} storage_offset={oracle_out.storage_offset()} "
        f"view_base={oracle_out._base is not None})"
    )
    return bool(layout_ok)


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
        ok = ok and _check_layout(instance, inputs)
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
