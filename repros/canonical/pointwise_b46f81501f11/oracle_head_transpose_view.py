"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete ALBERT attention layout scope by directly materializing the fresh contiguous `[8,512,64,64]` clone storage from the contiguous `[512,512,64]` input with a tiled head/sequence copy and returning the final `[4096,4096]` transposed view with stride `(1,4096)`, whereas Inductor lowers the captured view/permute/clone/view/permute chain through generic layout materialization that is slower under the shared CUDAGraph harness; Inductor cannot do this today because it does not recognize this head/sequence transpose clone followed by a metadata-only 2D transpose as one specialized layout-copy pattern; the fix is NEW_PATTERN: add a guarded lowering for `view(B,H,S,D).permute(0,2,1,3).contiguous().view(B*S,H*D).t()` that emits the direct clone-storage copy and preserves the final view metadata."""
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

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SEQ": 8}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SEQ": 16}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SEQ": 32}, num_warps=8, num_stages=3),
        ],
        key=["H", "S", "D"],
    )
    @triton.jit
    def _head_sequence_clone_kernel(
        input_ptr,
        clone_ptr,
        H: tl.constexpr,
        S: tl.constexpr,
        D: tl.constexpr,
        BLOCK_SEQ: tl.constexpr,
    ):
        token = tl.program_id(0)
        seq = tl.program_id(1) * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
        dim = tl.arange(0, D)
        head = token % H
        batch = token // H

        input_offsets = (token * S + seq[:, None]) * D + dim[None, :]
        clone_offsets = ((batch * S + seq[:, None]) * H + head) * D + dim[None, :]
        mask = seq[:, None] < S
        values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        tl.store(clone_ptr + clone_offsets, values, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"expected shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _numel(shape: tuple[int, ...]) -> int:
    product = 1
    for dim in shape:
        product *= dim
    return product


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
    resolved_numel = _numel(resolved)
    if resolved_numel != numel:
        raise ValueError(f"shape {resolved} has {resolved_numel} elements, expected {numel}")
    return resolved


def _validate_and_layout(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, tuple[int, int, int, int], tuple[int, int], int, int, int, int]:
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects four inputs, got {len(inputs)}")

    bmm_40, shape0, shape1, shape2 = inputs
    if not isinstance(bmm_40, torch.Tensor):
        raise TypeError(f"{REPRO_ID} first input must be a tensor")
    if not bmm_40.is_cuda:
        raise ValueError(f"{REPRO_ID} expects a CUDA input")
    if bmm_40.dtype is not torch.float32:
        raise ValueError(f"{REPRO_ID} expects torch.float32 input, got {bmm_40.dtype}")
    if bmm_40.ndim != 3:
        raise ValueError(f"{REPRO_ID} expects a rank-3 input, got shape={tuple(bmm_40.shape)}")
    if not bmm_40.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects contiguous input, got stride={tuple(bmm_40.stride())}")

    numel = int(bmm_40.numel())
    batch, heads, seq, head_dim = _resolve_view_shape(shape0, numel)
    if (batch * heads, seq, head_dim) != tuple(int(dim) for dim in bmm_40.shape):
        raise ValueError(
            f"first view shape {(batch, heads, seq, head_dim)} is incompatible "
            f"with input shape={tuple(bmm_40.shape)}"
        )

    batch1, seq1, hidden = _resolve_view_shape(shape1, numel)
    if (batch1, seq1, hidden) != (batch, seq, heads * head_dim):
        raise ValueError(
            f"second view shape {(batch1, seq1, hidden)} is incompatible with "
            f"permuted clone shape={(batch, seq, heads, head_dim)}"
        )

    final_shape = _resolve_view_shape(shape2, numel)
    if final_shape != (hidden, batch * seq):
        raise ValueError(f"unexpected final shape {final_shape}, expected {(hidden, batch * seq)}")

    clone_shape = (batch, seq, heads, head_dim)
    return bmm_40, clone_shape, final_shape, numel, heads, seq, head_dim


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

    bmm_40, clone_shape, final_shape, n_elements, heads, seq, head_dim = _validate_and_layout(inputs)
    clone_storage = torch.empty(
        clone_shape,
        device=bmm_40.device,
        dtype=bmm_40.dtype,
    )

    token_count = int(bmm_40.shape[0])
    grid = lambda meta: (token_count, triton.cdiv(seq, meta["BLOCK_SEQ"]))
    _head_sequence_clone_kernel[grid](
        bmm_40,
        clone_storage,
        H=heads,
        S=seq,
        D=head_dim,
    )
    return clone_storage.view(final_shape).permute(1, 0)


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
