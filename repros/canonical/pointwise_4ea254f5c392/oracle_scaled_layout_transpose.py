"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full Albert attention-score scale plus head-layout materialization scope by writing the required fresh contiguous `[B*S, H*D]` clone storage with one Triton kernel and returning the same `[H*D, B*S]` transpose view, whereas Inductor already lowers the view/mul/permute/clone/view/permute chain to a fused pointwise layout copy with only generic index-decoding overhead; Inductor cannot remove the dominant work today because the repro contract requires reading every input element, multiplying it, and materializing fresh transposed clone storage, so the fix is BANDWIDTH_BOUND: record this as at floor unless broader layout-copy bandwidth or index-specialization work moves both implementations."""
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
            triton.Config({"YBLOCK": 8}, num_warps=4, num_stages=3),
            triton.Config({"YBLOCK": 16}, num_warps=4, num_stages=3),
            triton.Config({"YBLOCK": 32}, num_warps=8, num_stages=3),
            triton.Config({"YBLOCK": 64}, num_warps=8, num_stages=3),
        ],
        key=["TOKENS", "HEADS", "SEQ", "D"],
    )
    @triton.jit
    def _scaled_layout_transpose_kernel(
        input_ptr,
        base_ptr,
        TOKENS: tl.constexpr,
        HEADS: tl.constexpr,
        SEQ: tl.constexpr,
        D: tl.constexpr,
        YBLOCK: tl.constexpr,
    ):
        token = tl.program_id(1) * YBLOCK + tl.arange(0, YBLOCK)[:, None]
        dim = tl.arange(0, D)[None, :]
        head = tl.program_id(0)
        mask = token < TOKENS

        batch = token // SEQ
        seq = token - batch * SEQ
        hidden = HEADS * D

        input_offsets = ((batch * HEADS + head) * SEQ + seq) * D + dim
        base_offsets = token * hidden + head * D + dim
        values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        tl.store(base_ptr + base_offsets, values * 0.3535533905932738, mask=mask)


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
    if _numel(resolved) != numel:
        raise ValueError(f"shape {resolved} has {_numel(resolved)} elements, expected {numel}")
    return resolved


def _validate_layout(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, tuple[int, int], int, int, int, int]:
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects four inputs, got {len(inputs)}")

    bmm_43, shape0, shape1, shape2 = inputs
    if not isinstance(bmm_43, torch.Tensor):
        raise TypeError(f"{REPRO_ID} first input must be a tensor")
    if not bmm_43.is_cuda:
        raise ValueError(f"{REPRO_ID} expects a CUDA input")
    if bmm_43.dtype is not torch.float32:
        raise ValueError(f"{REPRO_ID} expects torch.float32 input, got {bmm_43.dtype}")
    if bmm_43.ndim != 3:
        raise ValueError(f"{REPRO_ID} expects rank-3 input, got shape={tuple(bmm_43.shape)}")
    if not bmm_43.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects contiguous input, got stride={tuple(bmm_43.stride())}")

    numel = int(bmm_43.numel())
    batch, heads, seq, dim = _resolve_view_shape(shape0, numel)
    if tuple(bmm_43.shape) != (batch * heads, seq, dim):
        raise ValueError(
            f"first view shape {(batch, heads, seq, dim)} is incompatible "
            f"with input shape={tuple(bmm_43.shape)}"
        )
    if dim != 64:
        raise ValueError(f"{REPRO_ID} expects head dim 64 for this captured pattern, got {dim}")

    shape1_resolved = _resolve_view_shape(shape1, numel)
    expected_shape1 = (batch, seq, heads * dim)
    if shape1_resolved != expected_shape1:
        raise ValueError(f"unexpected intermediate shape {shape1_resolved}, expected {expected_shape1}")

    base_shape = _resolve_view_shape(shape2, numel)
    expected_base_shape = (batch * seq, heads * dim)
    if base_shape != expected_base_shape:
        raise ValueError(f"unexpected base shape {base_shape}, expected {expected_base_shape}")

    return bmm_43, base_shape, batch * seq, heads, seq, dim


def _check_output_layout(output: torch.Tensor, base_shape: tuple[int, int]) -> bool:
    tokens, hidden = base_shape
    return (
        tuple(output.shape) == (hidden, tokens)
        and tuple(output.stride()) == (1, hidden)
        and output.storage_offset() == 0
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

    bmm_43, base_shape, tokens, heads, seq, dim = _validate_layout(inputs)
    base = torch.empty_strided(
        base_shape,
        (base_shape[1], 1),
        device=bmm_43.device,
        dtype=bmm_43.dtype,
    )
    grid = lambda meta: (heads, triton.cdiv(tokens, meta["YBLOCK"]))
    _scaled_layout_transpose_kernel[grid](
        bmm_43,
        base,
        TOKENS=tokens,
        HEADS=heads,
        SEQ=seq,
        D=dim,
    )
    return base.t()


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
        _, base_shape, _, _, _, _ = _validate_layout(inputs)
        with torch.no_grad():
            layout_output = oracle_forward(inputs)
            if layout_output.is_cuda:
                torch.cuda.synchronize()
        layout_ok = _check_output_layout(layout_output, base_shape)
        print(
            f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(layout_output.shape)} stride={list(layout_output.stride())})"
        )
        ok = ok and layout_ok
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
