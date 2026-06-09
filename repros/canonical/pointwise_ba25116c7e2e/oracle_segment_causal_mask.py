"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete GPT-2 segment-aware causal attention-mask bias by materializing only the required fp32 `[32,1,512,512]` backing buffer and returning the exact `[32,12,512,512]` zero-stride head expand view, whereas Inductor lowers the paired advanced-index cumsum lookups, causal comparison, equality, where, and expand chain as generic pointwise work; Inductor cannot do this today because it has no shape-specialized segment-causal-mask pattern that recognizes the shared position vector, paired cumsum gathers, and metadata-only head expansion as one attention-mask idiom; the fix is NEW_PATTERN: add a guarded lowering/template for segment-aware causal mask construction that emits the compact base-buffer kernel and preserves the expanded output layout contract."""
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

BATCH = 32
HEADS = 12
SEQ = 512
BASE_SHAPE = (BATCH, 1, SEQ, SEQ)
BASE_STRIDE = (SEQ * SEQ, SEQ * SEQ, SEQ, 1)
OUTPUT_SHAPE = (BATCH, HEADS, SEQ, SEQ)
OUTPUT_STRIDE = (SEQ * SEQ, 0, SEQ, 1)
N_BASE_ELEMENTS = BATCH * SEQ * SEQ

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


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple, torch.Size)):
        raise TypeError(f"expected shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _resolve_expand_shape(
    shape_param: Any,
    source_shape: tuple[int, ...],
) -> tuple[int, ...]:
    dims = _shape_tuple(shape_param)
    if len(dims) != len(source_shape):
        raise ValueError(
            f"expand rank {len(dims)} does not match source rank {len(source_shape)}"
        )
    return tuple(source if dim == -1 else dim for dim, source in zip(dims, source_shape))


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects four inputs, got {len(inputs)}")

    positions, cumsum, shape_param_0, shape_param_1 = inputs
    if not isinstance(positions, torch.Tensor):
        raise TypeError(f"positions must be a tensor, got {type(positions)!r}")
    if not isinstance(cumsum, torch.Tensor):
        raise TypeError(f"cumsum must be a tensor, got {type(cumsum)!r}")

    if tuple(positions.shape) != (1, SEQ):
        raise ValueError(f"positions shape must be {(1, SEQ)}, got {tuple(positions.shape)}")
    if tuple(cumsum.shape) != (BATCH, SEQ):
        raise ValueError(f"cumsum shape must be {(BATCH, SEQ)}, got {tuple(cumsum.shape)}")
    if positions.dtype != torch.int64:
        raise TypeError(f"positions dtype must be torch.int64, got {positions.dtype}")
    if cumsum.dtype != torch.int64:
        raise TypeError(f"cumsum dtype must be torch.int64, got {cumsum.dtype}")
    if positions.device != cumsum.device:
        raise ValueError(f"positions and cumsum devices differ: {positions.device} vs {cumsum.device}")
    if tuple(positions.stride()) != (SEQ, 1):
        raise ValueError(f"positions stride must be {(SEQ, 1)}, got {tuple(positions.stride())}")
    if tuple(cumsum.stride()) != (SEQ, 1):
        raise ValueError(f"cumsum stride must be {(SEQ, 1)}, got {tuple(cumsum.stride())}")

    base_expand_shape = _resolve_expand_shape(shape_param_0, BASE_SHAPE)
    if base_expand_shape != BASE_SHAPE:
        raise ValueError(
            f"_shape_param_0 resolves to {base_expand_shape}, expected {BASE_SHAPE}"
        )

    output_shape = _resolve_expand_shape(shape_param_1, BASE_SHAPE)
    if output_shape != OUTPUT_SHAPE:
        raise ValueError(f"_shape_param_1 resolves to {output_shape}, expected {OUTPUT_SHAPE}")

    return positions, cumsum, output_shape


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_Q": 1, "BLOCK_K": 256}, num_warps=4),
            triton.Config({"BLOCK_Q": 1, "BLOCK_K": 512}, num_warps=8),
            triton.Config({"BLOCK_Q": 4, "BLOCK_K": 128}, num_warps=4),
            triton.Config({"BLOCK_Q": 8, "BLOCK_K": 128}, num_warps=4),
            triton.Config({"BLOCK_Q": 16, "BLOCK_K": 64}, num_warps=4),
            triton.Config({"BLOCK_Q": 16, "BLOCK_K": 128}, num_warps=8),
            triton.Config({"BLOCK_Q": 32, "BLOCK_K": 64}, num_warps=8),
        ],
        key=["SEQ_LEN"],
    )
    @triton.jit
    def _segment_causal_mask_kernel(
        positions_ptr,
        cumsum_ptr,
        out_ptr,
        SEQ_LEN: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        q_block = tl.program_id(0)
        k_block = tl.program_id(1)
        batch = tl.program_id(2)

        q_offsets = q_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
        k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        q_valid = q_offsets < SEQ_LEN
        k_valid = k_offsets < SEQ_LEN

        q_pos = tl.load(positions_ptr + q_offsets, mask=q_valid, other=0)
        k_pos = tl.load(positions_ptr + k_offsets, mask=k_valid, other=0)

        cumsum_base = cumsum_ptr + batch * SEQ_LEN
        q_segment = tl.load(cumsum_base + q_pos, mask=q_valid, other=-1)
        k_segment = tl.load(cumsum_base + k_pos, mask=k_valid, other=-2)

        keep = (k_pos[None, :] <= q_pos[:, None]) & (
            k_segment[None, :] == q_segment[:, None]
        )
        values = tl.where(keep, 0.0, -float("inf"))

        out_offsets = (
            batch * (SEQ_LEN * SEQ_LEN)
            + q_offsets[:, None] * SEQ_LEN
            + k_offsets[None, :]
        )
        tl.store(out_ptr + out_offsets, values, mask=q_valid[:, None] & k_valid[None, :])


@oracle_impl(hardware="H100", shapes="(T([1, 512], i64, gen=Index(512)), T([32, 512], i64, gen=Index(32)), S([32, -1, 512, 512]), S([32, 12, 512, 512]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]):
    """Run the full Repro.forward segment-aware causal mask computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    one fp32 `[32,12,512,512]` output with eager-compatible zero stride in the
    head dimension, storage offset 0, and a compact `[32,1,512,512]` backing
    storage.
    """
    if triton is None:
        raise RuntimeError("triton is required for this oracle")

    positions, cumsum, output_shape = _validate_inputs(inputs)
    if not positions.is_cuda:
        raise TypeError("this oracle expects CUDA inputs")

    base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=positions.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (
        triton.cdiv(SEQ, meta["BLOCK_Q"]),
        triton.cdiv(SEQ, meta["BLOCK_K"]),
        BATCH,
    )
    _segment_causal_mask_kernel[grid](
        positions,
        cumsum,
        base,
        SEQ_LEN=SEQ,
    )
    return base.expand(output_shape)


def check_output_layout(instance, inputs) -> bool:
    """The shared harness checks values/shapes; verify view metadata here."""
    with torch.no_grad():
        eager = instance(*inputs)
        oracle_out = oracle_forward(inputs)

    checks = {
        "stride": tuple(oracle_out.stride()) == tuple(eager.stride()) == OUTPUT_STRIDE,
        "storage_offset": oracle_out.storage_offset() == eager.storage_offset() == 0,
        "storage_bytes": oracle_out.untyped_storage().nbytes()
        == eager.untyped_storage().nbytes()
        == N_BASE_ELEMENTS * oracle_out.element_size(),
    }
    ok = all(checks.values())
    detail = ", ".join(f"{name}={'PASS' if passed else 'FAIL'}" for name, passed in checks.items())
    print(
        f"  output 0 layout: {'PASS' if ok else 'FAIL'} "
        f"({detail}, stride={tuple(oracle_out.stride())}, storage_offset={oracle_out.storage_offset()})"
    )
    return ok


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
        ok = check_output_layout(instance, inputs) and ok
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
