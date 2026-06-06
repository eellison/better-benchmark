"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete GPT-2 segment-aware causal mask materialization in one pointwise Triton kernel, writing twelve distinct fp32 [8,1,1024,1024] storages and returning the required expanded [8,12,1024,1024] zero-stride views, whereas Inductor already lowers the repro to one fused pointwise kernel with the same two cumsum loads and twelve stores; Inductor cannot remove much more work while preserving twelve non-aliased returned storages because the full materialized mask buffers are the observable outputs; the fix is BANDWIDTH_BOUND: treat this as a memory-traffic floor unless a downstream attention consumer can be fused to avoid returning the masks."""
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


BATCH = 8
SEQ = 1024
N_OUTPUTS = 12
BASE_SHAPE = (BATCH, 1, SEQ, SEQ)
BASE_STRIDE = (SEQ * SEQ, SEQ * SEQ, SEQ, 1)
OUTPUT_SHAPE = (BATCH, 12, SEQ, SEQ)
N_ELEMENTS = BATCH * SEQ * SEQ
BLOCK_SIZE = 1024


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _resolve_expand_shape(
    shape_param: Any,
    source_shape: tuple[int, ...],
) -> tuple[int, ...]:
    dims = tuple(int(dim) for dim in shape_param)
    if len(dims) != len(source_shape):
        raise ValueError(
            f"expand rank {len(dims)} does not match source rank {len(source_shape)}"
        )
    return tuple(source if dim == -1 else dim for dim, source in zip(dims, source_shape))


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, tuple[tuple[int, ...], ...]]:
    if len(inputs) != 1 + 1 + N_OUTPUTS:
        raise ValueError(f"{REPRO_ID} expects 14 inputs, got {len(inputs)}")

    cumsum = inputs[0]
    if not isinstance(cumsum, torch.Tensor):
        raise TypeError(f"cumsum must be a tensor, got {type(cumsum)!r}")
    if tuple(cumsum.shape) != (BATCH, SEQ):
        raise ValueError(f"cumsum has shape {tuple(cumsum.shape)}, expected {(BATCH, SEQ)}")
    if cumsum.dtype != torch.int64:
        raise TypeError(f"cumsum has dtype {cumsum.dtype}, expected torch.int64")
    if cumsum.stride() != (SEQ, 1):
        raise ValueError(f"cumsum has stride {cumsum.stride()}, expected {(SEQ, 1)}")

    mask_expand_shape = _resolve_expand_shape(inputs[1], BASE_SHAPE)
    if mask_expand_shape != BASE_SHAPE:
        raise ValueError(
            f"_shape_param_0 resolves to {mask_expand_shape}, expected {BASE_SHAPE}"
        )

    output_shapes = tuple(
        _resolve_expand_shape(shape_param, BASE_SHAPE) for shape_param in inputs[2:]
    )
    for idx, shape in enumerate(output_shapes):
        if shape != OUTPUT_SHAPE:
            raise ValueError(
                f"_shape_param_{idx + 1} resolves to {shape}, expected {OUTPUT_SHAPE}"
            )

    return cumsum, output_shapes


def _aten_fallback(
    cumsum: torch.Tensor,
    output_shapes: tuple[tuple[int, ...], ...],
) -> tuple[torch.Tensor, ...]:
    device = cumsum.device
    rows = torch.arange(SEQ, device=device, dtype=torch.int64).view(1, 1, SEQ, 1)
    cols = torch.arange(SEQ, device=device, dtype=torch.int64).view(1, 1, 1, SEQ)
    causal = cols <= rows
    segment_match = cumsum[:, None, :, None] == cumsum[:, None, None, :]
    mask = causal & segment_match
    zero = torch.tensor(0.0, device=device, dtype=torch.float32)
    neg_inf = torch.tensor(-float("inf"), device=device, dtype=torch.float32)
    return tuple(
        torch.where(mask, zero, neg_inf).expand(shape) for shape in output_shapes
    )


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _segment_causal_mask_12_kernel(
        cumsum_ptr,
        out0,
        out1,
        out2,
        out3,
        out4,
        out5,
        out6,
        out7,
        out8,
        out9,
        out10,
        out11,
        n_elements: tl.constexpr,
        seq: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        valid = offsets < n_elements

        col = offsets % seq
        row = (offsets // seq) % seq
        batch = offsets // (seq * seq)
        cumsum_base = cumsum_ptr + batch * seq

        row_value = tl.load(cumsum_base + row, mask=valid, other=0)
        col_value = tl.load(cumsum_base + col, mask=valid, other=1)
        keep = (col <= row) & (row_value == col_value)
        value = tl.where(keep, 0.0, -float("inf"))

        tl.store(out0 + offsets, value, mask=valid)
        tl.store(out1 + offsets, value, mask=valid)
        tl.store(out2 + offsets, value, mask=valid)
        tl.store(out3 + offsets, value, mask=valid)
        tl.store(out4 + offsets, value, mask=valid)
        tl.store(out5 + offsets, value, mask=valid)
        tl.store(out6 + offsets, value, mask=valid)
        tl.store(out7 + offsets, value, mask=valid)
        tl.store(out8 + offsets, value, mask=valid)
        tl.store(out9 + offsets, value, mask=valid)
        tl.store(out10 + offsets, value, mask=valid)
        tl.store(out11 + offsets, value, mask=valid)


def oracle_forward(inputs: list[Any] | tuple[Any, ...]):
    """Run the full-scope segment-aware causal mask computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same twelve outputs. Each output is an fp32 expanded view with shape
    [8,12,1024,1024], stride [1048576,0,1024,1], storage offset 0, and a
    distinct [8,1,1024,1024] backing storage.
    """
    cumsum, output_shapes = _validate_inputs(inputs)

    if triton is None or not cumsum.is_cuda:
        return _aten_fallback(cumsum, output_shapes)

    bases = tuple(
        torch.empty_strided(
            BASE_SHAPE,
            BASE_STRIDE,
            device=cumsum.device,
            dtype=torch.float32,
        )
        for _ in range(N_OUTPUTS)
    )

    grid = (triton.cdiv(N_ELEMENTS, BLOCK_SIZE),)
    _segment_causal_mask_12_kernel[grid](
        cumsum,
        bases[0],
        bases[1],
        bases[2],
        bases[3],
        bases[4],
        bases[5],
        bases[6],
        bases[7],
        bases[8],
        bases[9],
        bases[10],
        bases[11],
        n_elements=N_ELEMENTS,
        seq=SEQ,
        block_size=BLOCK_SIZE,
        num_warps=4,
    )

    return tuple(base.expand(shape) for base, shape in zip(bases, output_shapes))


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
