"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full broadcasted hardsigmoid gate multiply with Triton kernels specialized for this corpus family, using a 512-lane contiguous pointwise kernel for the small-spatial default shape and a row/spatial tile that reuses each [B,C,1,1] gate across larger [H,W] variants, whereas Inductor lowers the chain as a generic flattened pointwise loop over the same output contract; the specialized broadcast tiling does not produce a repeatable speedup over Inductor on this repro because both paths are dominated by the required input read, output write, and one launch; the fix is BANDWIDTH_BOUND: record this repro as at floor rather than pursuing scheduler fusion for this isolated pattern."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

    @triton.jit
    def _broadcast_gate_kernel(
        gate_ptr,
        x_ptr,
        out_ptr,
        rows: tl.constexpr,
        spatial: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        spatial_offsets = tl.program_id(1) * BLOCK_S + tl.arange(0, BLOCK_S)

        row_mask = row_offsets < rows
        spatial_mask = spatial_offsets < spatial
        mask = row_mask[:, None] & spatial_mask[None, :]

        gate = tl.load(
            gate_ptr + row_offsets,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        gate = tl.minimum(tl.maximum(gate + 3.0, 0.0), 6.0) / 6.0

        x_offsets = row_offsets[:, None] * spatial + spatial_offsets[None, :]
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + x_offsets, x * gate[:, None], mask=mask)


    @triton.jit
    def _flat_gate_kernel(
        gate_ptr,
        x_ptr,
        out_ptr,
        n_elements: tl.constexpr,
        spatial: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        gate_offsets = offsets // spatial
        gate = tl.load(
            gate_ptr + gate_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gate = tl.minimum(tl.maximum(gate + 3.0, 0.0), 6.0) / 6.0
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offsets, x * gate, mask=mask)


def _kernel_config(spatial: int) -> tuple[int, int, int]:
    if spatial <= 64:
        return 32, 64, 4
    if spatial <= 256:
        return 8, 256, 4
    return 2, 1024, 8


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

    convolution_59, div_25 = inputs
    batch, channels, _, _ = convolution_59.shape
    _, _, height, width = div_25.shape
    spatial = height * width
    out = torch.empty_like(div_25)

    n_elements = div_25.numel()
    if spatial <= 64:
        block = 512
        _flat_gate_kernel[(triton.cdiv(n_elements, block),)](
            convolution_59,
            div_25,
            out,
            n_elements=n_elements,
            spatial=spatial,
            BLOCK=block,
            num_warps=4,
        )
        return out

    block_m, block_s, num_warps = _kernel_config(spatial)
    grid = (triton.cdiv(batch * channels, block_m), triton.cdiv(spatial, block_s))
    _broadcast_gate_kernel[grid](
        convolution_59,
        div_25,
        out,
        rows=batch * channels,
        spatial=spatial,
        BLOCK_M=block_m,
        BLOCK_S=block_s,
        num_warps=num_warps,
    )
    return out


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
