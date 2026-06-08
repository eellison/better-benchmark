"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full bool-mask scale, row reduction, and fma epilogue in one Triton kernel, while Inductor already emits the same fused persistent inner reduction for the captured shape; the remaining cost is dominated by three input streams plus one fp32 output stream, so the fix is BANDWIDTH_BOUND: no scheduler change is indicated beyond tile-shape tuning for this memory path."""
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


if triton is not None:

    @triton.jit
    def oracle_kernel(
        bmm_ptr,
        mask_ptr,
        prob_ptr,
        out_ptr,
        ROWS: tl.constexpr,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        col_offsets = tl.arange(0, BLOCK_N)
        offsets = row_offsets[:, None] * N + col_offsets[None, :]
        valid = (row_offsets[:, None] < ROWS) & (col_offsets[None, :] < N)

        bmm = tl.load(bmm_ptr + offsets, mask=valid, other=0.0)
        mask_i1 = tl.load(mask_ptr + offsets, mask=valid, other=0).to(tl.int1)
        prob = tl.load(prob_ptr + offsets, mask=valid, other=0.0)

        mask_f32 = mask_i1.to(tl.float32)
        scale = tl.full([1, 1], 1.1111111111111112, tl.float32)
        scaled_mask = mask_f32 * scale
        product = bmm * scaled_mask
        product = product * prob
        row_sum = tl.sum(product, axis=1)[:, None].to(tl.float32)
        out = tl.fma(-prob, row_sum, product)
        tl.store(out_ptr + offsets, out, mask=valid)


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = []
    running = 1
    for size in reversed(shape):
        stride.append(running)
        running *= size
    return tuple(reversed(stride))


def _block_m_for_n(n: int) -> int:
    if n <= 128:
        return 4
    if n <= 512:
        return 2
    return 1


def _num_warps_for_n(n: int) -> int:
    if n <= 128:
        return 4
    return 8


def oracle_forward(inputs):
    """Run the full Repro.forward computation for the captured inputs."""
    if triton is None or tl is None:
        raise RuntimeError("Triton is required for oracle_dropout_softmax_backward.py")

    bmm_41, arg123_1, arg122_1, _shape_param_0, _shape_param_1 = inputs
    out_shape = tuple(int(dim) for dim in _shape_param_1)
    out = torch.empty_strided(
        out_shape,
        _contiguous_strides(out_shape),
        device=bmm_41.device,
        dtype=bmm_41.dtype,
    )

    n = int(out_shape[-1])
    rows = bmm_41.numel() // n
    block_m = _block_m_for_n(n)
    block_n = triton.next_power_of_2(n)
    grid = (triton.cdiv(rows, block_m),)

    oracle_kernel[grid](
        bmm_41,
        arg123_1,
        arg122_1,
        out,
        ROWS=rows,
        N=n,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=_num_warps_for_n(n),
        num_stages=1,
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
