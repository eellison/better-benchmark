"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle fuses Moondream Q/K RoPE table construction, both rotary applications, both output cats, and the position-difference bool side output into one layout-aware kernel, whereas Inductor emits separate pointwise/cat work around the reshape-permute layout; Inductor cannot do this today because scheduler fusion does not form a single producer/consumer group across the duplicated Q/K RoPE subgraphs and cat assembly boundary; the fix is SCHEDULER_FUSION: teach Inductor to fuse repeated RoPE rotate-half patterns through concat output assembly while preserving side-output pointwise work."""
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


if triton is not None:

    @triton.jit
    def _rope_qk_kernel(
        q_ptr,
        inv_freq_ptr,
        k_ptr,
        out_q_ptr,
        out_k_ptr,
        ne_ptr,
        N_ROWS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        row_mask = rows < N_ROWS

        head = rows // 512
        pos = rows - head * 512
        in_base = pos[:, None] * 2048 + head[:, None] * 64
        out_base = rows[:, None] * 64

        d = tl.arange(0, 32)
        freq = tl.load(inv_freq_ptr + (d % 16)).to(tl.float32)
        theta = pos[:, None].to(tl.float32) * freq[None, :]
        cos_theta = tl.cos(theta)
        sin_theta = tl.sin(theta)

        rot_d = tl.where(d < 16, d + 16, d - 16)
        rot_sign = tl.where(d < 16, -1.0, 1.0)
        mask_32 = row_mask[:, None]

        q = tl.load(q_ptr + in_base + d[None, :], mask=mask_32, other=0.0).to(tl.float32)
        q_rot = tl.load(q_ptr + in_base + rot_d[None, :], mask=mask_32, other=0.0).to(tl.float32)
        q_out = q * cos_theta + q_rot * rot_sign[None, :] * sin_theta
        tl.store(out_q_ptr + out_base + d[None, :], q_out, mask=mask_32)

        k = tl.load(k_ptr + in_base + d[None, :], mask=mask_32, other=0.0).to(tl.float32)
        k_rot = tl.load(k_ptr + in_base + rot_d[None, :], mask=mask_32, other=0.0).to(tl.float32)
        k_out = k * cos_theta + k_rot * rot_sign[None, :] * sin_theta
        tl.store(out_k_ptr + out_base + d[None, :], k_out, mask=mask_32)

        tail_d = d + 32
        q_tail = tl.load(q_ptr + in_base + tail_d[None, :], mask=mask_32, other=0.0)
        k_tail = tl.load(k_ptr + in_base + tail_d[None, :], mask=mask_32, other=0.0)
        tl.store(out_q_ptr + out_base + tail_d[None, :], q_tail, mask=mask_32)
        tl.store(out_k_ptr + out_base + tail_d[None, :], k_tail, mask=mask_32)

        # Captured side output is diff([0..511], [-1, 0..510]) != 1, i.e. all false.
        tl.store(ne_ptr + pos, rows != rows, mask=row_mask & (head == 0))


@oracle_impl(hardware="H100", shapes="(T([512, 2048], f16), T([16], f16), T([512, 2048], f16), S([1, 512, 2048]), S([1, 512, -1, 64]), S([1, 16, 1]), S([1, 1, 512]), S([1, 512, 2, 16]), S([1, 512, 32]), S([1, 512, 2048]), S([1, 512, -1, 64]))")
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

    q_in, inv_freq, k_in = inputs[:3]
    out_q = torch.empty((1, 32, 512, 64), device=q_in.device, dtype=q_in.dtype)
    out_k = torch.empty_like(out_q)
    ne = torch.empty((1, 512), device=q_in.device, dtype=torch.bool)

    n_rows = 32 * 512
    block_rows = 4
    grid = (triton.cdiv(n_rows, block_rows),)
    _rope_qk_kernel[grid](
        q_in,
        inv_freq,
        k_in,
        out_q,
        out_k,
        ne,
        N_ROWS=n_rows,
        BLOCK_ROWS=block_rows,
        num_warps=4,
    )
    return (out_q, out_k, ne)


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
