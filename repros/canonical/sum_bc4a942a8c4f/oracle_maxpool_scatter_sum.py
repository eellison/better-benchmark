"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete AlexNet maxpool-backward scatter_add, boolean where fill, and per-channel sum as a direct gather-mask-reduce over the low-memory maxpool offset domain, whereas Inductor currently materializes the dense [262144,169] scatter target, reshapes it to [1024,256,13,13], then applies where and a generic channel reduction; Inductor cannot do this today because scheduler/codegen does not recognize low-memory maxpool scatter_add as a structured scatter-reduce producer that can feed the masked channel sum without materializing the dense scatter buffer; the fix is SCATTER_REDUCE: add a maxpool-backward scatter-reduce lowering that maps offsets directly into the consumer mask and accumulates the final per-channel reduction."""
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


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _maxpool_where_sum_partials_kernel(
        grad_ptr,
        offsets_ptr,
        mask_ptr,
        full_ptr,
        partials_ptr,
        N: tl.constexpr,
        C: tl.constexpr,
        OH: tl.constexpr,
        OW: tl.constexpr,
        IH: tl.constexpr,
        IW: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        channel = tl.program_id(0)
        block_id = tl.program_id(1)
        lanes = block_id * BLOCK_N + tl.arange(0, BLOCK_N)

        mask_hw: tl.constexpr = IH * IW
        src_hw: tl.constexpr = OH * OW
        mask_total: tl.constexpr = N * IH * IW
        src_total: tl.constexpr = N * OH * OW

        valid_mask = lanes < mask_total
        mask_n = lanes // mask_hw
        mask_rem = lanes - mask_n * mask_hw
        mask_h = mask_rem // IW
        mask_w = mask_rem - mask_h * IW
        mask_index = ((mask_n * C + channel) * IH + mask_h) * IW + mask_w
        mask_value = tl.load(mask_ptr + mask_index, mask=valid_mask, other=0)
        full = tl.load(full_ptr).to(tl.float32)
        mask_sum = tl.sum(tl.where((mask_value != 0) & valid_mask, full, 0.0), axis=0)

        valid_src = lanes < src_total
        src_n = lanes // src_hw
        src_rem = lanes - src_n * src_hw
        out_h = src_rem // OW
        out_w = src_rem - out_h * OW
        src_index = ((src_n * C + channel) * OH + out_h) * OW + out_w

        lowmem_offset = tl.load(offsets_ptr + src_index, mask=valid_src, other=0).to(tl.int32)
        kernel_h = lowmem_offset // 3
        kernel_w = lowmem_offset - kernel_h * 3
        in_h = out_h * 2 + kernel_h
        in_w = out_w * 2 + kernel_w

        dst_mask_index = ((src_n * C + channel) * IH + in_h) * IW + in_w
        dst_mask = tl.load(mask_ptr + dst_mask_index, mask=valid_src, other=1)
        grad = tl.load(grad_ptr + src_index, mask=valid_src, other=0.0).to(tl.float32)
        scatter_sum = tl.sum(tl.where((dst_mask == 0) & valid_src, grad, 0.0), axis=0)

        n_blocks: tl.constexpr = tl.cdiv(mask_total, BLOCK_N)
        tl.store(partials_ptr + channel * n_blocks + block_id, mask_sum + scatter_sum)


    @triton.jit
    def _finalize_channel_sums_kernel(
        partials_ptr,
        out_ptr,
        N_BLOCKS: tl.constexpr,
        BLOCK_PARTIALS: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_PARTIALS)
        partials = tl.load(
            partials_ptr + channel * N_BLOCKS + offsets,
            mask=offsets < N_BLOCKS,
            other=0.0,
        ).to(tl.float32)
        tl.store(out_ptr + channel, tl.sum(partials, axis=0))


@oracle_impl(hardware="H100", shapes="(T([1024, 256, 6, 6], f32), T([1024, 256, 6, 6], i8, gen=Index(9)), T([1024, 256, 13, 13], b8), T([], f32), S([262144, 36]), S([262144, 36]), S([1024, 256, 13, 13]))")
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

    grad, offsets, mask, full, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    n_batches, n_channels, out_h, out_w = grad.shape
    in_h, in_w = mask.shape[2:]

    block_n = 4096
    n_blocks = triton.cdiv(n_batches * in_h * in_w, block_n)
    partials = torch.empty((n_channels, n_blocks), device=grad.device, dtype=torch.float32)
    out = torch.empty((n_channels,), device=grad.device, dtype=torch.float32)

    _maxpool_where_sum_partials_kernel[(n_channels, n_blocks)](
        grad,
        offsets,
        mask,
        full,
        partials,
        N=n_batches,
        C=n_channels,
        OH=out_h,
        OW=out_w,
        IH=in_h,
        IW=in_w,
        BLOCK_N=block_n,
        num_warps=8,
    )
    _finalize_channel_sums_kernel[(n_channels,)](
        partials,
        out,
        N_BLOCKS=n_blocks,
        BLOCK_PARTIALS=triton.next_power_of_2(n_blocks),
        num_warps=1,
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
