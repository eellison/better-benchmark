"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete GhostNet BN-backward-style captured scope, including the sliced-add producer, both sibling per-channel reductions, the dependent dense epilogue, and the 8-element scale-gradient side output, while sharing one split-K Triton reduction with two f32 accumulators and recomputing the cheap sliced-add/sub producers for the final tensor epilogue; Inductor already lands within 2.1% of this full-scope floor, so the possible scheduler savings from co-planning the sibling reductions are hidden by the required full-tensor reads and write; Inductor cannot materially improve this repro without reducing that mandatory bandwidth, not merely by changing launch structure; the fix is BANDWIDTH_BOUND: mark this repro at floor and avoid filing a local scheduler optimization unless a broader memory-traffic reduction applies."""
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


BATCH = 512
IN_CHANNELS = 16
CHANNELS = 8
HEIGHT = 112
WIDTH = 112
HW = HEIGHT * WIDTH
BATCHES_PER_CHUNK = 8
BATCH_CHUNKS = BATCH // BATCHES_PER_CHUNK
CHUNK_K = BATCHES_PER_CHUNK * HW
NUMEL = BATCH * CHANNELS * HW
REDUCE_SCALE = 1.5570192920918366e-07
REDUCE_BLOCK = 1024
POINTWISE_BLOCK = 256


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _add_rn_f32(a, b):
        return tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _sub_rn_f32(a, b):
        return tl.inline_asm_elementwise(
            "sub.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _mul_rn_f32(a, b):
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _partial_dual_reduce_kernel(
        copy_25_ptr,
        getitem_270_ptr,
        arg206_ptr,
        arg537_ptr,
        partial_add_ptr,
        partial_prod_ptr,
        C_: tl.constexpr,
        IN_C_: tl.constexpr,
        HW_: tl.constexpr,
        CHUNK_K_: tl.constexpr,
        BATCHES_PER_CHUNK_: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        channel = tl.program_id(0)
        chunk = tl.program_id(1)
        offsets = tl.arange(0, BLOCK_K)
        acc_add = tl.full((BLOCK_K,), 0.0, tl.float32)
        acc_prod = tl.full((BLOCK_K,), 0.0, tl.float32)
        mean = tl.load(arg537_ptr + channel).to(tl.float32)

        for base in tl.range(0, CHUNK_K_, BLOCK_K):
            k = base + offsets
            batch_in_chunk = k // HW_
            spatial = k - batch_in_chunk * HW_
            batch = chunk * BATCHES_PER_CHUNK_ + batch_in_chunk
            narrow_offset = batch * (C_ * HW_) + channel * HW_ + spatial
            wide_offset = batch * (IN_C_ * HW_) + channel * HW_ + spatial

            copy_value = tl.load(copy_25_ptr + wide_offset).to(tl.float32)
            rhs_value = tl.load(getitem_270_ptr + narrow_offset).to(tl.float32)
            add_value = _add_rn_f32(copy_value, rhs_value)
            x_value = tl.load(arg206_ptr + narrow_offset).to(tl.float32)
            centered = _sub_rn_f32(x_value, mean)
            prod = _mul_rn_f32(add_value, centered)

            acc_add = _add_rn_f32(acc_add, add_value)
            acc_prod = _add_rn_f32(acc_prod, prod)

        partial_offset = chunk * C_ + channel
        tl.store(partial_add_ptr + partial_offset, tl.sum(acc_add, axis=0))
        tl.store(partial_prod_ptr + partial_offset, tl.sum(acc_prod, axis=0))

    @triton.jit
    def _finalize_kernel(
        partial_add_ptr,
        partial_prod_ptr,
        arg207_ptr,
        arg9_ptr,
        mean_term_ptr,
        prod_coeff_ptr,
        output_scale_ptr,
        out_vec_ptr,
        C_: tl.constexpr,
        BATCH_CHUNKS_: tl.constexpr,
        BLOCK_CHUNKS: tl.constexpr,
        REDUCE_SCALE_: tl.constexpr,
    ):
        channel = tl.program_id(0)
        chunks = tl.arange(0, BLOCK_CHUNKS)
        mask = chunks < BATCH_CHUNKS_
        partial_offsets = chunks * C_ + channel
        add_values = tl.load(partial_add_ptr + partial_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        prod_values = tl.load(
            partial_prod_ptr + partial_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        sum_add = tl.sum(add_values, axis=0).to(tl.float32)
        sum_prod = tl.sum(prod_values, axis=0).to(tl.float32)
        invstd = tl.load(arg207_ptr + channel).to(tl.float32)
        affine_weight = tl.load(arg9_ptr + channel).to(tl.float32)

        mean_term = _mul_rn_f32(sum_add, REDUCE_SCALE_)
        sum_prod_scaled = _mul_rn_f32(sum_prod, REDUCE_SCALE_)
        invstd_sq = _mul_rn_f32(invstd, invstd)
        prod_coeff = _mul_rn_f32(sum_prod_scaled, invstd_sq)
        output_scale = _mul_rn_f32(invstd, affine_weight)
        out_vec = _mul_rn_f32(sum_prod, invstd)

        tl.store(mean_term_ptr + channel, mean_term)
        tl.store(prod_coeff_ptr + channel, prod_coeff)
        tl.store(output_scale_ptr + channel, output_scale)
        tl.store(out_vec_ptr + channel, out_vec)

    @triton.jit
    def _epilogue_kernel(
        copy_25_ptr,
        getitem_270_ptr,
        arg206_ptr,
        arg537_ptr,
        mean_term_ptr,
        prod_coeff_ptr,
        output_scale_ptr,
        out_ptr,
        NUMEL_: tl.constexpr,
        C_: tl.constexpr,
        IN_C_: tl.constexpr,
        HW_: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        xindex = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = xindex < NUMEL_
        batch = xindex // (C_ * HW_)
        channel = (xindex // HW_) % C_
        channel_spatial = xindex % (C_ * HW_)
        wide_offset = batch * (IN_C_ * HW_) + channel_spatial

        copy_value = tl.load(copy_25_ptr + wide_offset, mask=mask, other=0.0).to(
            tl.float32
        )
        rhs_value = tl.load(getitem_270_ptr + xindex, mask=mask, other=0.0).to(
            tl.float32
        )
        x_value = tl.load(arg206_ptr + xindex, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(arg537_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        mean_term = tl.load(mean_term_ptr + channel, mask=mask, other=0.0).to(
            tl.float32
        )
        prod_coeff = tl.load(prod_coeff_ptr + channel, mask=mask, other=0.0).to(
            tl.float32
        )
        output_scale = tl.load(output_scale_ptr + channel, mask=mask, other=0.0).to(
            tl.float32
        )

        add_value = _add_rn_f32(copy_value, rhs_value)
        centered = _sub_rn_f32(x_value, mean)
        correction = _mul_rn_f32(centered, prod_coeff)
        residual = _sub_rn_f32(add_value, correction)
        residual = _sub_rn_f32(residual, mean_term)
        out_value = _mul_rn_f32(residual, output_scale)
        tl.store(out_ptr + xindex, out_value, mask=mask)


@oracle_impl(hardware="H100", shapes="(T([512, 16, 112, 112], f32), T([512, 8, 112, 112], f32), T([512, 8, 112, 112], f32), T([1, 8, 1, 1], f32), T([8], f32), T([8], f32))")
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

    (
        copy_25,
        getitem_270,
        arg206_1,
        arg537_1,
        arg207_1,
        arg9_1,
    ) = inputs

    assert copy_25.shape == (BATCH, IN_CHANNELS, HEIGHT, WIDTH)
    assert getitem_270.shape == (BATCH, CHANNELS, HEIGHT, WIDTH)
    assert arg206_1.shape == (BATCH, CHANNELS, HEIGHT, WIDTH)
    assert arg537_1.shape == (1, CHANNELS, 1, 1)
    assert arg207_1.shape == (CHANNELS,)
    assert arg9_1.shape == (CHANNELS,)
    assert copy_25.is_contiguous()
    assert getitem_270.is_contiguous()
    assert arg206_1.is_contiguous()
    assert arg537_1.is_contiguous()
    assert arg207_1.is_contiguous()
    assert arg9_1.is_contiguous()

    partial_add = torch.empty(
        (BATCH_CHUNKS, CHANNELS), device=copy_25.device, dtype=torch.float32
    )
    partial_prod = torch.empty_like(partial_add)
    mean_term = torch.empty((CHANNELS,), device=copy_25.device, dtype=torch.float32)
    prod_coeff = torch.empty_like(mean_term)
    output_scale = torch.empty_like(mean_term)
    out_tensor = torch.empty_like(getitem_270)
    out_vec = torch.empty_like(arg207_1)

    _partial_dual_reduce_kernel[(CHANNELS, BATCH_CHUNKS)](
        copy_25,
        getitem_270,
        arg206_1,
        arg537_1,
        partial_add,
        partial_prod,
        C_=CHANNELS,
        IN_C_=IN_CHANNELS,
        HW_=HW,
        CHUNK_K_=CHUNK_K,
        BATCHES_PER_CHUNK_=BATCHES_PER_CHUNK,
        BLOCK_K=REDUCE_BLOCK,
        num_warps=8,
    )
    _finalize_kernel[(CHANNELS,)](
        partial_add,
        partial_prod,
        arg207_1,
        arg9_1,
        mean_term,
        prod_coeff,
        output_scale,
        out_vec,
        C_=CHANNELS,
        BATCH_CHUNKS_=BATCH_CHUNKS,
        BLOCK_CHUNKS=BATCH_CHUNKS,
        REDUCE_SCALE_=REDUCE_SCALE,
        num_warps=2,
    )
    _epilogue_kernel[(triton.cdiv(NUMEL, POINTWISE_BLOCK),)](
        copy_25,
        getitem_270,
        arg206_1,
        arg537_1,
        mean_term,
        prod_coeff,
        output_scale,
        out_tensor,
        NUMEL_=NUMEL,
        C_=CHANNELS,
        IN_C_=IN_CHANNELS,
        HW_=HW,
        BLOCK=POINTWISE_BLOCK,
        num_warps=4,
    )

    return out_tensor, out_vec


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
