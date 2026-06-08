"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete GhostNet BN-backward-style captured scope, including the high-channel slice of the `getitem_252 + getitem_267` producer, both per-channel reductions, the dependent full-tensor epilogue, and the 8-element scale-gradient side output, while sharing the final reduction over the partial chunks for `sum(slice)` and `sum(slice * centered)` in one Triton finalizer; Inductor already schedules this as a memory-traffic-dominated multi-output reduction plus epilogue, so the oracle's structural difference is only direct high-slice producer recomputation and co-finalizing sibling summaries rather than exposing a materially cheaper algorithm; the fix is BANDWIDTH_BOUND: mark this repro at floor unless broader launch-overhead or memory-traffic improvements move both Inductor and the oracle together."""
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


BATCH = 512
IN_CHANNELS = 16
CHANNELS = 8
SLICE_START = 8
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
    def _partial_reduce_kernel(
        getitem_252_ptr,
        getitem_267_ptr,
        arg209_ptr,
        arg536_ptr,
        partial_slice_ptr,
        partial_prod_ptr,
        C_: tl.constexpr,
        IN_C_: tl.constexpr,
        SLICE_START_: tl.constexpr,
        HW_: tl.constexpr,
        CHUNK_K_: tl.constexpr,
        BATCHES_PER_CHUNK_: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        channel = tl.program_id(0)
        chunk = tl.program_id(1)
        offsets = tl.arange(0, BLOCK_K)

        mean = tl.load(arg536_ptr + channel).to(tl.float32)
        acc_slice = tl.full((BLOCK_K,), 0.0, tl.float32)
        acc_prod = tl.full((BLOCK_K,), 0.0, tl.float32)

        for base in tl.range(0, CHUNK_K_, BLOCK_K):
            k = base + offsets
            batch_in_chunk = k // HW_
            spatial = k - batch_in_chunk * HW_
            batch = chunk * BATCHES_PER_CHUNK_ + batch_in_chunk
            narrow_offset = batch * (C_ * HW_) + channel * HW_ + spatial
            wide_channel = channel + SLICE_START_
            wide_offset = batch * (IN_C_ * HW_) + wide_channel * HW_ + spatial

            lhs = tl.load(getitem_252_ptr + wide_offset).to(tl.float32)
            rhs = tl.load(getitem_267_ptr + wide_offset).to(tl.float32)
            slice_value = _add_rn_f32(lhs, rhs)
            centered = _sub_rn_f32(tl.load(arg209_ptr + narrow_offset).to(tl.float32), mean)
            prod = _mul_rn_f32(slice_value, centered)
            acc_slice = _add_rn_f32(acc_slice, slice_value)
            acc_prod = _add_rn_f32(acc_prod, prod)

        partial_offset = chunk * C_ + channel
        tl.store(partial_slice_ptr + partial_offset, tl.sum(acc_slice, axis=0))
        tl.store(partial_prod_ptr + partial_offset, tl.sum(acc_prod, axis=0))

    @triton.jit
    def _finalize_kernel(
        partial_slice_ptr,
        partial_prod_ptr,
        arg210_ptr,
        sum_slice_ptr,
        sum_prod_ptr,
        out_vec_ptr,
        C_: tl.constexpr,
        BATCH_CHUNKS_: tl.constexpr,
        BLOCK_CHUNKS: tl.constexpr,
    ):
        channel = tl.program_id(0)
        chunks = tl.arange(0, BLOCK_CHUNKS)
        mask = chunks < BATCH_CHUNKS_
        partial_offsets = chunks * C_ + channel
        slice_values = tl.load(
            partial_slice_ptr + partial_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        prod_values = tl.load(
            partial_prod_ptr + partial_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        sum_slice = tl.sum(slice_values, axis=0).to(tl.float32)
        sum_prod = tl.sum(prod_values, axis=0).to(tl.float32)
        scale = tl.load(arg210_ptr + channel).to(tl.float32)
        tl.store(sum_slice_ptr + channel, sum_slice)
        tl.store(sum_prod_ptr + channel, sum_prod)
        tl.store(out_vec_ptr + channel, _mul_rn_f32(sum_prod, scale))

    @triton.jit
    def _epilogue_kernel(
        getitem_252_ptr,
        getitem_267_ptr,
        arg209_ptr,
        arg536_ptr,
        sum_slice_ptr,
        sum_prod_ptr,
        arg210_ptr,
        arg11_ptr,
        out_ptr,
        NUMEL_: tl.constexpr,
        C_: tl.constexpr,
        IN_C_: tl.constexpr,
        SLICE_START_: tl.constexpr,
        HW_: tl.constexpr,
        REDUCE_SCALE_: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        xindex = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = xindex < NUMEL_
        batch = xindex // (C_ * HW_)
        channel = (xindex // HW_) % C_
        channel_spatial = xindex % (C_ * HW_)
        narrow_offset = xindex
        wide_channel_spatial = channel_spatial + SLICE_START_ * HW_
        wide_offset = batch * (IN_C_ * HW_) + wide_channel_spatial

        tmp0 = tl.load(getitem_252_ptr + wide_offset, mask=mask, other=0.0).to(
            tl.float32
        )
        tmp1 = tl.load(getitem_267_ptr + wide_offset, mask=mask, other=0.0).to(
            tl.float32
        )
        tmp3 = tl.load(arg209_ptr + narrow_offset, mask=mask, other=0.0).to(tl.float32)
        tmp4 = tl.load(arg536_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        tmp6 = tl.load(sum_prod_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        tmp9 = tl.load(arg210_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        tmp14 = tl.load(sum_slice_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        tmp17 = tl.load(arg11_ptr + channel, mask=mask, other=0.0).to(tl.float32)

        tmp2 = _add_rn_f32(tmp0, tmp1)
        tmp5 = _sub_rn_f32(tmp3, tmp4)
        tmp8 = _mul_rn_f32(tmp6, REDUCE_SCALE_)
        tmp10 = _mul_rn_f32(tmp9, tmp9)
        tmp11 = _mul_rn_f32(tmp8, tmp10)
        tmp12 = _mul_rn_f32(tmp5, tmp11)
        tmp13 = _sub_rn_f32(tmp2, tmp12)
        tmp15 = _mul_rn_f32(tmp14, REDUCE_SCALE_)
        tmp16 = _sub_rn_f32(tmp13, tmp15)
        tmp18 = _mul_rn_f32(tmp9, tmp17)
        tmp19 = _mul_rn_f32(tmp16, tmp18)
        tl.store(out_ptr + xindex, tmp19, mask=mask)


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
        getitem_252,
        getitem_267,
        arg209_1,
        arg536_1,
        arg210_1,
        arg11_1,
    ) = inputs

    assert getitem_252.shape == (BATCH, IN_CHANNELS, HEIGHT, WIDTH)
    assert getitem_267.shape == (BATCH, IN_CHANNELS, HEIGHT, WIDTH)
    assert arg209_1.shape == (BATCH, CHANNELS, HEIGHT, WIDTH)
    assert arg536_1.shape == (1, CHANNELS, 1, 1)
    assert arg210_1.shape == (CHANNELS,)
    assert arg11_1.shape == (CHANNELS,)
    assert getitem_252.is_contiguous()
    assert getitem_267.is_contiguous()
    assert arg209_1.is_contiguous()
    assert arg536_1.is_contiguous()
    assert arg210_1.is_contiguous()
    assert arg11_1.is_contiguous()

    partial_slice = torch.empty(
        (BATCH_CHUNKS, CHANNELS), device=getitem_252.device, dtype=torch.float32
    )
    partial_prod = torch.empty_like(partial_slice)
    sum_slice = torch.empty((CHANNELS,), device=getitem_252.device, dtype=torch.float32)
    sum_prod = torch.empty_like(sum_slice)
    out_tensor = torch.empty_like(arg209_1)
    out_vec = torch.empty_like(arg210_1)

    _partial_reduce_kernel[(CHANNELS, BATCH_CHUNKS)](
        getitem_252,
        getitem_267,
        arg209_1,
        arg536_1,
        partial_slice,
        partial_prod,
        C_=CHANNELS,
        IN_C_=IN_CHANNELS,
        SLICE_START_=SLICE_START,
        HW_=HW,
        CHUNK_K_=CHUNK_K,
        BATCHES_PER_CHUNK_=BATCHES_PER_CHUNK,
        BLOCK_K=REDUCE_BLOCK,
        num_warps=8,
    )
    _finalize_kernel[(CHANNELS,)](
        partial_slice,
        partial_prod,
        arg210_1,
        sum_slice,
        sum_prod,
        out_vec,
        C_=CHANNELS,
        BATCH_CHUNKS_=BATCH_CHUNKS,
        BLOCK_CHUNKS=BATCH_CHUNKS,
        num_warps=2,
    )
    _epilogue_kernel[(triton.cdiv(NUMEL, POINTWISE_BLOCK),)](
        getitem_252,
        getitem_267,
        arg209_1,
        arg536_1,
        sum_slice,
        sum_prod,
        arg210_1,
        arg11_1,
        out_tensor,
        NUMEL_=NUMEL,
        C_=CHANNELS,
        IN_C_=IN_CHANNELS,
        SLICE_START_=SLICE_START,
        HW_=HW,
        REDUCE_SCALE_=REDUCE_SCALE,
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
