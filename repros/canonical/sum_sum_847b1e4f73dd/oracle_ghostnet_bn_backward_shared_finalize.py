"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete GhostNet BN-backward-style captured scope, including the masked sliced-add producer, both per-channel reductions, the dependent full-tensor epilogue, and the 8-element scale-gradient side output, while sharing the final reduction over the 64 partial chunks for `sum(where * centered)` and `sum(where)` in one Triton finalizer; Inductor already reads the big input tensors once with two accumulators and the oracle's only structural difference is avoiding one tiny sibling-finalizer launch before the same bandwidth-heavy epilogue; Inductor cannot co-finalize those sibling partial reductions today because the scheduler does not fuse equal-`xnumel/rnumel` final-reduction epilogues, but bench_oracle shows that difference is hidden by full-tensor memory traffic; the fix is BANDWIDTH_BOUND: mark this repro at floor rather than filing a local Inductor optimization, unless broader launch-overhead or memory-traffic work moves both implementations."""
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
    def _partial_reduce_kernel(
        getitem_273_ptr,
        getitem_276_ptr,
        arg201_ptr,
        full_ptr,
        arg199_ptr,
        arg538_ptr,
        partial_prod_ptr,
        partial_where_ptr,
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

        full_value = tl.load(full_ptr).to(tl.float32)
        mean = tl.load(arg538_ptr + channel).to(tl.float32)
        acc_prod = tl.full((BLOCK_K,), 0.0, tl.float32)
        acc_where = tl.full((BLOCK_K,), 0.0, tl.float32)

        for base in tl.range(0, CHUNK_K_, BLOCK_K):
            k = base + offsets
            batch_in_chunk = k // HW_
            spatial = k - batch_in_chunk * HW_
            batch = chunk * BATCHES_PER_CHUNK_ + batch_in_chunk
            narrow_offset = batch * (C_ * HW_) + channel * HW_ + spatial
            wide_offset = batch * (IN_C_ * HW_) + channel * HW_ + spatial

            mask_source = tl.load(arg201_ptr + narrow_offset).to(tl.float32)
            lhs = tl.load(getitem_273_ptr + wide_offset).to(tl.float32)
            rhs = tl.load(getitem_276_ptr + narrow_offset).to(tl.float32)
            add_value = lhs + rhs
            where_value = tl.where(mask_source <= 0.0, full_value, add_value)
            centered = tl.load(arg199_ptr + narrow_offset).to(tl.float32) - mean
            prod = where_value * centered
            acc_prod = acc_prod + prod
            acc_where = acc_where + where_value

        partial_offset = chunk * C_ + channel
        tl.store(partial_prod_ptr + partial_offset, tl.sum(acc_prod, axis=0))
        tl.store(partial_where_ptr + partial_offset, tl.sum(acc_where, axis=0))

    @triton.jit
    def _finalize_kernel(
        partial_prod_ptr,
        partial_where_ptr,
        arg200_ptr,
        sum_prod_ptr,
        sum_where_ptr,
        out_vec_ptr,
        C_: tl.constexpr,
        BATCH_CHUNKS_: tl.constexpr,
        BLOCK_CHUNKS: tl.constexpr,
    ):
        channel = tl.program_id(0)
        chunks = tl.arange(0, BLOCK_CHUNKS)
        mask = chunks < BATCH_CHUNKS_
        partial_offsets = chunks * C_ + channel
        prod_values = tl.load(partial_prod_ptr + partial_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        where_values = tl.load(
            partial_where_ptr + partial_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        sum_prod = tl.sum(prod_values, axis=0).to(tl.float32)
        sum_where = tl.sum(where_values, axis=0).to(tl.float32)
        scale = tl.load(arg200_ptr + channel).to(tl.float32)
        tl.store(sum_prod_ptr + channel, sum_prod)
        tl.store(sum_where_ptr + channel, sum_where)
        tl.store(out_vec_ptr + channel, sum_prod * scale)

    @triton.jit
    def _epilogue_kernel(
        getitem_273_ptr,
        getitem_276_ptr,
        arg201_ptr,
        full_ptr,
        arg199_ptr,
        arg538_ptr,
        sum_prod_ptr,
        arg200_ptr,
        sum_where_ptr,
        arg4_ptr,
        out_ptr,
        NUMEL_: tl.constexpr,
        C_: tl.constexpr,
        IN_C_: tl.constexpr,
        HW_: tl.constexpr,
        REDUCE_SCALE_: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        xindex = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = xindex < NUMEL_
        batch = xindex // (C_ * HW_)
        channel = (xindex // HW_) % C_
        channel_spatial = xindex % (C_ * HW_)
        wide_offset = batch * (IN_C_ * HW_) + channel_spatial

        tmp0 = tl.load(arg201_ptr + xindex, mask=mask, other=0.0).to(tl.float32)
        tmp3 = tl.load(full_ptr).to(tl.float32)
        tmp5 = tl.load(getitem_273_ptr + wide_offset, mask=mask, other=0.0).to(
            tl.float32
        )
        tmp6 = tl.load(getitem_276_ptr + xindex, mask=mask, other=0.0).to(tl.float32)
        tmp9 = tl.load(arg199_ptr + xindex, mask=mask, other=0.0).to(tl.float32)
        tmp10 = tl.load(arg538_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        tmp12 = tl.load(sum_prod_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        tmp15 = tl.load(arg200_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        tmp20 = tl.load(sum_where_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        tmp23 = tl.load(arg4_ptr + channel, mask=mask, other=0.0).to(tl.float32)

        tmp2 = tmp0 <= 0.0
        tmp7 = tmp5 + tmp6
        tmp8 = tl.where(tmp2, tmp3, tmp7)
        tmp11 = tmp9 - tmp10
        tmp14 = tmp12 * REDUCE_SCALE_
        tmp16 = tmp15 * tmp15
        tmp17 = tmp14 * tmp16
        tmp18 = tmp11 * tmp17
        tmp19 = tmp8 - tmp18
        tmp21 = tmp20 * REDUCE_SCALE_
        tmp22 = tmp19 - tmp21
        tmp24 = tmp15 * tmp23
        tmp25 = tmp22 * tmp24
        tl.store(out_ptr + xindex, tmp25, mask=mask)


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
        getitem_273,
        getitem_276,
        arg201_1,
        full,
        arg199_1,
        arg538_1,
        arg200_1,
        arg4_1,
    ) = inputs

    assert getitem_273.shape == (BATCH, IN_CHANNELS, HEIGHT, WIDTH)
    assert getitem_276.shape == (BATCH, CHANNELS, HEIGHT, WIDTH)
    assert arg201_1.shape == (BATCH, CHANNELS, HEIGHT, WIDTH)
    assert full.shape == ()
    assert arg199_1.shape == (BATCH, CHANNELS, HEIGHT, WIDTH)
    assert arg538_1.shape == (1, CHANNELS, 1, 1)
    assert arg200_1.shape == (CHANNELS,)
    assert arg4_1.shape == (CHANNELS,)
    assert getitem_273.is_contiguous()
    assert getitem_276.is_contiguous()
    assert arg201_1.is_contiguous()
    assert arg199_1.is_contiguous()
    assert arg538_1.is_contiguous()
    assert arg200_1.is_contiguous()
    assert arg4_1.is_contiguous()

    partial_prod = torch.empty(
        (BATCH_CHUNKS, CHANNELS), device=getitem_273.device, dtype=torch.float32
    )
    partial_where = torch.empty_like(partial_prod)
    sum_prod = torch.empty((CHANNELS,), device=getitem_273.device, dtype=torch.float32)
    sum_where = torch.empty_like(sum_prod)
    out_tensor = torch.empty_like(getitem_276)
    out_vec = torch.empty_like(arg200_1)

    _partial_reduce_kernel[(CHANNELS, BATCH_CHUNKS)](
        getitem_273,
        getitem_276,
        arg201_1,
        full,
        arg199_1,
        arg538_1,
        partial_prod,
        partial_where,
        C_=CHANNELS,
        IN_C_=IN_CHANNELS,
        HW_=HW,
        CHUNK_K_=CHUNK_K,
        BATCHES_PER_CHUNK_=BATCHES_PER_CHUNK,
        BLOCK_K=REDUCE_BLOCK,
        num_warps=8,
    )
    _finalize_kernel[(CHANNELS,)](
        partial_prod,
        partial_where,
        arg200_1,
        sum_prod,
        sum_where,
        out_vec,
        C_=CHANNELS,
        BATCH_CHUNKS_=BATCH_CHUNKS,
        BLOCK_CHUNKS=BATCH_CHUNKS,
        num_warps=2,
    )
    _epilogue_kernel[(triton.cdiv(NUMEL, POINTWISE_BLOCK),)](
        getitem_273,
        getitem_276,
        arg201_1,
        full,
        arg199_1,
        arg538_1,
        sum_prod,
        arg200_1,
        sum_where,
        arg4_1,
        out_tensor,
        NUMEL_=NUMEL,
        C_=CHANNELS,
        IN_C_=IN_CHANNELS,
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
