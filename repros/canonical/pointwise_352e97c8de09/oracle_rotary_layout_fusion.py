"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle fuses the full rotary embedding scope for Q and repeated KV into one layout-aware Triton kernel, preserving Q's non-contiguous permuted output stride while materializing the KV expand/clone/view output directly in contiguous repeated-head layout; Inductor currently emits separate layout and pointwise work for the rotate-half cat/mul/add and repeated KV materialization because its scheduler cannot assign two different output layouts, including a repeat expansion side write, to one producer tile; the fix is SCHEDULER_FUSION: teach pointwise/layout scheduling to fuse rotary-style slice/cat producers with expand/clone consumers and generate multi-output tiles with per-output layout ownership."""
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
    def _round_to_bf16_f32(x):
        return tl.inline_asm_elementwise(
            "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
            "=f,f",
            [x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_PAIRS": 128}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_PAIRS": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_PAIRS": 512}, num_warps=8, num_stages=3),
        ],
        key=["N_PAIRS", "D"],
    )
    @triton.jit
    def _rotary_q_repeat_kv_kernel(
        q_ptr,
        cos_ptr,
        sin_ptr,
        kv_ptr,
        out_q_ptr,
        out_kv_ptr,
        N_PAIRS: tl.constexpr,
        S: tl.constexpr,
        QH: tl.constexpr,
        KVH: tl.constexpr,
        D: tl.constexpr,
        HALF: tl.constexpr,
        GROUPS: tl.constexpr,
        BLOCK_PAIRS: tl.constexpr,
    ):
        pair_offsets = tl.program_id(0) * BLOCK_PAIRS + tl.arange(0, BLOCK_PAIRS)
        mask = pair_offsets < N_PAIRS

        half_d = pair_offsets % HALF
        q_head = (pair_offsets // HALF) % QH
        seq = (pair_offsets // (HALF * QH)) % S
        batch = pair_offsets // (HALF * QH * S)

        q_base = ((batch * S + seq) * QH + q_head) * D + half_d
        cos_base = seq * D + half_d

        q_lo = tl.load(q_ptr + q_base, mask=mask).to(tl.float32)
        q_hi = tl.load(q_ptr + q_base + HALF, mask=mask).to(tl.float32)
        cos_lo = tl.load(cos_ptr + cos_base, mask=mask).to(tl.float32)
        cos_hi = tl.load(cos_ptr + cos_base + HALF, mask=mask).to(tl.float32)
        sin_lo = tl.load(sin_ptr + cos_base, mask=mask).to(tl.float32)
        sin_hi = tl.load(sin_ptr + cos_base + HALF, mask=mask).to(tl.float32)

        q_mul_lo = _round_to_bf16_f32(q_lo * cos_lo)
        q_rot_lo = _round_to_bf16_f32((-q_hi) * sin_lo)
        q_mul_hi = _round_to_bf16_f32(q_hi * cos_hi)
        q_rot_hi = _round_to_bf16_f32(q_lo * sin_hi)
        out_q_lo = q_mul_lo + q_rot_lo
        out_q_hi = q_mul_hi + q_rot_hi

        tl.store(out_q_ptr + q_base, out_q_lo.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)
        tl.store(out_q_ptr + q_base + HALF, out_q_hi.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)

        kv_mask = mask & (q_head < KVH)
        kv_base = ((batch * S + seq) * KVH + q_head) * D + half_d
        kv_lo = tl.load(kv_ptr + kv_base, mask=kv_mask).to(tl.float32)
        kv_hi = tl.load(kv_ptr + kv_base + HALF, mask=kv_mask).to(tl.float32)

        kv_mul_lo = _round_to_bf16_f32(kv_lo * cos_lo)
        kv_rot_lo = _round_to_bf16_f32((-kv_hi) * sin_lo)
        kv_mul_hi = _round_to_bf16_f32(kv_hi * cos_hi)
        kv_rot_hi = _round_to_bf16_f32(kv_lo * sin_hi)
        out_kv_lo = kv_mul_lo + kv_rot_lo
        out_kv_hi = kv_mul_hi + kv_rot_hi

        for group in tl.static_range(0, GROUPS):
            out_head = q_head * GROUPS + group
            out_base = ((batch * QH + out_head) * S + seq) * D + half_d
            tl.store(out_kv_ptr + out_base, out_kv_lo.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=kv_mask)
            tl.store(out_kv_ptr + out_base + HALF, out_kv_hi.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=kv_mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_ELEMS": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ELEMS": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ELEMS": 1024}, num_warps=8, num_stages=3),
        ],
        key=["N_ELEMS", "D"],
    )
    @triton.jit
    def _rotary_q_repeat_kv_fixed_split_kernel(
        q_ptr,
        cos_ptr,
        sin_ptr,
        kv_ptr,
        out_q_ptr,
        out_kv_ptr,
        N_ELEMS: tl.constexpr,
        S: tl.constexpr,
        QH: tl.constexpr,
        KVH: tl.constexpr,
        D: tl.constexpr,
        ROTARY_SPLIT: tl.constexpr,
        GROUPS: tl.constexpr,
        BLOCK_ELEMS: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
        mask = offsets < N_ELEMS

        dim = offsets % D
        q_head = (offsets // D) % QH
        seq = (offsets // (D * QH)) % S
        batch = offsets // (D * QH * S)

        neg_span = D - ROTARY_SPLIT
        src_dim = tl.where(dim < neg_span, dim + ROTARY_SPLIT, dim - neg_span)
        sign = tl.where(dim < neg_span, -1.0, 1.0)

        q_base = ((batch * S + seq) * QH + q_head) * D
        cos_base = seq * D + dim
        q_val = tl.load(q_ptr + q_base + dim, mask=mask).to(tl.float32)
        q_src = tl.load(q_ptr + q_base + src_dim, mask=mask).to(tl.float32)
        cos_val = tl.load(cos_ptr + cos_base, mask=mask).to(tl.float32)
        sin_val = tl.load(sin_ptr + cos_base, mask=mask).to(tl.float32)

        out_q = _round_to_bf16_f32(q_val * cos_val) + _round_to_bf16_f32(sign * q_src * sin_val)
        tl.store(
            out_q_ptr + q_base + dim,
            out_q.to(tl.bfloat16, fp_downcast_rounding="rtne"),
            mask=mask,
        )

        kv_mask = mask & (q_head < KVH)
        kv_base = ((batch * S + seq) * KVH + q_head) * D
        kv_val = tl.load(kv_ptr + kv_base + dim, mask=kv_mask).to(tl.float32)
        kv_src = tl.load(kv_ptr + kv_base + src_dim, mask=kv_mask).to(tl.float32)
        out_kv = _round_to_bf16_f32(kv_val * cos_val) + _round_to_bf16_f32(sign * kv_src * sin_val)

        for group in tl.static_range(0, GROUPS):
            out_head = q_head * GROUPS + group
            out_base = ((batch * QH + out_head) * S + seq) * D + dim
            tl.store(
                out_kv_ptr + out_base,
                out_kv.to(tl.bfloat16, fp_downcast_rounding="rtne"),
                mask=kv_mask,
            )


@oracle_impl(hardware="H100", shapes="(T([2048, 2048], bf16), T([1, 512, 64], bf16), T([1, 512, 64], bf16), T([2048, 512], bf16), S([4, 512, 2048]), S([4, 512, -1, 64]), S([4, 512, 512]), S([4, 512, -1, 64]), S([4, 8, 4, 512, 64]), S([4, 32, 512, 64]))")
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
        raise RuntimeError("Triton is required for oracle_rotary_layout_fusion.py")

    (
        q_mm,
        cos,
        sin,
        kv_mm,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
        _shape_param_5,
    ) = inputs

    if not q_mm.is_cuda:
        raise ValueError("oracle_rotary_layout_fusion.py expects CUDA inputs")
    if q_mm.dtype is not torch.bfloat16 or kv_mm.dtype is not torch.bfloat16:
        raise ValueError(f"unexpected input dtypes: q={q_mm.dtype}, kv={kv_mm.dtype}")
    if cos.dtype is not torch.bfloat16 or sin.dtype is not torch.bfloat16:
        raise ValueError(f"unexpected rotary dtypes: cos={cos.dtype}, sin={sin.dtype}")
    if tuple(cos.shape) != tuple(sin.shape) or len(cos.shape) != 3 or cos.shape[0] != 1:
        raise ValueError(f"unexpected rotary shapes: cos={tuple(cos.shape)}, sin={tuple(sin.shape)}")

    seq = cos.shape[1]
    head_dim = cos.shape[2]
    rotary_split = 32
    if head_dim < rotary_split:
        raise ValueError(f"head_dim must be at least {rotary_split}, got {head_dim}")
    if q_mm.shape[0] % seq != 0:
        raise ValueError(f"q rows {q_mm.shape[0]} are not divisible by sequence {seq}")

    batch = q_mm.shape[0] // seq
    q_heads = q_mm.shape[1] // head_dim
    kv_heads = kv_mm.shape[1] // head_dim
    if q_mm.shape != (batch * seq, q_heads * head_dim):
        raise ValueError(f"unexpected q shape: {tuple(q_mm.shape)}")
    if kv_mm.shape != (batch * seq, kv_heads * head_dim):
        raise ValueError(f"unexpected kv shape: {tuple(kv_mm.shape)}")
    if q_heads % kv_heads != 0:
        raise ValueError(f"q_heads={q_heads} must be divisible by kv_heads={kv_heads}")

    groups = q_heads // kv_heads
    out_shape = (batch, q_heads, seq, head_dim)
    out_q = torch.empty_strided(
        out_shape,
        (seq * q_heads * head_dim, head_dim, q_heads * head_dim, 1),
        device=q_mm.device,
        dtype=torch.bfloat16,
    )
    out_kv = torch.empty_strided(
        out_shape,
        (q_heads * seq * head_dim, seq * head_dim, head_dim, 1),
        device=q_mm.device,
        dtype=torch.bfloat16,
    )

    if head_dim == rotary_split * 2:
        n_pairs = batch * seq * q_heads * rotary_split
        grid = lambda meta: (triton.cdiv(n_pairs, meta["BLOCK_PAIRS"]),)
        _rotary_q_repeat_kv_kernel[grid](
            q_mm,
            cos,
            sin,
            kv_mm,
            out_q,
            out_kv,
            N_PAIRS=n_pairs,
            S=seq,
            QH=q_heads,
            KVH=kv_heads,
            D=head_dim,
            HALF=rotary_split,
            GROUPS=groups,
        )
    else:
        n_elems = batch * seq * q_heads * head_dim
        grid = lambda meta: (triton.cdiv(n_elems, meta["BLOCK_ELEMS"]),)
        _rotary_q_repeat_kv_fixed_split_kernel[grid](
            q_mm,
            cos,
            sin,
            kv_mm,
            out_q,
            out_kv,
            N_ELEMS=n_elems,
            S=seq,
            QH=q_heads,
            KVH=kv_heads,
            D=head_dim,
            ROTARY_SPLIT=rotary_split,
            GROUPS=groups,
        )
    return (out_q, out_kv)


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
