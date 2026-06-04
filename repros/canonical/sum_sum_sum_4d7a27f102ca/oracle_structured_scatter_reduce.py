"""
Oracle for sum_sum_sum_4d7a27f102ca

Gap diagnosis:
  Classification: SCATTER_REDUCE
  What oracle does differently: Fuses both Swin relative-position batch reductions, duplicate-index scatter-adds, and the softmax-backward side-output store into one Triton producer.
  What Inductor change would fix: Add a structured relative-position scatter-reduce lowering that recognizes index_put(accumulate=True) buckets and emits the dependent full-tensor output from the same producer.
"""
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

# Import shared oracle infrastructure (installed via pip install -e .)
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


N = 8192
C = 4
H = 49
W = 49
NC = N * C
BUCKETS = 169
BLOCK_N = 64
BLOCK_W = 64
ZERO_BLOCK = 1024


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _validate_inputs(
    fma_22: torch.Tensor,
    primals_26: torch.Tensor,
    bmm_141: torch.Tensor,
    div: torch.Tensor,
    primals_11: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
) -> None:
    expected = [
        (tuple(fma_22.shape), (N, C, H, W), "fma_22"),
        (tuple(primals_26.shape), (H, W), "primals_26"),
        (tuple(bmm_141.shape), (NC, H, W), "bmm_141"),
        (tuple(div.shape), (N, C, H, W), "div"),
        (tuple(primals_11.shape), (H, W), "primals_11"),
    ]
    for got, want, name in expected:
        if got != want:
            raise ValueError(f"unexpected {name} shape: got={got} expected={want}")
    if fma_22.dtype != torch.float32 or bmm_141.dtype != torch.float32 or div.dtype != torch.float32:
        raise ValueError("expected float32 source tensors")
    if primals_26.dtype != torch.int64 or primals_11.dtype != torch.int64:
        raise ValueError("expected int64 relative-position indices")
    if list(_shape_param_0) != [H * W, C] or list(_shape_param_2) != [H * W, C]:
        raise ValueError("unexpected scatter view shape parameter")
    if list(_shape_param_1) != [N, C, H, W] or list(_shape_param_3) != [NC, H, W]:
        raise ValueError("unexpected softmax-backward view shape parameter")


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _zero_pair_kernel(
        out0_ptr,
        out1_ptr,
        total: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total
        zeros = tl.zeros((BLOCK,), tl.float32)
        tl.store(out0_ptr + offsets, zeros, mask=mask)
        tl.store(out1_ptr + offsets, zeros, mask=mask)

    @triton.jit
    def oracle_kernel(
        fma_ptr,
        index0_ptr,
        bmm_ptr,
        softmax_ptr,
        index1_ptr,
        out0_ptr,
        out1_ptr,
        out2_ptr,
        fma_stride_n: tl.constexpr,
        fma_stride_c: tl.constexpr,
        fma_stride_h: tl.constexpr,
        fma_stride_w: tl.constexpr,
        index0_stride_h: tl.constexpr,
        index0_stride_w: tl.constexpr,
        bmm_stride_m: tl.constexpr,
        bmm_stride_h: tl.constexpr,
        bmm_stride_w: tl.constexpr,
        softmax_stride_n: tl.constexpr,
        softmax_stride_c: tl.constexpr,
        softmax_stride_h: tl.constexpr,
        softmax_stride_w: tl.constexpr,
        index1_stride_h: tl.constexpr,
        index1_stride_w: tl.constexpr,
        out2_stride_m: tl.constexpr,
        out2_stride_h: tl.constexpr,
        out2_stride_w: tl.constexpr,
        N_: tl.constexpr,
        C_: tl.constexpr,
        H_: tl.constexpr,
        W_: tl.constexpr,
        BUCKETS_: tl.constexpr,
        BLOCK_N_: tl.constexpr,
        BLOCK_W_: tl.constexpr,
    ):
        c = tl.program_id(0)
        h = tl.program_id(1)
        n_block = tl.program_id(2)

        n_offsets = n_block * BLOCK_N_ + tl.arange(0, BLOCK_N_)[:, None]
        w_vec = tl.arange(0, BLOCK_W_)
        w_offsets = w_vec[None, :]
        n_mask = n_offsets < N_
        w_mask = w_offsets < W_
        mask = n_mask & w_mask
        m_offsets = n_offsets * C_ + c

        fma_offsets = (
            n_offsets * fma_stride_n
            + c * fma_stride_c
            + h * fma_stride_h
            + w_offsets * fma_stride_w
        )
        fma_vals = tl.load(fma_ptr + fma_offsets, mask=mask, other=0.0).to(tl.float32)
        partial0 = tl.sum(tl.where(mask, fma_vals, 0.0), axis=0)

        bmm_offsets = m_offsets * bmm_stride_m + h * bmm_stride_h + w_offsets * bmm_stride_w
        softmax_offsets = (
            n_offsets * softmax_stride_n
            + c * softmax_stride_c
            + h * softmax_stride_h
            + w_offsets * softmax_stride_w
        )
        bmm_vals = tl.load(bmm_ptr + bmm_offsets, mask=mask, other=0.0).to(tl.float32)
        softmax_vals = tl.load(softmax_ptr + softmax_offsets, mask=mask, other=0.0).to(tl.float32)
        product = bmm_vals * softmax_vals
        row_sum = tl.sum(tl.where(mask, product, 0.0), axis=1)
        softmax_bwd = product - softmax_vals * row_sum[:, None]

        out2_offsets = m_offsets * out2_stride_m + h * out2_stride_h + w_offsets * out2_stride_w
        tl.store(out2_ptr + out2_offsets, softmax_bwd, mask=mask)
        partial1 = tl.sum(tl.where(mask, softmax_bwd, 0.0), axis=0)

        bucket0 = tl.load(
            index0_ptr + h * index0_stride_h + w_vec * index0_stride_w,
            mask=w_vec < W_,
            other=0,
        ).to(tl.int64)
        bucket1 = tl.load(
            index1_ptr + h * index1_stride_h + w_vec * index1_stride_w,
            mask=w_vec < W_,
            other=0,
        ).to(tl.int64)
        scatter0_mask = (w_vec < W_) & (bucket0 >= 0) & (bucket0 < BUCKETS_)
        scatter1_mask = (w_vec < W_) & (bucket1 >= 0) & (bucket1 < BUCKETS_)
        tl.atomic_add(out0_ptr + bucket0 * C_ + c, partial0, sem="relaxed", mask=scatter0_mask)
        tl.atomic_add(out1_ptr + bucket1 * C_ + c, partial1, sem="relaxed", mask=scatter1_mask)


def oracle_forward(inputs):
    """Run the full-scope Triton oracle for Repro.forward()."""
    if triton is None:
        raise RuntimeError("triton is not available")

    (
        fma_22,
        primals_26,
        bmm_141,
        div,
        primals_11,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
    ) = inputs

    if fma_22.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    _validate_inputs(
        fma_22,
        primals_26,
        bmm_141,
        div,
        primals_11,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
    )

    out0 = torch.empty_strided((BUCKETS, C), (C, 1), device=fma_22.device, dtype=fma_22.dtype)
    out1 = torch.empty_strided((BUCKETS, C), (C, 1), device=fma_22.device, dtype=fma_22.dtype)
    out2 = torch.empty_strided((NC, H, W), (H * W, W, 1), device=fma_22.device, dtype=fma_22.dtype)

    _zero_pair_kernel[(triton.cdiv(BUCKETS * C, ZERO_BLOCK),)](
        out0,
        out1,
        total=BUCKETS * C,
        BLOCK=ZERO_BLOCK,
        num_warps=4,
    )

    grid = (C, H, triton.cdiv(N, BLOCK_N))
    oracle_kernel[grid](
        fma_22,
        primals_26,
        bmm_141,
        div,
        primals_11,
        out0,
        out1,
        out2,
        fma_stride_n=fma_22.stride(0),
        fma_stride_c=fma_22.stride(1),
        fma_stride_h=fma_22.stride(2),
        fma_stride_w=fma_22.stride(3),
        index0_stride_h=primals_26.stride(0),
        index0_stride_w=primals_26.stride(1),
        bmm_stride_m=bmm_141.stride(0),
        bmm_stride_h=bmm_141.stride(1),
        bmm_stride_w=bmm_141.stride(2),
        softmax_stride_n=div.stride(0),
        softmax_stride_c=div.stride(1),
        softmax_stride_h=div.stride(2),
        softmax_stride_w=div.stride(3),
        index1_stride_h=primals_11.stride(0),
        index1_stride_w=primals_11.stride(1),
        out2_stride_m=out2.stride(0),
        out2_stride_h=out2.stride(1),
        out2_stride_w=out2.stride(2),
        N_=N,
        C_=C,
        H_=H,
        W_=W,
        BUCKETS_=BUCKETS,
        BLOCK_N_=BLOCK_N,
        BLOCK_W_=BLOCK_W,
        num_warps=4,
    )
    return out0, out1, out2


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
