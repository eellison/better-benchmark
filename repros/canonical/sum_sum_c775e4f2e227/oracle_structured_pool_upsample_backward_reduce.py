"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete Visformer adaptive-average-pool backward and batch-norm-backward-like return tuple by gathering each expanded pool-gradient value directly from the original `[128, 768]` source, masking the fixed `[128, 7, 7]` channel tile, reducing the two channel sums, and emitting the full contiguous `[128, 768, 7, 7]` epilogue plus `[768]` reduction output without materializing the zero-fill `as_strided_scatter -> as_strided -> expand -> div` tensor, whereas Inductor currently lowers that structured scatter/expand producer and the sibling reductions/full-tensor epilogue as separate generic kernels over materialized intermediates; Inductor cannot do this today because scheduler/codegen does not recognize this zero-filled as_strided scatter followed by broadcasted average-pool backward as a structured gather-mask-reduce producer that can feed both reductions and the required pointwise output; the fix is SCATTER_REDUCE: add a structured average-pool-backward scatter/expand lowering that keeps the pooled-gradient source live, tiles it with the activation layout, accumulates sibling channel reductions, and emits the full return tuple from the same fused template."""
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

N = 128
C = 768
H = 7
W = 7
HW = H * W
N_HW = N * HW
INV_HW = 1.0 / HW
REDUCTION_SCALE = 1.0 / (N * HW)
BLOCK_N = 128
BLOCK_HW = 64
BLOCK_C = 32


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def oracle_torch(
    mm: torch.Tensor,
    sub_36: torch.Tensor,
    squeeze_82: torch.Tensor,
    primals_203: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch fallback for CPU syntax/correctness checks."""
    pool_grad = mm[:, :, None, None] * INV_HW
    sum0 = mm.sum(dim=0)
    sum1 = (pool_grad * sub_36).sum(dim=(0, 2, 3))
    mean_term = sum0 * REDUCTION_SCALE
    var_term = sum1 * REDUCTION_SCALE * squeeze_82 * squeeze_82
    input_scale = squeeze_82 * primals_203
    out_value = (
        pool_grad
        - sub_36 * var_term[None, :, None, None]
        - mean_term[None, :, None, None]
    ) * input_scale[None, :, None, None]
    out0 = torch.empty(
        tuple(sub_36.shape),
        device=sub_36.device,
        dtype=sub_36.dtype,
    )
    out0.copy_(out_value)
    out1 = sum1 * squeeze_82
    return out0, out1


if triton is not None:

    @triton.jit
    def _pool_bn_reduce_hw_kernel(
        mm_ptr,
        sub_ptr,
        partial1_ptr,
        sub_stride_n: tl.constexpr,
        sub_stride_c: tl.constexpr,
        sub_stride_h: tl.constexpr,
        sub_stride_w: tl.constexpr,
        C_: tl.constexpr,
        W_: tl.constexpr,
        HW_: tl.constexpr,
        INV_HW_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        c_block = tl.program_id(0)
        n_idx = tl.program_id(1)
        offs_hw = tl.arange(0, BLOCK_HW_)
        offs_c = c_block * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        h_idx = offs_hw // W_
        w_idx = offs_hw - h_idx * W_
        mask = (offs_hw[:, None] < HW_) & (offs_c[None, :] < C_)

        sub_offsets = (
            n_idx * sub_stride_n
            + offs_c[None, :] * sub_stride_c
            + h_idx[:, None] * sub_stride_h
            + w_idx[:, None] * sub_stride_w
        )
        sub_vals = tl.load(sub_ptr + sub_offsets, mask=mask, other=0.0).to(tl.float32)
        mm_vals = tl.load(mm_ptr + n_idx * C_ + offs_c, mask=offs_c < C_, other=0.0).to(tl.float32)
        sum1 = tl.sum(sub_vals * (mm_vals[None, :] * INV_HW_), axis=0)
        tl.store(partial1_ptr + n_idx * C_ + offs_c, sum1, mask=offs_c < C_)

    @triton.jit
    def _pool_bn_finalize_kernel(
        mm_ptr,
        partial1_ptr,
        squeeze_ptr,
        primals_ptr,
        stats_ptr,
        out1_ptr,
        C_: tl.constexpr,
        N_: tl.constexpr,
        REDUCTION_SCALE_: tl.constexpr,
        BLOCK_N_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        c_block = tl.program_id(0)
        offs_n = tl.arange(0, BLOCK_N_)[:, None]
        offs_c = c_block * BLOCK_C_ + tl.arange(0, BLOCK_C_)[None, :]
        mask = (offs_n < N_) & (offs_c < C_)
        offsets = offs_n * C_ + offs_c

        mm_vals = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        partial1 = tl.load(partial1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sum0 = tl.sum(mm_vals, axis=0)
        sum1 = tl.sum(partial1, axis=0)

        channel = c_block * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        channel_mask = channel < C_
        squeeze = tl.load(squeeze_ptr + channel, mask=channel_mask, other=0.0).to(tl.float32)
        primal = tl.load(primals_ptr + channel, mask=channel_mask, other=0.0).to(tl.float32)
        mean_term = sum0 * REDUCTION_SCALE_
        var_term = sum1 * REDUCTION_SCALE_ * squeeze * squeeze
        input_scale = squeeze * primal

        tl.store(stats_ptr + channel, mean_term, mask=channel_mask)
        tl.store(stats_ptr + C_ + channel, var_term, mask=channel_mask)
        tl.store(stats_ptr + 2 * C_ + channel, input_scale, mask=channel_mask)
        tl.store(out1_ptr + channel, sum1 * squeeze, mask=channel_mask)

    @triton.jit
    def _pool_bn_epilogue_kernel(
        mm_ptr,
        sub_ptr,
        stats_ptr,
        out0_ptr,
        sub_stride_n: tl.constexpr,
        sub_stride_c: tl.constexpr,
        sub_stride_h: tl.constexpr,
        sub_stride_w: tl.constexpr,
        out_stride_n: tl.constexpr,
        out_stride_c: tl.constexpr,
        out_stride_h: tl.constexpr,
        out_stride_w: tl.constexpr,
        C_: tl.constexpr,
        W_: tl.constexpr,
        HW_: tl.constexpr,
        INV_HW_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        c_block = tl.program_id(0)
        n_idx = tl.program_id(1)
        offs_c = c_block * BLOCK_C_ + tl.arange(0, BLOCK_C_)[:, None]
        offs_hw = tl.arange(0, BLOCK_HW_)[None, :]
        h_idx = offs_hw // W_
        w_idx = offs_hw - h_idx * W_
        mask = (offs_c < C_) & (offs_hw < HW_)

        sub_offsets = (
            n_idx * sub_stride_n
            + offs_c * sub_stride_c
            + h_idx * sub_stride_h
            + w_idx * sub_stride_w
        )
        out_offsets = (
            n_idx * out_stride_n
            + offs_c * out_stride_c
            + h_idx * out_stride_h
            + w_idx * out_stride_w
        )

        sub_vals = tl.load(sub_ptr + sub_offsets, mask=mask, other=0.0).to(tl.float32)
        mm_vals = tl.load(mm_ptr + n_idx * C_ + offs_c, mask=offs_c < C_, other=0.0).to(tl.float32)
        mean_term = tl.load(stats_ptr + offs_c, mask=offs_c < C_, other=0.0).to(tl.float32)
        var_term = tl.load(stats_ptr + C_ + offs_c, mask=offs_c < C_, other=0.0).to(tl.float32)
        input_scale = tl.load(stats_ptr + 2 * C_ + offs_c, mask=offs_c < C_, other=0.0).to(tl.float32)

        pool_grad = mm_vals * INV_HW_
        out_vals = (
            pool_grad
            - sub_vals * var_term
            - mean_term
        ) * input_scale
        tl.store(out0_ptr + out_offsets, out_vals, mask=mask)

    @triton.jit
    def _pool_bn_gather_reduce_kernel(
        mm_ptr,
        sub_ptr,
        squeeze_ptr,
        primals_ptr,
        out0_ptr,
        out1_ptr,
        sub_stride_n: tl.constexpr,
        sub_stride_c: tl.constexpr,
        sub_stride_h: tl.constexpr,
        sub_stride_w: tl.constexpr,
        out_stride_n: tl.constexpr,
        out_stride_c: tl.constexpr,
        out_stride_h: tl.constexpr,
        out_stride_w: tl.constexpr,
        C_: tl.constexpr,
        N_: tl.constexpr,
        W_: tl.constexpr,
        HW_: tl.constexpr,
        N_HW_: tl.constexpr,
        INV_HW_: tl.constexpr,
        REDUCTION_SCALE_: tl.constexpr,
        BLOCK_N_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        channel = tl.program_id(0)
        n_idx = tl.arange(0, BLOCK_N_)[:, None]
        hw_idx = tl.arange(0, BLOCK_HW_)[None, :]
        h_idx = hw_idx // W_
        w_idx = hw_idx - h_idx * W_
        mask = (n_idx < N_) & (hw_idx < HW_)

        sub_offsets = (
            n_idx * sub_stride_n
            + channel * sub_stride_c
            + h_idx * sub_stride_h
            + w_idx * sub_stride_w
        )
        out_offsets = (
            n_idx * out_stride_n
            + channel * out_stride_c
            + h_idx * out_stride_h
            + w_idx * out_stride_w
        )
        mm_offsets = n_idx * C_ + channel

        sub_vals = tl.load(sub_ptr + sub_offsets, mask=mask, other=0.0).to(tl.float32)
        mm_vals = tl.load(mm_ptr + mm_offsets, mask=n_idx < N_, other=0.0).to(tl.float32)
        pool_grad = mm_vals * INV_HW_

        sum0 = tl.sum(mm_vals, axis=0)
        sum1 = tl.sum(tl.sum(pool_grad * sub_vals, axis=0), axis=0)

        squeeze = tl.load(squeeze_ptr + channel).to(tl.float32)
        primal = tl.load(primals_ptr + channel).to(tl.float32)
        mean_term = sum0 * REDUCTION_SCALE_
        var_term = sum1 * REDUCTION_SCALE_ * squeeze * squeeze
        input_scale = squeeze * primal

        out_vals = (pool_grad - sub_vals * var_term - mean_term) * input_scale
        tl.store(out0_ptr + out_offsets, out_vals, mask=mask)
        tl.store(out1_ptr + channel, sum1 * squeeze)


def oracle_triton(
    mm: torch.Tensor,
    sub_36: torch.Tensor,
    squeeze_82: torch.Tensor,
    primals_203: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is not available")
    if mm.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")
    if tuple(mm.shape) != (N, C) or tuple(sub_36.shape) != (N, C, H, W):
        raise ValueError("unexpected repro shape")

    out0 = torch.empty(
        (N, C, H, W),
        device=sub_36.device,
        dtype=sub_36.dtype,
    )
    out1 = torch.empty_like(squeeze_82)
    partial1 = torch.empty((N, C), device=sub_36.device, dtype=torch.float32)
    stats = torch.empty((3, C), device=sub_36.device, dtype=torch.float32)

    grid = (triton.cdiv(C, BLOCK_C), N)
    _pool_bn_reduce_hw_kernel[grid](
        mm,
        sub_36,
        partial1,
        sub_stride_n=sub_36.stride(0),
        sub_stride_c=sub_36.stride(1),
        sub_stride_h=sub_36.stride(2),
        sub_stride_w=sub_36.stride(3),
        C_=C,
        W_=W,
        HW_=HW,
        INV_HW_=INV_HW,
        BLOCK_HW_=BLOCK_HW,
        BLOCK_C_=BLOCK_C,
        num_warps=8,
    )
    _pool_bn_finalize_kernel[(triton.cdiv(C, BLOCK_C),)](
        mm,
        partial1,
        squeeze_82,
        primals_203,
        stats,
        out1,
        C_=C,
        N_=N,
        REDUCTION_SCALE_=REDUCTION_SCALE,
        BLOCK_N_=BLOCK_N,
        BLOCK_C_=BLOCK_C,
        num_warps=8,
    )
    _pool_bn_epilogue_kernel[grid](
        mm,
        sub_36,
        stats,
        out0,
        sub_stride_n=sub_36.stride(0),
        sub_stride_c=sub_36.stride(1),
        sub_stride_h=sub_36.stride(2),
        sub_stride_w=sub_36.stride(3),
        out_stride_n=out0.stride(0),
        out_stride_c=out0.stride(1),
        out_stride_h=out0.stride(2),
        out_stride_w=out0.stride(3),
        C_=C,
        W_=W,
        HW_=HW,
        INV_HW_=INV_HW,
        BLOCK_HW_=BLOCK_HW,
        BLOCK_C_=BLOCK_C,
        num_warps=16,
    )
    return out0, out1


def oracle_forward(inputs):
    """Run the full-scope oracle computation."""
    mm = inputs[0]
    if isinstance(mm, torch.Tensor) and mm.device.type == "cuda" and triton is not None:
        return oracle_triton(*inputs)
    return oracle_torch(*inputs)


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
