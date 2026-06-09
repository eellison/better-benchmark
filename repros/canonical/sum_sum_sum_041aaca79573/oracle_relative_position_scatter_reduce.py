"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete Swin relative-position-bias backward scope by reducing both batch-summed `[8192, 4, 49, 49]` producers directly into duplicate `[169, 4]` relative-position buckets while writing the required `[32768, 49, 49]` softmax-backward side output, whereas Inductor currently lowers the two `sum(dim=0) -> permute -> view -> index_put(accumulate=True)` branches and the `mul -> sum -> fma -> view` branch as separate generic reduction, layout, scatter, and pointwise kernels over materialized intermediates; Inductor cannot do this today because scheduler/codegen does not recognize duplicate-index relative-position `index_put(accumulate=True)` as a structured scatter-reduce epilogue that can share the rowwise softmax-backward producer with required side-output stores; the fix is SCATTER_REDUCE: add a structured relative-position scatter-reduce lowering that fuses batch reductions with indexed bucket accumulation and emits any full-tensor side output from the same producer."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None

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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

N = 8192
C = 4
H = 49
W = 49
NC = N * C
BUCKETS = 169
BLOCK_N = 64
BLOCK_W = 64
ZERO_BLOCK = 1024


def get_inputs() -> tuple[Any, ...]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


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
    def _relative_position_scatter_reduce_kernel(
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
        softmax_vals = tl.load(softmax_ptr + softmax_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
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


def _validate_inputs(inputs: tuple[Any, ...]) -> None:
    if len(inputs) != 9:
        raise ValueError(f"expected 9 inputs, got {len(inputs)}")

    fma_22, arg13_1, bmm_93, arg186_1, arg6_1 = inputs[:5]
    tensor_specs = {
        "fma_22": (fma_22, (N, C, H, W), torch.float32),
        "arg13_1": (arg13_1, (H, W), torch.int64),
        "bmm_93": (bmm_93, (NC, H, W), torch.float32),
        "arg186_1": (arg186_1, (N, C, H, W), torch.float32),
        "arg6_1": (arg6_1, (H, W), torch.int64),
    }
    for name, (tensor, shape, dtype) in tensor_specs.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.device.type != "cuda":
            raise RuntimeError("Triton oracle requires CUDA inputs")
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(
                f"{name} expected shape={shape} dtype={dtype}, "
                f"got shape={tuple(tensor.shape)} dtype={tensor.dtype}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous for this captured-shape oracle")

    expected_shapes = {
        5: [H * W, C],
        6: [N, C, H, W],
        7: [H * W, C],
        8: [NC, H, W],
    }
    for index, expected in expected_shapes.items():
        if list(inputs[index]) != expected:
            raise ValueError(f"unexpected shape parameter {index}: {inputs[index]}")


def _oracle_relative_position_scatter_reduce(
    fma_22: torch.Tensor,
    arg13_1: torch.Tensor,
    bmm_93: torch.Tensor,
    arg186_1: torch.Tensor,
    arg6_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is not available")

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
    _relative_position_scatter_reduce_kernel[grid](
        fma_22,
        arg13_1,
        bmm_93,
        arg186_1,
        arg6_1,
        out0,
        out1,
        out2,
        fma_stride_n=fma_22.stride(0),
        fma_stride_c=fma_22.stride(1),
        fma_stride_h=fma_22.stride(2),
        fma_stride_w=fma_22.stride(3),
        index0_stride_h=arg13_1.stride(0),
        index0_stride_w=arg13_1.stride(1),
        bmm_stride_m=bmm_93.stride(0),
        bmm_stride_h=bmm_93.stride(1),
        bmm_stride_w=bmm_93.stride(2),
        softmax_stride_n=arg186_1.stride(0),
        softmax_stride_c=arg186_1.stride(1),
        softmax_stride_h=arg186_1.stride(2),
        softmax_stride_w=arg186_1.stride(3),
        index1_stride_h=arg6_1.stride(0),
        index1_stride_w=arg6_1.stride(1),
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


@oracle_impl(hardware="H100", shapes="(T([8192, 4, 49, 49], f32), T([49, 49], i64, gen=Index(169)), T([32768, 49, 49], f32), T([8192, 4, 49, 49], f32), T([49, 49], i64, gen=Index(169)), S([2401, 4]), S([8192, 4, 49, 49]), S([2401, 4]), S([32768, 49, 49]))")
def oracle_forward(inputs: tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the oracle computation over the full Repro.forward output scope."""
    _validate_inputs(inputs)
    return _oracle_relative_position_scatter_reduce(*inputs)


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
