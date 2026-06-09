"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Visformer softmax-backward scale return from Repro.forward in one fixed-shape last-dimension row kernel, using the same product reduction plus epilogue product recompute needed for bit-exact Inductor numerics, explicit fma.rn.f32, final multiply-by-0.125 ordering, and a contiguous f32[768,196,196] output; the shared CUDAGraph harness places this full-scope row oracle at the torch.compile floor, so this is BANDWIDTH_BOUND rather than SCHEDULER_FUSION unless a row-reduction epilogue oracle later materially beats compile."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps syntax checks useful without Triton.
    triton = None
    tl = None


# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"


from oracle_harness import (
    oracle_impl,  # noqa: E402
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


BATCH = 128
HEADS = 6
FLAT_BATCH_HEADS = BATCH * HEADS
DEFAULT_K = 196
DEFAULT_OUT_SHAPE = (FLAT_BATCH_HEADS, DEFAULT_K, DEFAULT_K)
DEFAULT_ATTN_SHAPE = (BATCH, HEADS, DEFAULT_K, DEFAULT_K)
SCALE = 0.125


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _fma_rn_f32(a, b, c):
        return tl.inline_asm_elementwise(
            "fma.rn.f32 $0, $1, $2, $3;",
            constraints="=f,f,f,f",
            args=[a, b, c],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _visformer_softmax_backward_scale_kernel(
        bmm_ptr,
        prob_ptr,
        out_ptr,
        n_rows: tl.constexpr,
        k_len: tl.constexpr,
        rows_per_program: tl.constexpr,
        block_k: tl.constexpr,
    ):
        rows = tl.program_id(0) * rows_per_program + tl.arange(0, rows_per_program)
        cols = tl.arange(0, block_k)
        row_mask = rows < n_rows
        col_mask = cols < k_len
        offsets = rows[:, None] * k_len + cols[None, :]
        mask = row_mask[:, None] & col_mask[None, :]

        bmm = tl.load(bmm_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
        probs = tl.load(prob_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_last").to(tl.float32)

        product = bmm * probs
        acc = tl.full([rows_per_program, block_k], 0.0, tl.float32)
        acc = tl.where(mask, acc + product, acc)
        row_sum = tl.sum(acc, axis=1)

        probs_epilogue = tl.load(
            prob_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first"
        ).to(tl.float32)
        bmm_epilogue = tl.load(
            bmm_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first"
        ).to(tl.float32)
        product_epilogue = bmm_epilogue * probs_epilogue
        fma = _fma_rn_f32(-probs_epilogue, row_sum[:, None], product_epilogue)
        out = fma * 0.125

        tl.store(out_ptr + offsets, out, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride: list[int] = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...], tuple[int, ...]]:
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects four inputs, got {len(inputs)}")

    bmm_29, arg147_1, shape0, shape1 = inputs
    if not isinstance(bmm_29, torch.Tensor) or not isinstance(arg147_1, torch.Tensor):
        raise TypeError("first two inputs must be tensors")
    if bmm_29.dtype is not torch.float32 or arg147_1.dtype is not torch.float32:
        raise ValueError(f"expected f32 tensors, got {bmm_29.dtype} and {arg147_1.dtype}")
    if bmm_29.device != arg147_1.device:
        raise ValueError(f"input devices differ: {bmm_29.device} vs {arg147_1.device}")

    bmm_shape = tuple(bmm_29.shape)
    prob_shape = tuple(arg147_1.shape)
    view_shape = _shape_tuple(shape0)
    out_shape = _shape_tuple(shape1)

    if len(bmm_shape) != 3 or len(prob_shape) != 4:
        raise ValueError(f"unexpected input ranks: bmm={bmm_shape}, probs={prob_shape}")
    if prob_shape[0] != BATCH or prob_shape[1] != HEADS:
        raise ValueError(f"unexpected probability shape prefix {prob_shape[:2]}")
    if prob_shape[2] != prob_shape[3]:
        raise ValueError(f"expected square probability rows, got {prob_shape}")

    k_len = prob_shape[-1]
    expected_bmm_shape = (FLAT_BATCH_HEADS, k_len, k_len)
    expected_prob_shape = (BATCH, HEADS, k_len, k_len)
    if bmm_shape != expected_bmm_shape or prob_shape != expected_prob_shape:
        raise ValueError(
            f"shape mismatch: expected bmm={expected_bmm_shape} probs={expected_prob_shape}, "
            f"got bmm={bmm_shape} probs={prob_shape}"
        )
    if view_shape != expected_prob_shape or out_shape != expected_bmm_shape:
        raise ValueError(
            f"shape params mismatch: expected {expected_prob_shape} and {expected_bmm_shape}, "
            f"got {view_shape} and {out_shape}"
        )

    expected_bmm_stride = _contiguous_stride(expected_bmm_shape)
    expected_prob_stride = _contiguous_stride(expected_prob_shape)
    if tuple(bmm_29.stride()) != expected_bmm_stride:
        raise ValueError(f"bmm_29 must be contiguous with stride {expected_bmm_stride}, got {tuple(bmm_29.stride())}")
    if tuple(arg147_1.stride()) != expected_prob_stride:
        raise ValueError(f"arg147_1 must be contiguous with stride {expected_prob_stride}, got {tuple(arg147_1.stride())}")

    return bmm_29, arg147_1, view_shape, out_shape


def _torch_full_scope(inputs: tuple[Any, ...] | list[Any]) -> torch.Tensor:
    bmm_29, arg147_1, shape0, shape1 = inputs
    view_default = torch.ops.aten.view.default(bmm_29, shape0)
    mul_tensor = torch.ops.aten.mul.Tensor(view_default, arg147_1)
    sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [-1], True)
    neg_default = torch.ops.aten.neg.default(arg147_1)
    fma_default = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor)
    mul_tensor_1 = torch.ops.aten.mul.Tensor(fma_default, SCALE)
    return torch.ops.aten.view.default(mul_tensor_1, shape1)


@oracle_impl(hardware="H100", shapes="(T([768, 196, 196], f32), T([128, 6, 196, 196], f32), S([128, 6, 196, 196]), S([768, 196, 196]))")
def oracle_forward(inputs):
    """Run the oracle computation for the exact Repro()(*make_inputs()) scope."""
    bmm_29, arg147_1, _view_shape, out_shape = _validate_inputs(inputs)

    if triton is None or bmm_29.device.type != "cuda":
        return _torch_full_scope(inputs)

    k_len = int(bmm_29.shape[-1])
    n_rows = FLAT_BATCH_HEADS * k_len
    block_k = _next_power_of_2(k_len)
    rows_per_program = 2

    out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=bmm_29.device,
        dtype=torch.float32,
    )
    grid = (triton.cdiv(n_rows, rows_per_program),)
    _visformer_softmax_backward_scale_kernel[grid](
        bmm_29,
        arg147_1,
        out,
        n_rows=n_rows,
        k_len=k_len,
        rows_per_program=rows_per_program,
        block_k=block_k,
        num_warps=2,
    )
    return out


def _check_output_layout(output: torch.Tensor, expected_shape: tuple[int, ...]) -> bool:
    return (
        tuple(output.shape) == expected_shape
        and tuple(output.stride()) == _contiguous_stride(expected_shape)
        and output.dtype is torch.float32
        and output.storage_offset() == 0
    )


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

    if args.show_hw:
        import json

        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

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
        with torch.no_grad():
            layout_output = oracle_forward(inputs)
            if layout_output.is_cuda:
                torch.cuda.synchronize()
        expected_shape = _shape_tuple(inputs[3])
        layout_ok = _check_output_layout(layout_output, expected_shape)
        print(
            f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(layout_output.shape)} stride={list(layout_output.stride())})"
        )
        ok = ok and layout_ok
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
