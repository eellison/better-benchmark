"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileBERT softmax-backward return from Repro.forward in one last-dimension row kernel, folding the BMM/dropout mask scaling, broadcast attention mask, natural-exp probability reconstruction, row dot-product reduction, row-mask zeroing, exact final fma epilogue, and final contiguous [1024,128,128] view; Inductor currently lowers the decomposed view/where/add/exp/div/where/mul/sum/fma graph as separate generic producers and reduction/epilogue work over materialized [256,4,128,128] values; Inductor cannot do this today because scheduler fusion does not sink the softmax-backward row reduction through the probability-reconstruction producer and its epilogue consumer while preserving the mask broadcasts and fma boundary; the fix is SCHEDULER_FUSION: allow this row-reduction plus epilogue fusion for decomposed softmax-backward tails with exact libdevice exp and fma.rn semantics."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 256
HEADS = 4
Q_LEN = 128
K_LEN = 128
FLAT_BH = BATCH * HEADS
N_ROWS = FLAT_BH * Q_LEN
OUT_SHAPE = (FLAT_BH, Q_LEN, K_LEN)
OUT_STRIDE = (Q_LEN * K_LEN, K_LEN, 1)
ATTN_SHAPE = (BATCH, HEADS, Q_LEN, K_LEN)
EXPAND_MASK_SHAPE = (BATCH, -1, Q_LEN, K_LEN)
ROW_SHAPE = (BATCH, HEADS, Q_LEN, 1)
SCALE_F32 = 1.1111111640930176


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
    def _mobilebert_softmax_backward_kernel(
        bmm_ptr,
        dropout_mask_ptr,
        attn_mask_ptr,
        full_ptr,
        logits_ptr,
        row_shift_ptr,
        row_denom_ptr,
        prob_zero_mask_ptr,
        out_ptr,
        n_rows: tl.constexpr,
        q_len: tl.constexpr,
        k_len: tl.constexpr,
        scale: tl.constexpr,
        block_m: tl.constexpr,
        block_k: tl.constexpr,
    ):
        rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
        cols = tl.arange(0, block_k)
        row_mask = rows < n_rows
        col_mask = cols < k_len
        tile_mask = row_mask[:, None] & col_mask[None, :]

        q_index = rows % q_len
        flat_offsets = rows[:, None] * k_len + cols[None, :]
        row_offsets = rows

        bmm = tl.load(bmm_ptr + flat_offsets, mask=tile_mask, other=0.0).to(tl.float32)
        dropout_keep = tl.load(
            dropout_mask_ptr + flat_offsets, mask=tile_mask, other=0
        ).to(tl.float32)
        logits = tl.load(logits_ptr + flat_offsets, mask=tile_mask, other=0.0).to(tl.float32)

        full_value = tl.load(full_ptr).to(tl.float32)
        attn_keep = tl.load(attn_mask_ptr + q_index, mask=row_mask, other=0).to(tl.int1)
        mask_bias = tl.where(attn_keep[:, None], full_value, -float("inf"))
        shifted = (logits + mask_bias) - tl.load(
            row_shift_ptr + row_offsets, mask=row_mask, other=0.0
        ).to(tl.float32)[:, None]

        numer = tl.extra.libdevice.exp(shifted)
        probs = numer / tl.load(row_denom_ptr + row_offsets, mask=row_mask, other=1.0).to(
            tl.float32
        )[:, None]
        zero_row = tl.load(prob_zero_mask_ptr + row_offsets, mask=row_mask, other=1).to(
            tl.int1
        )
        probs = tl.where(zero_row[:, None], 0.0, probs)

        scaled_bmm = bmm * (dropout_keep * scale)
        product = scaled_bmm * probs
        row_sum = tl.sum(tl.where(col_mask[None, :], product, 0.0), axis=1)
        out = _fma_rn_f32(-probs, row_sum[:, None], product)

        tl.store(out_ptr + flat_offsets, out, mask=tile_mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _validate_shape_param(name: str, actual: Any, expected: tuple[int, ...]) -> None:
    got = _shape_tuple(actual)
    if got != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {got}")


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 12:
        raise ValueError(f"{REPRO_ID} expects 12 inputs, got {len(inputs)}")

    (
        bmm_93,
        arg595_1,
        arg586_1,
        full_1,
        arg591_1,
        arg592_1,
        arg593_1,
        arg594_1,
        shape0,
        shape1,
        shape2,
        shape3,
    ) = inputs

    tensor_inputs = (
        bmm_93,
        arg595_1,
        arg586_1,
        full_1,
        arg591_1,
        arg592_1,
        arg593_1,
        arg594_1,
    )
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("first eight inputs must be tensors")

    expected = (
        (OUT_SHAPE, OUT_STRIDE, torch.float32),
        (ATTN_SHAPE, (HEADS * Q_LEN * K_LEN, Q_LEN * K_LEN, K_LEN, 1), torch.bool),
        ((1, 1, Q_LEN, 1), (Q_LEN, Q_LEN, 1, 1), torch.bool),
        ((), (), torch.float32),
        (OUT_SHAPE, OUT_STRIDE, torch.float32),
        (ROW_SHAPE, (HEADS * Q_LEN, Q_LEN, 1, 1), torch.float32),
        (ROW_SHAPE, (HEADS * Q_LEN, Q_LEN, 1, 1), torch.float32),
        (ROW_SHAPE, (HEADS * Q_LEN, Q_LEN, 1, 1), torch.bool),
    )
    for index, (tensor, (shape, stride, dtype)) in enumerate(zip(tensor_inputs, expected)):
        if tuple(tensor.shape) != shape or tuple(tensor.stride()) != stride or tensor.dtype != dtype:
            raise ValueError(
                f"input {index} expected shape={shape} stride={stride} dtype={dtype}, "
                f"got shape={tuple(tensor.shape)} stride={tuple(tensor.stride())} "
                f"dtype={tensor.dtype}"
            )

    _validate_shape_param("_shape_param_0", shape0, ATTN_SHAPE)
    _validate_shape_param("_shape_param_1", shape1, EXPAND_MASK_SHAPE)
    _validate_shape_param("_shape_param_2", shape2, ATTN_SHAPE)
    _validate_shape_param("_shape_param_3", shape3, OUT_SHAPE)

    return tensor_inputs


def _torch_full_scope(inputs: tuple[Any, ...] | list[Any]) -> torch.Tensor:
    (
        bmm_93,
        arg595_1,
        arg586_1,
        full_1,
        arg591_1,
        arg592_1,
        arg593_1,
        arg594_1,
        shape0,
        shape1,
        shape2,
        shape3,
    ) = inputs
    view_default = torch.ops.aten.view.default(bmm_93, shape0)
    convert_element_type_default = torch.ops.prims.convert_element_type.default(
        arg595_1, torch.float32
    )
    mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112)
    mul_tensor_1 = torch.ops.aten.mul.Tensor(view_default, mul_tensor)
    expand_default = torch.ops.aten.expand.default(arg586_1, shape1)
    full_default = torch.ops.aten.full.default(
        [],
        -float("inf"),
        dtype=torch.float32,
        layout=torch.strided,
        device=bmm_93.device,
        pin_memory=False,
    )
    where_self = torch.ops.aten.where.self(expand_default, full_1, full_default)
    view_default_1 = torch.ops.aten.view.default(arg591_1, shape2)
    add_tensor = torch.ops.aten.add.Tensor(view_default_1, where_self)
    sub_tensor = torch.ops.aten.sub.Tensor(add_tensor, arg592_1)
    exp_default = torch.ops.aten.exp.default(sub_tensor)
    div_tensor = torch.ops.aten.div.Tensor(exp_default, arg593_1)
    full_default_1 = torch.ops.aten.full.default(
        ATTN_SHAPE,
        0,
        dtype=torch.float32,
        layout=torch.strided,
        device=bmm_93.device,
        pin_memory=False,
    )
    where_self_1 = torch.ops.aten.where.self(arg594_1, full_default_1, div_tensor)
    mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, where_self_1)
    sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [-1], True)
    neg_default = torch.ops.aten.neg.default(where_self_1)
    fma_default = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_2)
    return torch.ops.aten.view.default(fma_default, shape3)


def oracle_mobilebert_softmax_backward(
    bmm_93: torch.Tensor,
    arg595_1: torch.Tensor,
    arg586_1: torch.Tensor,
    full_1: torch.Tensor,
    arg591_1: torch.Tensor,
    arg592_1: torch.Tensor,
    arg593_1: torch.Tensor,
    arg594_1: torch.Tensor,
    *_shape_params: Any,
    block_m: int = 16,
    block_k: int = K_LEN,
    num_warps: int = 4,
) -> torch.Tensor:
    if triton is None or bmm_93.device.type != "cuda":
        return _torch_full_scope(
            (
                bmm_93,
                arg595_1,
                arg586_1,
                full_1,
                arg591_1,
                arg592_1,
                arg593_1,
                arg594_1,
                *_shape_params,
            )
        )
    if block_k != K_LEN:
        raise ValueError(f"block_k must equal {K_LEN}, got {block_k}")
    if block_m <= 0 or N_ROWS % block_m != 0:
        raise ValueError(f"block_m must evenly divide {N_ROWS}, got {block_m}")

    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=bmm_93.device,
        dtype=torch.float32,
    )
    _mobilebert_softmax_backward_kernel[(N_ROWS // block_m,)](
        bmm_93,
        arg595_1,
        arg586_1,
        full_1,
        arg591_1,
        arg592_1,
        arg593_1,
        arg594_1,
        out,
        n_rows=N_ROWS,
        q_len=Q_LEN,
        k_len=K_LEN,
        scale=SCALE_F32,
        block_m=block_m,
        block_k=block_k,
        num_warps=num_warps,
    )
    return out


@oracle_impl(hardware="H100", shapes="(T([1024, 128, 128], f32), T([256, 4, 128, 128], b8), T([1, 1, 128, 1], b8), T([], f32), T([1024, 128, 128], f32), T([256, 4, 128, 1], f32), T([256, 4, 128, 1], f32), T([256, 4, 128, 1], b8), S([256, 4, 128, 128]), S([256, -1, 128, 128]), S([256, 4, 128, 128]), S([1024, 128, 128]))")
def oracle_forward(inputs):
    """Run the oracle computation for the exact Repro()(*make_inputs()) scope."""
    tensor_inputs = _validate_inputs(inputs)
    return oracle_mobilebert_softmax_backward(*tensor_inputs, *inputs[8:])


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
