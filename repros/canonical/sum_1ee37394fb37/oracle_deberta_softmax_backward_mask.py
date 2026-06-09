"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete DeBERTa softmax-backward masked row update returned by Repro.forward, including the metadata-only [192,512,512] -> [8,24,512,512] views, f32 dropout-mask scale, broadcast [8,1,512,512] logit mask to the f32-min sentinel before natural exp, row product reduction, exact fma.rn.f32 epilogue, final masked fill with full_1, and contiguous f32 [192,512,512] output in one Triton row kernel, whereas the shared CUDAGraph benchmark places torch.compile at the same fixed-shape row-reduction floor; Inductor cannot materially improve this local repro through scheduler fusion because the required bmm/logit/dropout/mask reads, libdevice exp, row reduction, and output write dominate; the fix is BANDWIDTH_BOUND: record as at_floor unless broader row-reduction memory-traffic or libdevice-exp work moves both paths."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:
    triton = None
    tl = None
    libdevice = None

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


BATCH = 8
HEADS = 24
Q_LEN = 512
K_LEN = 512
FLAT_BH = BATCH * HEADS
N_ROWS = FLAT_BH * Q_LEN
OUT_SHAPE = (FLAT_BH, Q_LEN, K_LEN)
OUT_STRIDE = (Q_LEN * K_LEN, K_LEN, 1)
ATTN_SHAPE = (BATCH, HEADS, Q_LEN, K_LEN)
ATTN_STRIDE = (HEADS * Q_LEN * K_LEN, Q_LEN * K_LEN, K_LEN, 1)
MASK_SHAPE = (BATCH, 1, Q_LEN, K_LEN)
MASK_STRIDE = (Q_LEN * K_LEN, Q_LEN * K_LEN, K_LEN, 1)
ROW_SHAPE = (BATCH, HEADS, Q_LEN, 1)
ROW_STRIDE = (HEADS * Q_LEN, Q_LEN, 1, 1)
LOGITS_VIEW_PARAM = (-1, HEADS, Q_LEN, K_LEN)
MASKED_LOGIT_F32 = -3.4028234663852886e38
SCALE_F32 = 1.1111111640930176
ROWS_PER_BLOCK = 1


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
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
    def oracle_kernel(
        bmm_ptr,
        dropout_mask_ptr,
        logits_ptr,
        full_mask_ptr,
        row_shift_ptr,
        row_denom_ptr,
        full_value_ptr,
        out_ptr,
        n_rows: tl.constexpr,
        q_len: tl.constexpr,
        k_len: tl.constexpr,
        heads: tl.constexpr,
        scale: tl.constexpr,
        masked_logit: tl.constexpr,
        rows_per_block: tl.constexpr,
        block_k: tl.constexpr,
    ):
        rows = tl.program_id(0) * rows_per_block + tl.arange(0, rows_per_block)
        cols = tl.arange(0, block_k)
        row_mask = rows < n_rows
        col_mask = cols < k_len
        tile_mask = row_mask[:, None] & col_mask[None, :]

        batch = rows // (heads * q_len)
        query = rows % q_len
        flat_offsets = rows[:, None] * k_len + cols[None, :]
        mask_offsets = (batch[:, None] * q_len + query[:, None]) * k_len + cols[None, :]

        bmm = tl.load(bmm_ptr + flat_offsets, mask=tile_mask, other=0.0).to(tl.float32)
        keep = tl.load(dropout_mask_ptr + flat_offsets, mask=tile_mask, other=0).to(tl.float32)
        logits = tl.load(logits_ptr + flat_offsets, mask=tile_mask, other=0.0).to(tl.float32)
        full_mask = tl.load(full_mask_ptr + mask_offsets, mask=tile_mask, other=1).to(tl.int1)
        row_shift = tl.load(row_shift_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        row_denom = tl.load(row_denom_ptr + rows, mask=row_mask, other=1.0).to(tl.float32)
        full_value = tl.load(full_value_ptr).to(tl.float32)

        masked_logits = tl.where(full_mask, masked_logit, logits)
        probs = libdevice.exp(masked_logits - row_shift[:, None]) / row_denom[:, None]

        scaled_bmm = bmm * (keep * scale)
        product = scaled_bmm * probs
        row_sum = tl.sum(tl.where(col_mask[None, :], product, 0.0), axis=1)
        fma_out = _fma_rn_f32(-probs, row_sum[:, None], product)
        out = tl.where(full_mask, full_value, fma_out)

        tl.store(out_ptr + flat_offsets, out, mask=tile_mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _validate_shape_param(name: str, actual: Any, expected: tuple[int, ...]) -> None:
    got = _shape_tuple(actual)
    if got != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {got}")


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects 10 inputs, got {len(inputs)}")

    (
        bmm_93,
        arg208_1,
        arg205_1,
        full_2,
        arg206_1,
        arg207_1,
        full_1,
        shape0,
        shape1,
        shape2,
    ) = inputs

    tensor_inputs = (
        bmm_93,
        arg208_1,
        arg205_1,
        full_2,
        arg206_1,
        arg207_1,
        full_1,
    )
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("first seven inputs must be tensors")

    expected = (
        (OUT_SHAPE, OUT_STRIDE, torch.float32),
        (ATTN_SHAPE, ATTN_STRIDE, torch.bool),
        (OUT_SHAPE, OUT_STRIDE, torch.float32),
        (MASK_SHAPE, MASK_STRIDE, torch.bool),
        (ROW_SHAPE, ROW_STRIDE, torch.float32),
        (ROW_SHAPE, ROW_STRIDE, torch.float32),
        ((), (), torch.float32),
    )
    for index, (tensor, (shape, stride, dtype)) in enumerate(zip(tensor_inputs, expected)):
        if tuple(tensor.shape) != shape or tuple(tensor.stride()) != stride or tensor.dtype != dtype:
            raise ValueError(
                f"input {index} expected shape={shape} stride={stride} dtype={dtype}, "
                f"got shape={tuple(tensor.shape)} stride={tuple(tensor.stride())} "
                f"dtype={tensor.dtype}"
            )

    _validate_shape_param("_shape_param_0", shape0, ATTN_SHAPE)
    _validate_shape_param("_shape_param_1", shape1, LOGITS_VIEW_PARAM)
    _validate_shape_param("_shape_param_2", shape2, OUT_SHAPE)

    return tensor_inputs


def _torch_full_scope(inputs: tuple[Any, ...] | list[Any]) -> torch.Tensor:
    (
        bmm_93,
        arg208_1,
        arg205_1,
        full_2,
        arg206_1,
        arg207_1,
        full_1,
        shape0,
        shape1,
        shape2,
    ) = inputs
    view_default = torch.ops.aten.view.default(bmm_93, shape0)
    convert_element_type_default = torch.ops.prims.convert_element_type.default(
        arg208_1, torch.float32
    )
    mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112)
    mul_tensor_1 = torch.ops.aten.mul.Tensor(view_default, mul_tensor)
    view_default_1 = torch.ops.aten.view.default(arg205_1, shape1)
    full_default = torch.ops.aten.full.default(
        [],
        MASKED_LOGIT_F32,
        dtype=torch.float32,
        layout=torch.strided,
        device=bmm_93.device,
        pin_memory=False,
    )
    where_self = torch.ops.aten.where.self(full_2, full_default, view_default_1)
    sub_tensor = torch.ops.aten.sub.Tensor(where_self, arg206_1)
    exp_default = torch.ops.aten.exp.default(sub_tensor)
    div_tensor = torch.ops.aten.div.Tensor(exp_default, arg207_1)
    mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, div_tensor)
    sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [-1], True)
    neg_default = torch.ops.aten.neg.default(div_tensor)
    fma_default = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_2)
    where_self_1 = torch.ops.aten.where.self(full_2, full_1, fma_default)
    return torch.ops.aten.view.default(where_self_1, shape2)


def oracle_deberta_softmax_backward_mask(
    bmm_93: torch.Tensor,
    arg208_1: torch.Tensor,
    arg205_1: torch.Tensor,
    full_2: torch.Tensor,
    arg206_1: torch.Tensor,
    arg207_1: torch.Tensor,
    full_1: torch.Tensor,
    *_shape_params: Any,
) -> torch.Tensor:
    if triton is None or libdevice is None or bmm_93.device.type != "cuda":
        return _torch_full_scope(
            (
                bmm_93,
                arg208_1,
                arg205_1,
                full_2,
                arg206_1,
                arg207_1,
                full_1,
                *_shape_params,
            )
        )

    out = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=bmm_93.device, dtype=torch.float32)
    oracle_kernel[(triton.cdiv(N_ROWS, ROWS_PER_BLOCK),)](
        bmm_93,
        arg208_1,
        arg205_1,
        full_2,
        arg206_1,
        arg207_1,
        full_1,
        out,
        n_rows=N_ROWS,
        q_len=Q_LEN,
        k_len=K_LEN,
        heads=HEADS,
        scale=SCALE_F32,
        masked_logit=MASKED_LOGIT_F32,
        rows_per_block=ROWS_PER_BLOCK,
        block_k=K_LEN,
        num_warps=8,
        num_stages=3,
    )
    return out


@oracle_impl(hardware="H100", shapes="(T([192, 512, 512], f32), T([8, 24, 512, 512], b8), T([192, 512, 512], f32), T([8, 1, 512, 512], b8), T([8, 24, 512, 1], f32), T([8, 24, 512, 1], f32), T([], f32), S([8, 24, 512, 512]), S([-1, 24, 512, 512]), S([192, 512, 512]))")
def oracle_forward(inputs):
    """Run the oracle computation for the exact Repro()(*make_inputs()) scope."""
    tensor_inputs = _validate_inputs(inputs)
    _ = get_shape_key(inputs)
    return oracle_deberta_softmax_backward_mask(*tensor_inputs, *inputs[7:])


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
