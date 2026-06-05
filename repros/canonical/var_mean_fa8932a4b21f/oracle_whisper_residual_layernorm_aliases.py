"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Whisper residual LayerNorm scope in one Triton row kernel, including the strided fp16 residual add, fp16-range clamp and clone round-trip, fp32 population var_mean over the hidden dimension, affine scale/bias, fp16 cast, and three aliasing `[12000,384]`/shape-param views over one final buffer, whereas Inductor leaves extra scheduling overhead around the explicit clamp/clone producer before the normalization template; Inductor cannot do this today because norm-template canonicalization does not absorb this precision-preserving fp16 clamp/clone residual producer and aliasing view fanout into the fixed-hidden row reduction as one scheduled region; the fix is SCHEDULER_FUSION: extend the LayerNorm scheduler pattern to fuse the clamped fp16 residual producer and final aliasing views into the row-normalization kernel."""
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

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

EPS = 1.0e-5
CLAMP_BOUND = 64504.0

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
    def _residual_layernorm_aliases_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        total_rows: tl.constexpr,
        seq_len: tl.constexpr,
        hidden: tl.constexpr,
        residual_stride_b: tl.constexpr,
        residual_stride_t: tl.constexpr,
        residual_stride_h: tl.constexpr,
        eps: tl.constexpr,
        clamp_bound: tl.constexpr,
        block_h: tl.constexpr,
        row_block: tl.constexpr,
    ):
        rows = tl.program_id(0) * row_block + tl.arange(0, row_block)[:, None]
        cols = tl.arange(0, block_h)[None, :]
        row_mask = rows < total_rows
        col_mask = cols < hidden
        mask = row_mask & col_mask

        batch_idx = rows // seq_len
        token_idx = rows - batch_idx * seq_len
        addmm_offsets = rows * hidden + cols
        residual_offsets = (
            batch_idx * residual_stride_b
            + token_idx * residual_stride_t
            + cols * residual_stride_h
        )

        addmm = tl.load(addmm_ptr + addmm_offsets, mask=mask, other=0.0)
        residual = tl.load(residual_ptr + residual_offsets, mask=mask, other=0.0)

        # aten.add on fp16 materializes fp16 before the explicit fp32 clamp.
        x = (addmm + residual).to(tl.float16).to(tl.float32)
        x = tl.where(x < -clamp_bound, -clamp_bound, x)
        x = tl.where(x > clamp_bound, clamp_bound, x).to(tl.float16).to(tl.float32)

        x_for_reduce = tl.where(mask, x, 0.0)
        mean = tl.sum(x_for_reduce, axis=1)[:, None] / hidden
        centered = x - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=1)[:, None] / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        y = (centered * invstd * weight + bias).to(tl.float16)
        tl.store(out_ptr + addmm_offsets, y, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _numel_from_shape(shape: tuple[int, ...], known_numel: int) -> int:
    product = 1
    inferred_dims = 0
    for dim in shape:
        if dim == -1:
            inferred_dims += 1
        else:
            product *= dim
    if inferred_dims > 1:
        raise ValueError(f"shape has more than one inferred dimension: {shape}")
    if inferred_dims == 0:
        return product
    if known_numel % product != 0:
        raise ValueError(f"shape {shape} cannot represent {known_numel} elements")
    return known_numel


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int, int],
    tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    addmm, residual, weight, bias, shape0, shape1, shape2, shape3 = inputs
    tensor_inputs = (addmm, residual, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first four repro inputs must be tensors")

    view_shape = _shape_tuple(shape0)
    output_shapes = (_shape_tuple(shape1), _shape_tuple(shape2), _shape_tuple(shape3))
    if len(view_shape) != 3:
        raise ValueError(f"first view shape must be rank 3, got {view_shape}")
    batch, seq_len, hidden = view_shape
    rows = batch * seq_len
    total = rows * hidden

    expected_tensor_shapes = (
        (rows, hidden),
        view_shape,
        (hidden,),
        (hidden,),
    )
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_tensor_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")
        if value.dtype != torch.float16:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float16")
        if value.device != addmm.device:
            raise ValueError(f"input {index} device {value.device} != {addmm.device}")

    if not addmm.is_contiguous():
        raise ValueError(f"addmm input must be contiguous, got stride={addmm.stride()}")
    if not weight.is_contiguous() or not bias.is_contiguous():
        raise ValueError("weight and bias inputs must be contiguous")
    for index, shape in enumerate(output_shapes, start=1):
        if _numel_from_shape(shape, total) != total:
            raise ValueError(f"output shape parameter {index} {shape} does not match {total} elements")
        if len(shape) != 2 or shape[-1] != hidden:
            raise ValueError(f"unexpected output shape parameter {index}: {shape}")

    return addmm, residual, weight, bias, view_shape, output_shapes


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    addmm, residual, weight, bias, shape0, output_shapes = _validate_inputs(inputs)
    view_default = torch.ops.aten.view.default(addmm, shape0)
    add_tensor = torch.ops.aten.add.Tensor(residual, view_default)
    add_f32 = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32)
    clamp_min = torch.ops.aten.clamp_min.default(add_f32, -CLAMP_BOUND)
    clamp_max = torch.ops.aten.clamp_max.default(clamp_min, CLAMP_BOUND)
    clamped_f16 = torch.ops.prims.convert_element_type.default(clamp_max, torch.float16)
    clone = torch.ops.aten.clone.default(clamped_f16, memory_format=torch.contiguous_format)
    x = torch.ops.prims.convert_element_type.default(clone, torch.float32)
    var, mean = torch.ops.aten.var_mean.correction(x, [2], correction=0, keepdim=True)
    normalized = (x - mean) * torch.ops.aten.rsqrt.default(var + EPS)
    out_base = (normalized * weight + bias).to(torch.float16)
    return tuple(torch.ops.aten.view.default(out_base, shape) for shape in output_shapes)


def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same three fp16 view outputs. The returned views alias one contiguous
    `[batch, seq, hidden]` base buffer, matching the repro's final view fanout.
    """
    addmm, residual, weight, bias, view_shape, output_shapes = _validate_inputs(inputs)
    if triton is None or not addmm.is_cuda:
        return _torch_full_scope(inputs)

    batch, seq_len, hidden = view_shape
    rows = batch * seq_len
    block_h = triton.next_power_of_2(hidden)
    row_block = 4 if hidden <= 512 else 1
    out_base = torch.empty_strided(
        view_shape,
        (seq_len * hidden, hidden, 1),
        device=addmm.device,
        dtype=torch.float16,
    )

    _residual_layernorm_aliases_kernel[(triton.cdiv(rows, row_block),)](
        addmm,
        residual,
        weight,
        bias,
        out_base,
        rows,
        seq_len,
        hidden,
        residual.stride(0),
        residual.stride(1),
        residual.stride(2),
        EPS,
        CLAMP_BOUND,
        block_h,
        row_block,
        num_warps=4 if hidden <= 512 else 8,
        num_stages=3,
    )
    return tuple(out_base.view(shape) for shape in output_shapes)


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
