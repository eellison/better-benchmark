"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Whisper residual-add, fp16 clamp/canonicalization, contiguous-clone-equivalent row source, fp32 population var_mean, eps=1e-5 affine LayerNorm, fp16 cast, and final `[rows, hidden]` view in one Triton row kernel that reads the strided residual input directly, whereas Inductor currently lowers this norm-template case through generic normalization scheduling for the add/canonicalization/source-layout chain; Inductor cannot do this today because norm-template canonicalization does not preserve this non-contiguous residual producer plus fp16 clamp/cast clone as a single fixed-hidden LayerNorm input; the fix is SCHEDULER_FUSION: teach the normalization scheduler to canonicalize the strided residual-add and fp16 clamp/cast producer into the row-wise LayerNorm lowering."""
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


EPS = 1.0e-5
CLAMP_MIN = -64504.0
CLAMP_MAX = 64504.0
CLASSIFICATION = "SCHEDULER_FUSION"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides: list[int] = []
    stride = 1
    for size in reversed(shape):
        strides.append(stride)
        stride *= max(int(size), 1)
    return tuple(reversed(strides))


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int, int],
    tuple[int, int],
]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    addmm, residual, weight, bias, shape0, shape1 = inputs
    tensor_inputs = (addmm, residual, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first four repro inputs must be tensors")

    shape0_tuple = _shape_tuple(shape0)
    shape1_tuple = _shape_tuple(shape1)
    if len(shape0_tuple) != 3 or len(shape1_tuple) != 2:
        raise ValueError(f"unexpected shape parameters: {shape0!r}, {shape1!r}")

    batch, seq_len, hidden = shape0_tuple
    rows, out_hidden = shape1_tuple
    if rows != batch * seq_len or out_hidden != hidden:
        raise ValueError(f"shape parameters are inconsistent: {shape0_tuple}, {shape1_tuple}")

    expected_shapes = (shape1_tuple, shape0_tuple, (hidden,), (hidden,))
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")
        if value.dtype != torch.float16:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float16")
        if value.device != addmm.device:
            raise ValueError(f"input {index} device {value.device} != {addmm.device}")

    if not addmm.is_contiguous():
        raise ValueError(f"addmm input must be contiguous, got stride={addmm.stride()}")
    if not weight.is_contiguous() or not bias.is_contiguous():
        raise ValueError("affine weight and bias must be contiguous")

    residual_stride = tuple(int(stride) for stride in residual.stride())
    expected_residual_stride = (seq_len * hidden, 1, seq_len)
    if residual_stride != expected_residual_stride:
        raise ValueError(
            f"residual stride {residual_stride} != expected {expected_residual_stride}"
        )

    return addmm, residual, weight, bias, shape0_tuple, shape1_tuple


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"ROW_BLOCK": 1}, num_warps=1, num_stages=3),
            triton.Config({"ROW_BLOCK": 1}, num_warps=2, num_stages=3),
            triton.Config({"ROW_BLOCK": 1}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=1, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=2, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 4}, num_warps=1, num_stages=3),
            triton.Config({"ROW_BLOCK": 4}, num_warps=2, num_stages=3),
            triton.Config({"ROW_BLOCK": 4}, num_warps=4, num_stages=3),
        ],
        key=["hidden"],
    )
    @triton.jit
    def _whisper_layernorm_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        total_rows: tl.constexpr,
        seq_len: tl.constexpr,
        hidden: tl.constexpr,
        residual_stride_b: tl.constexpr,
        residual_stride_s: tl.constexpr,
        residual_stride_h: tl.constexpr,
        eps: tl.constexpr,
        clamp_min: tl.constexpr,
        clamp_max: tl.constexpr,
        block_h: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        rows = row_ids[:, None]
        cols = tl.arange(0, block_h)[None, :]
        row_mask = row_ids[:, None] < total_rows
        col_mask = cols < hidden
        mask = row_mask & col_mask

        batch_ids = row_ids // seq_len
        seq_ids = row_ids - batch_ids * seq_len
        residual_offsets = (
            batch_ids[:, None] * residual_stride_b
            + seq_ids[:, None] * residual_stride_s
            + cols * residual_stride_h
        )
        contiguous_offsets = rows * hidden + cols

        addmm = tl.load(
            addmm_ptr + contiguous_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        )
        residual = tl.load(
            residual_ptr + residual_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        )

        # Reproduce add -> f32 -> clamp -> f16 -> clone -> f32 before var_mean.
        summed = (addmm + residual).to(tl.float16).to(tl.float32)
        clamped = tl.minimum(tl.maximum(summed, clamp_min), clamp_max)
        x = clamped.to(tl.float16).to(tl.float32)
        x_for_reduce = tl.where(mask, x, 0.0)

        mean = tl.sum(x_for_reduce, axis=1)[:, None] / hidden
        mean_sq = tl.sum(x_for_reduce * x_for_reduce, axis=1)[:, None] / hidden
        variance = mean_sq - mean * mean
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(
            weight_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        bias = tl.load(
            bias_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        y = (x - mean) * invstd * weight + bias
        tl.store(output_ptr + contiguous_offsets, y, mask=mask)


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    addmm, residual, weight, bias, shape0, shape1 = _validate_inputs(inputs)
    view_default = torch.ops.aten.view.default(addmm, shape0)
    add_tensor = torch.ops.aten.add.Tensor(residual, view_default)
    convert_element_type_default = torch.ops.prims.convert_element_type.default(
        add_tensor, torch.float32
    )
    clamp_min_default = torch.ops.aten.clamp_min.default(
        convert_element_type_default, CLAMP_MIN
    )
    clamp_max_default = torch.ops.aten.clamp_max.default(clamp_min_default, CLAMP_MAX)
    convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(
        clamp_max_default, torch.float16
    )
    clone_default = torch.ops.aten.clone.default(
        convert_element_type_default_1, memory_format=torch.contiguous_format
    )
    convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(
        clone_default, torch.float32
    )
    var, mean = torch.ops.aten.var_mean.correction(
        convert_element_type_default_2, [2], correction=0, keepdim=True
    )
    normalized = torch.ops.aten.mul.Tensor(
        torch.ops.aten.sub.Tensor(convert_element_type_default_2, mean),
        torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(var, EPS)),
    )
    affine = torch.ops.aten.add.Tensor(
        torch.ops.aten.mul.Tensor(normalized, weight),
        bias,
    )
    output = torch.ops.prims.convert_element_type.default(affine, torch.float16)
    return torch.ops.aten.view.default(output, shape1)


@oracle_impl(hardware="H100", shapes="(T([12000, 384], f16), T([8, 1500, 384], f16, stride=(576000, 1, 1500)), T([384], f16), T([384], f16), S([8, 1500, 384]), S([12000, 384]))")
def oracle_forward(inputs):
    """Run the full Repro.forward computation with a fused row LayerNorm kernel.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns the
    same single fp16 `[rows, hidden]` contiguous output tensor.
    """
    addmm, residual, weight, bias, shape0, shape1 = _validate_inputs(inputs)
    if triton is None or not addmm.is_cuda:
        return _torch_full_scope(inputs)

    batch, seq_len, hidden = shape0
    del batch
    rows, _out_hidden = shape1
    output = torch.empty_strided(
        shape1,
        _contiguous_strides(shape1),
        device=addmm.device,
        dtype=torch.float16,
    )
    block_h = triton.next_power_of_2(hidden)
    grid = lambda meta: (triton.cdiv(rows, meta["ROW_BLOCK"]),)
    _whisper_layernorm_kernel[grid](
        addmm,
        residual,
        weight,
        bias,
        output,
        total_rows=rows,
        seq_len=seq_len,
        hidden=hidden,
        residual_stride_b=int(residual.stride(0)),
        residual_stride_s=int(residual.stride(1)),
        residual_stride_h=int(residual.stride(2)),
        eps=EPS,
        clamp_min=CLAMP_MIN,
        clamp_max=CLAMP_MAX,
        block_h=block_h,
    )
    return output


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
