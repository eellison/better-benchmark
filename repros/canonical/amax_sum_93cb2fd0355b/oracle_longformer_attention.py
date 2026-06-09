"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Longformer attention-score assembly, two mask substitutions, position-bias add, row softmax, dropout, and final skew/pad/permute output surface with one fused Triton row kernel plus an output zero-fill, whereas Inductor materializes and schedules the captured constant_pad/view/slice_scatter/permute/clone chain around the softmax as generic tensor operations; Inductor cannot do this today because it has no Longformer diagonal-skew/banded-attention pattern that rewrites the slice-scatter assembly and final skew into direct indexed loads and stores across the reduction; the fix is NEW_PATTERN: add a graph rewrite or lowering for this Longformer sliding-window attention pattern that fuses band assembly, bias/masking, online softmax/dropout, and skewed output layout generation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401 - registers prims.* RNG ops

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
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


BATCH = 2
HEADS = 12
SEQ = 1024
WINDOW = 513
CHUNKS = 4
CHUNK = 256
LOCAL_HEADS = 3
Q_HEADS = 24
OUT_Q = 96
OUT_X = 768
OUT_Y = 256
SOFTMAX_ROWS = OUT_Q * OUT_Y
OUT_STRIDE_Q = 197120
OUT_STRIDE_X = 1
OUT_STRIDE_Y = 769
OUT_STORAGE = (OUT_Q - 1) * OUT_STRIDE_Q + (OUT_X - 1) * OUT_STRIDE_X + (OUT_Y - 1) * OUT_STRIDE_Y + 1
BLOCK_W = 1024
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
SEED_INDEX = 33
STOCHASTIC_OUTPUTS = (0,)
CLASSIFICATION = "NEW_PATTERN"


if triton is not None:

    @triton.jit
    def _skew_value(bmm_ptr, a, local_h, row, col):
        flat = row * 513 + col
        src_row = flat // 512
        src_col = flat - src_row * 512
        valid = (
            (local_h >= 0)
            & (local_h < 3)
            & (row >= 0)
            & (row < 512)
            & (col >= 0)
            & (col < 513)
            & (src_row >= 0)
            & (src_row < 512)
        )
        offset = ((a * 3 + local_h) * 512 + src_row) * 512 + src_col
        return tl.load(bmm_ptr + offset, mask=valid, other=0.0).to(tl.float32)

    @triton.jit
    def _zero_kernel(out_ptr, total: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(out_ptr + offsets, tl.zeros((BLOCK,), tl.float32), mask=offsets < total)

    @triton.jit
    def _longformer_softmax_dropout_kernel(
        bmm_ptr,
        slice4_ptr,
        slice3_ptr,
        full_ptr,
        mask0_ptr,
        permute12_ptr,
        mask1_ptr,
        bias_ptr,
        global_mask_ptr,
        full2_ptr,
        seeds_ptr,
        out_ptr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        q = pid // 256
        r = pid - q * 256
        a = q // 4
        c = q - a * 4
        n = a // 12
        h = a - n * 12
        pos = c * 256 + r
        cols = tl.arange(0, BLOCK)
        in_window = cols < 513

        full_off = ((a * 4 + c) * 256 + r) * 513 + cols
        score = tl.load(full_ptr + full_off, mask=in_window, other=0.0).to(tl.float32)

        slice3_c = tl.minimum(c, 2)
        slice3_off = a * 525312 + slice3_c * 131328 + r * 513 + cols
        slice3_val = tl.load(
            slice3_ptr + slice3_off,
            mask=in_window & (c < 3),
            other=0.0,
        ).to(tl.float32)

        first_high = _skew_value(bmm_ptr, a, c, r, cols - 256)
        chan3_high = _skew_value(bmm_ptr, a, 2, r + 256, cols - 256)
        high_val = tl.where(c < 3, first_high, chan3_high)
        score = tl.where((c < 3) & (cols < 256), slice3_val, score)
        score = tl.where(cols >= 256, high_val, score)

        middle_low = _skew_value(bmm_ptr, a, c - 1, r + 255, cols + 257)
        score = tl.where((c >= 1) & (cols < 256), middle_low, score)

        chan0_inner = _skew_value(bmm_ptr, a, 0, r - 1, cols + 257)
        score = tl.where((c == 0) & (r >= 1) & (cols >= 1) & (cols < 256), chan0_inner, score)

        mask0_off = n * 789504 + r * 3084 + h * 257 + cols
        perm12_off = n * 789504 + r * 257 + h * 65792 + cols
        m0 = tl.load(mask0_ptr + mask0_off, mask=(pos < 256) & (cols < 257), other=0)
        p0 = tl.load(permute12_ptr + perm12_off, mask=(pos < 256) & (cols < 257), other=0.0).to(tl.float32)
        score = tl.where((pos < 256) & (cols < 257) & m0, p0, score)

        tail_pos = pos - 768
        tail_col = cols - 256
        mask1_off = n * 789504 + tail_pos * 3084 + h * 257 + tail_col
        perm12_tail_off = n * 789504 + tail_pos * 257 + h * 65792 + tail_col
        tail_region = (pos >= 768) & (cols >= 256)
        m1 = tl.load(mask1_ptr + mask1_off, mask=tail_region, other=0)
        p1 = tl.load(permute12_ptr + perm12_tail_off, mask=tail_region, other=0.0).to(tl.float32)
        score = tl.where(tail_region & m1, p1, score)

        bias_off = (n * 1024 + pos) * 513 + cols
        bias = tl.load(bias_ptr + bias_off, mask=in_window, other=0.0).to(tl.float32)
        score = tl.where(in_window, score + bias, -float("inf"))

        row_max = tl.max(score, axis=0)
        numer = tl.exp(score - row_max)
        denom = tl.sum(numer, axis=0)
        softmax = numer / denom

        global_mask = tl.load(global_mask_ptr + n * 1024 + pos, mask=True, other=0)
        fill_value = tl.load(full2_ptr).to(tl.float32)
        value = tl.where(global_mask, fill_value, softmax)

        seed = tl.load(seeds_ptr + 33)
        random_offsets = (pid * 513 + cols).to(tl.uint32)
        keep = tl.rand(seed, random_offsets) > 0.1
        value = value * keep.to(tl.float32) * 1.1111111111111112

        out_x = r + cols
        out_off = q * 197120 + out_x + r * 769
        tl.store(out_ptr + out_off, value, mask=in_window)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 21:
        raise ValueError(f"{REPRO_ID} expects 21 inputs, got {len(inputs)}")

    tensors = inputs[:11]
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("the first eleven repro inputs must be tensors")
    if not all(value.is_cuda for value in tensors):
        raise RuntimeError("CUDA tensors are required for the Triton oracle")

    expected_shapes = (
        (72, 512, 512),
        (24, 3, 256, 257),
        (24, 3, 256, 513),
        (24, 4, 256, 513),
        (2, 256, 12, 257),
        (2, 256, 12, 257),
        (2, 256, 12, 257),
        (2, 1024, 1, 513),
        (2, 1024, 1, 1),
        (),
        (36,),
    )
    expected_dtypes = (
        torch.float32,
        torch.float32,
        torch.float32,
        torch.float32,
        torch.bool,
        torch.float32,
        torch.bool,
        torch.float32,
        torch.bool,
        torch.float32,
        torch.int64,
    )
    for index, (value, shape, dtype) in enumerate(zip(tensors, expected_shapes, expected_dtypes)):
        if tuple(value.shape) != shape:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {shape}")
        if value.dtype != dtype:
            raise TypeError(f"input {index} must be {dtype}, got {value.dtype}")

    expected_shape_params = (
        (24, 3, 512, 1, 512),
        (24, 3, 512, 512),
        (24, 3, 512, 513),
        (2, 12, 1024, 513),
        (24, 4, 256, 513),
        (2, 12, 1024, 513),
        (24, 4, 256, 513),
        (24, 4, -1),
        (24, 4, 256, 769),
        (96, 256, 768),
    )
    for index, (value, expected) in enumerate(zip(inputs[11:], expected_shape_params), start=11):
        shape = _shape_tuple(value)
        if shape != expected:
            raise ValueError(f"shape input {index} is {shape}, expected {expected}")

    return tensors  # type: ignore[return-value]


@oracle_impl(hardware="H100", shapes="(T([72, 512, 512], f32), T([24, 3, 256, 257], f32, stride=(525312, 131328, 513, 1)), T([24, 3, 256, 513], f32, stride=(525312, 131328, 513, 1)), T([24, 4, 256, 513], f32), T([2, 256, 12, 257], b8), T([2, 256, 12, 257], f32, stride=(789504, 257, 65792, 1)), T([2, 256, 12, 257], b8), T([2, 1024, 1, 513], f32), T([2, 1024, 1, 1], b8), T([], f32), T([36], i64), S([24, 3, 512, 1, 512]), S([24, 3, 512, 512]), S([24, 3, 512, 513]), S([2, 12, 1024, 513]), S([24, 4, 256, 513]), S([2, 12, 1024, 513]), S([24, 4, 256, 513]), S([24, 4, -1]), S([24, 4, 256, 769]), S([96, 256, 768]))")
def oracle_forward(inputs):
    """Run the complete Repro.forward scope with fused Longformer indexing."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_longformer_attention.py")

    (
        bmm_22,
        slice_4,
        slice_3,
        full,
        convert_element_type,
        permute_12,
        convert_element_type_1,
        permute_25,
        unsqueeze_11,
        full_2,
        inductor_seeds,
    ) = _validate_inputs(inputs)

    output = torch.empty_strided(
        (OUT_Q, OUT_X, OUT_Y),
        (OUT_STRIDE_Q, OUT_STRIDE_X, OUT_STRIDE_Y),
        device=bmm_22.device,
        dtype=torch.float32,
    )

    zero_grid = (triton.cdiv(OUT_STORAGE, 1024),)
    _zero_kernel[zero_grid](output, total=OUT_STORAGE, BLOCK=1024)

    _longformer_softmax_dropout_kernel[(SOFTMAX_ROWS,)](
        bmm_22,
        slice_4,
        slice_3,
        full,
        convert_element_type,
        permute_12,
        convert_element_type_1,
        permute_25,
        unsqueeze_11,
        full_2,
        inductor_seeds,
        output,
        BLOCK=BLOCK_W,
    )
    return output


def _normalize_outputs(outputs):
    if isinstance(outputs, torch.Tensor):
        return (outputs,)
    if isinstance(outputs, tuple):
        return outputs
    if isinstance(outputs, list):
        return tuple(outputs)
    return (outputs,)


def _tensor_metadata_matches(expected: torch.Tensor, actual: torch.Tensor) -> bool:
    return (
        expected.shape == actual.shape
        and expected.dtype == actual.dtype
        and expected.stride() == actual.stride()
        and expected.device == actual.device
    )


def _check_oracle(instance, inputs, *, atol: float, rtol: float, skip_stochastic: bool) -> bool:
    with torch.no_grad():
        eager = _normalize_outputs(instance(*inputs))
        oracle_out = _normalize_outputs(oracle_forward(inputs))
        torch.cuda.synchronize()

    if len(oracle_out) != len(eager):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(oracle_out)} outputs, "
            f"eager produces {len(eager)}"
        )
        return False

    all_pass = True
    for index, (expected, actual) in enumerate(zip(eager, oracle_out)):
        if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
            ok = expected == actual
            print(f"  output {index}: {'PASS' if ok else 'FAIL'} (non-tensor)")
            all_pass = all_pass and bool(ok)
            continue

        metadata = (
            f"shape={list(expected.shape)} dtype={expected.dtype} "
            f"stride={list(expected.stride())}"
        )
        if not _tensor_metadata_matches(expected, actual):
            print(
                f"  output {index}: SCOPE_MISMATCH metadata "
                f"oracle_shape={list(actual.shape)} eager_shape={list(expected.shape)} "
                f"oracle_dtype={actual.dtype} eager_dtype={expected.dtype} "
                f"oracle_stride={list(actual.stride())} eager_stride={list(expected.stride())} "
                f"oracle_device={actual.device} eager_device={expected.device}"
            )
            all_pass = False
            continue

        if skip_stochastic and index in STOCHASTIC_OUTPUTS:
            print(f"  output {index}: SKIP values (stochastic dropout; metadata PASS {metadata})")
            continue

        if not expected.is_floating_point():
            ok = torch.equal(expected, actual)
            print(f"  output {index}: {'PASS' if ok else 'FAIL'} (exact, {metadata})")
            all_pass = all_pass and bool(ok)
            continue

        expected_f32 = expected.float()
        actual_f32 = actual.float()
        max_diff = (expected_f32 - actual_f32).abs().max().item()
        ok = torch.allclose(expected_f32, actual_f32, atol=atol, rtol=rtol)
        print(
            f"  output {index}: {'PASS' if ok else 'FAIL'} "
            f"({metadata} max_diff={max_diff:.2e})"
        )
        all_pass = all_pass and bool(ok)

    return all_pass


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
                        help="Disable skipping the stochastic dropout output")
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
        if args.no_skip_stochastic:
            print(
                f"NOTE: {REPRO_ID} contains stochastic ops; --no-skip-stochastic "
                "requested, so dropout-dependent values will be compared"
            )
        else:
            print(
                f"NOTE: {REPRO_ID} contains stochastic ops; output 0 metadata is "
                "checked and output 0 values are skipped"
            )

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = _check_oracle(
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
