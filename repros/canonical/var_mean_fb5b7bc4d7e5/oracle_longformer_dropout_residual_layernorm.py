"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Longformer training dropout-residual LayerNorm scope in one shape-specialized Triton row kernel, including the pre-dropout hidden bias add, seed-index-34 p=0.1 dropout, residual add, fp32 hidden-size-768 var_mean(correction=0, keepdim=True), eps=1e-5 affine epilogue, final [2048,768] view output, and rsqrt/768 side output, whereas Inductor currently schedules the stochastic pointwise dropout producer and row normalization/side-output work through separate generic scheduler boundaries; Inductor cannot do this today because the reduction scheduler does not sink an input-dependent prims.inductor_random dropout mask and its scaled producer expression into the fixed-hidden LayerNorm template while preserving the live inverse-std side output; the fix is SCHEDULER_FUSION: teach Inductor's norm scheduler to fuse stochastic dropout producers, residual adds, affine epilogues, and inverse-std side-output stores into one row-normalization lowering."""
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
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None

from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 2
SEQ_LEN = 1024
ROWS = BATCH * SEQ_LEN
HIDDEN = 768
INPUT_SHAPE = (ROWS, HIDDEN)
RESIDUAL_SHAPE = (BATCH, SEQ_LEN, HIDDEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
INVSTD_DIV_SHAPE = (BATCH, SEQ_LEN, 1)
INVSTD_DIV_STRIDE = (SEQ_LEN, 1, 1)
EPS = 1.0e-5
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
SEED_COUNT = 36
SEED_INDEX = 34
BLOCK_H = 1024
STOCHASTIC_OUTPUTS = (0, 1)
CLASSIFICATION = "SCHEDULER_FUSION"


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _repro_has_stochastic_ops() -> bool:
    return has_stochastic_ops(REPRO_PATH) or "inductor_random" in REPRO_PATH.read_text()


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"ROW_BLOCK": 1}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=8, num_stages=3),
            triton.Config({"ROW_BLOCK": 4}, num_warps=8, num_stages=3),
        ],
        key=["hidden"],
    )
    @triton.jit
    def _dropout_residual_layernorm_kernel(
        mm_ptr,
        pre_dropout_bias_ptr,
        seeds_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        invstd_div_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        dropout_p: tl.constexpr,
        dropout_scale: tl.constexpr,
        seed_index: tl.constexpr,
        block_h: tl.constexpr,
        total_rows: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        cols = tl.arange(0, block_h)
        row_mask = rows < total_rows
        col_mask = cols < hidden
        mask = row_mask[:, None] & col_mask[None, :]
        offsets = rows[:, None] * hidden + cols[None, :]

        mm = tl.load(
            mm_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        pre_dropout_bias = tl.load(
            pre_dropout_bias_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        seed = tl.load(seeds_ptr + seed_index)
        random = tl.rand(seed, offsets.to(tl.uint32))
        keep = (random > dropout_p).to(tl.float32)
        dropped = keep * (mm + pre_dropout_bias[None, :]) * dropout_scale
        x = residual + dropped

        x_masked = tl.where(mask, x, 0.0)
        mean_vec = tl.sum(x_masked, axis=1) / hidden
        mean = mean_vec[:, None]
        centered = x - mean
        variance_vec = tl.sum(tl.where(mask, centered * centered, 0.0), axis=1) / hidden
        invstd_vec = tl.rsqrt(variance_vec + eps)
        invstd = invstd_vec[:, None]

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
        y = centered * invstd * weight[None, :] + bias[None, :]

        tl.store(out_ptr + offsets, y, mask=mask)
        tl.store(invstd_div_ptr + rows, invstd_vec / hidden, mask=row_mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    mm_47, pre_dropout_bias, inductor_seeds, residual, weight, bias, shape0, shape1 = inputs
    tensor_inputs = (mm_47, pre_dropout_bias, inductor_seeds, residual, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first six repro inputs must be tensors")
    if not all(value.is_cuda for value in tensor_inputs):
        raise RuntimeError("CUDA tensors are required for the Triton oracle")
    if not all(value.is_contiguous() for value in tensor_inputs):
        strides = [tuple(value.stride()) for value in tensor_inputs]
        raise ValueError(f"all tensor inputs must be contiguous, got strides={strides}")

    expected_shapes = (
        INPUT_SHAPE,
        (HIDDEN,),
        (SEED_COUNT,),
        RESIDUAL_SHAPE,
        (HIDDEN,),
        (HIDDEN,),
    )
    for index, (value, expected_shape) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected_shape}")

    fp32_tensors = (mm_47, pre_dropout_bias, residual, weight, bias)
    if not all(value.dtype == torch.float32 for value in fp32_tensors):
        dtypes = [value.dtype for value in fp32_tensors]
        raise TypeError(f"expected float32 data tensors, got {dtypes}")
    if inductor_seeds.dtype != torch.int64:
        raise TypeError(f"expected int64 inductor_seeds, got {inductor_seeds.dtype}")

    devices = {value.device for value in tensor_inputs}
    if len(devices) != 1:
        raise ValueError(f"all tensor inputs must be on one device, got {devices}")

    if _shape_tuple(shape0) != RESIDUAL_SHAPE:
        raise ValueError(f"unexpected first view shape: {shape0!r}")
    output_shape = _shape_tuple(shape1)
    if output_shape != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output view shape: {shape1!r}")

    return mm_47, pre_dropout_bias, inductor_seeds, residual, weight, bias, output_shape


@oracle_impl(hardware="H100", shapes="(T([2048, 768], f32), T([768], f32), T([36], i64), T([2, 1024, 768], f32), T([768], f32), T([768], f32), S([2, 1024, 768]), S([2048, 768]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the complete Repro.forward dropout-residual LayerNorm scope."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_longformer_dropout_residual_layernorm.py")

    (
        mm_47,
        pre_dropout_bias,
        inductor_seeds,
        residual,
        weight,
        bias,
        output_shape,
    ) = _validate_inputs(inputs)
    out = torch.empty_strided(
        output_shape,
        OUTPUT_STRIDE,
        device=mm_47.device,
        dtype=torch.float32,
    )
    invstd_div = torch.empty_strided(
        INVSTD_DIV_SHAPE,
        INVSTD_DIV_STRIDE,
        device=mm_47.device,
        dtype=torch.float32,
    )

    grid = lambda meta: (triton.cdiv(ROWS, meta["ROW_BLOCK"]),)
    _dropout_residual_layernorm_kernel[grid](
        mm_47,
        pre_dropout_bias,
        inductor_seeds,
        residual,
        weight,
        bias,
        out,
        invstd_div,
        hidden=HIDDEN,
        eps=EPS,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
        seed_index=SEED_INDEX,
        block_h=BLOCK_H,
        total_rows=ROWS,
    )
    return out, invstd_div


def _normalize_outputs(outputs: Any) -> tuple[Any, ...]:
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


def _check_oracle(
    instance: torch.nn.Module,
    inputs: list[Any],
    *,
    atol: float,
    rtol: float,
    skip_stochastic: bool,
) -> bool:
    with torch.no_grad():
        eager = _normalize_outputs(instance(*inputs))
        oracle_out = _normalize_outputs(oracle_forward(inputs))
        if any(isinstance(value, torch.Tensor) and value.is_cuda for value in oracle_out):
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable skipping stochastic dropout-dependent output values",
    )
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        import json

        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    if _repro_has_stochastic_ops():
        if args.no_skip_stochastic:
            print(
                f"NOTE: {REPRO_ID} contains stochastic ops; --no-skip-stochastic "
                "requested, so dropout-dependent values will be compared"
            )
        else:
            skipped = ", ".join(str(index) for index in STOCHASTIC_OUTPUTS)
            print(
                f"NOTE: {REPRO_ID} contains stochastic ops; output values for "
                f"{skipped} are skipped after metadata checks"
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
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(
                        f"WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
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
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
