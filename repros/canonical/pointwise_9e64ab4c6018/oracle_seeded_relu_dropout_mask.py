"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete T5 explicit-seed ReLU/dropout/side-mask scope by erasing the metadata-only `[8192,2048] -> [8,1024,2048] -> [8192,2048]` views, threading seed-index 61 directly into one flat Triton pointwise kernel, and storing both the scaled dropout result and `relu <= 0` boolean mask from the shared ReLU value, whereas Inductor already reaches the same practical one-pass stochastic pointwise envelope for this full scope; Inductor cannot be assigned a material local gap today because the remaining time is dominated by the mandatory f32 input read, backend RNG, f32 output store, bool side-mask store, and launch/replay overhead rather than an avoidable scheduler split or layout materialization; the fix is BANDWIDTH_BOUND: record this repro as at floor/not_true_floor for this oracle attempt and only revisit for broader pointwise RNG, bandwidth, or launch-overhead improvements. Exact stochastic equality is not established when values are skipped, so this is a structural not_true_floor oracle."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401 - registers Inductor RNG prims

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

from oracle_harness import (
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

INPUT_SHAPE = (8192, 2048)
INPUT_STRIDE = (2048, 1)
VIEW_SHAPE = (8, 1024, 2048)
VIEW_STRIDE = (1024 * 2048, 2048, 1)
OUT0_SHAPE = INPUT_SHAPE
OUT0_STRIDE = INPUT_STRIDE
OUT1_SHAPE = VIEW_SHAPE
OUT1_STRIDE = VIEW_STRIDE
NUMEL = 8192 * 2048
SEED_COUNT = 64
SEED_INDEX = 61
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
STOCHASTIC_OUTPUTS = (0,)
EXACT_STOCHASTIC_EQUALITY = False


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
        ],
        key=["numel"],
    )
    @triton.jit
    def _relu_dropout_mask_kernel(
        mm_ptr,
        seeds_ptr,
        out0_ptr,
        out1_ptr,
        numel: tl.constexpr,
        seed_index: tl.constexpr,
        dropout_p: tl.constexpr,
        dropout_scale: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel

        x = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        relu = tl.maximum(x, 0.0)
        seed = tl.load(seeds_ptr + seed_index)
        keep = tl.rand(seed, offsets.to(tl.uint32)) > dropout_p
        dropped = tl.where(keep, relu * dropout_scale, 0.0)
        non_positive = relu <= 0.0

        tl.store(out0_ptr + offsets, dropped, mask=mask)
        tl.store(out1_ptr + offsets, non_positive, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_seeded_relu_dropout_mask.py")
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    mm_94 = _require_tensor("mm_94", inputs[0], INPUT_SHAPE, INPUT_STRIDE, torch.float32)
    seeds = _require_tensor(
        "inductor_seeds",
        inputs[1],
        (SEED_COUNT,),
        (1,),
        torch.int64,
    )
    if _shape_tuple(inputs[2]) != VIEW_SHAPE:
        raise ValueError(f"unexpected _shape_param_0: {inputs[2]!r}")
    if _shape_tuple(inputs[3]) != OUT0_SHAPE:
        raise ValueError(f"unexpected _shape_param_1: {inputs[3]!r}")
    if seeds.device != mm_94.device:
        raise RuntimeError("all tensor inputs must be on the same CUDA device")
    return mm_94, seeds


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full ReLU, explicit-seed dropout, final view, and side-mask scope."""
    mm_94, seeds = _validate_inputs(inputs)
    out0 = torch.empty_strided(
        OUT0_SHAPE,
        OUT0_STRIDE,
        device=mm_94.device,
        dtype=torch.float32,
    )
    out1 = torch.empty_strided(
        OUT1_SHAPE,
        OUT1_STRIDE,
        device=mm_94.device,
        dtype=torch.bool,
    )
    grid = lambda meta: (triton.cdiv(NUMEL, meta["BLOCK_SIZE"]),)
    _relu_dropout_mask_kernel[grid](
        mm_94,
        seeds,
        out0,
        out1,
        numel=NUMEL,
        seed_index=SEED_INDEX,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
    )
    return out0, out1


def _repro_has_stochastic_ops() -> bool:
    return has_stochastic_ops(REPRO_PATH) or "inductor_random" in REPRO_PATH.read_text()


def _normalize_outputs(outputs: Any) -> tuple[Any, ...]:
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


def _check_oracle(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
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
            all_pass = bool(ok) and all_pass
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
            print(f"  output {index}: SKIP values (stochastic RNG; metadata PASS {metadata})")
            continue

        if not expected.is_floating_point():
            ok = torch.equal(expected, actual)
            print(f"  output {index}: {'PASS' if ok else 'FAIL'} (exact, {metadata})")
            all_pass = bool(ok) and all_pass
            continue

        expected_f32 = expected.float()
        actual_f32 = actual.float()
        max_diff = (expected_f32 - actual_f32).abs().max().item()
        ok = torch.allclose(expected_f32, actual_f32, atol=atol, rtol=rtol)
        print(
            f"  output {index}: {'PASS' if ok else 'FAIL'} "
            f"({metadata} max_diff={max_diff:.2e})"
        )
        all_pass = bool(ok) and all_pass

    return all_pass


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
                        help="Disable skipping of RNG-dependent output values")
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

    if _repro_has_stochastic_ops():
        if args.no_skip_stochastic:
            print(
                f"NOTE: {REPRO_ID} contains input-seeded Inductor RNG; "
                "--no-skip-stochastic requested, so RNG-dependent values will be compared"
            )
        else:
            print(
                f"NOTE: {REPRO_ID} contains input-seeded Inductor RNG; output metadata "
                "is checked and RNG-dependent values are skipped"
            )
    if not EXACT_STOCHASTIC_EQUALITY:
        print("NOTE: exact stochastic equality is not established; floor status not_true_floor")

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
