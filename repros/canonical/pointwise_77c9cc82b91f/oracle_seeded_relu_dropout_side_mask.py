"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete T5 training ReLU, supplied-seed `rand > 0.1` dropout scale, final `[8192, 3072]` view, and sibling `[8, 1024, 3072]` `relu <= 0` bool mask in one shape-specialized Triton pointwise kernel, whereas Inductor already lowers the same observable work to one fused supplied-seed stochastic pointwise kernel with one activation read, one float store, and one bool store; Inductor cannot materially improve this local graph today because the measured full-scope oracle is at floor and the remaining cost is dominated by required Triton RNG generation plus mandatory memory traffic rather than avoidable launch or layout materialization overhead; the fix is BANDWIDTH_BOUND: record this as an at-floor stochastic pointwise case unless broader RNG codegen, pointwise bandwidth, or launch-overhead work moves both implementations. Exact eager stochastic equality is not established because `prims.inductor_random` advances independently under eager, so the float output is checked structurally while the deterministic bool output is checked exactly."""
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
CLASSIFICATION = "BANDWIDTH_BOUND"

BATCH = 8
SEQ = 1024
HIDDEN = 3072
ROWS = BATCH * SEQ
NUMEL = ROWS * HIDDEN
INPUT_SHAPE = (ROWS, HIDDEN)
INPUT_STRIDE = (HIDDEN, 1)
VIEW3_SHAPE = (BATCH, SEQ, HIDDEN)
VIEW3_STRIDE = (SEQ * HIDDEN, HIDDEN, 1)
OUT0_SHAPE = (ROWS, HIDDEN)
OUT0_STRIDE = (HIDDEN, 1)
SEED_INDEX = 121
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EXACT_STOCHASTIC_EQUALITY = False


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects four inputs, got {len(inputs)}")

    mm_190, seeds, shape0, shape1 = inputs
    if not isinstance(mm_190, torch.Tensor) or not isinstance(seeds, torch.Tensor):
        raise TypeError("the first two repro inputs must be tensors")
    if mm_190.device.type != "cuda" or seeds.device.type != "cuda":
        raise RuntimeError("CUDA tensors are required for this Triton oracle")
    if mm_190.dtype != torch.float32:
        raise TypeError(f"unexpected activation dtype {mm_190.dtype}, expected torch.float32")
    if seeds.dtype != torch.int64:
        raise TypeError(f"unexpected seed dtype {seeds.dtype}, expected torch.int64")
    if tuple(mm_190.shape) != INPUT_SHAPE:
        raise ValueError(f"unexpected activation shape {tuple(mm_190.shape)}, expected {INPUT_SHAPE}")
    if tuple(mm_190.stride()) != INPUT_STRIDE:
        raise ValueError(f"unexpected activation stride {tuple(mm_190.stride())}, expected {INPUT_STRIDE}")
    if seeds.ndim != 1 or seeds.shape[0] <= SEED_INDEX:
        raise ValueError(f"seed tensor must have at least {SEED_INDEX + 1} elements")
    if tuple(seeds.stride()) != (1,):
        raise ValueError(f"unexpected seed stride {tuple(seeds.stride())}, expected (1,)")
    if _shape_tuple(shape0) != VIEW3_SHAPE:
        raise ValueError(f"unexpected first view shape {shape0!r}, expected {VIEW3_SHAPE}")
    if _shape_tuple(shape1) != OUT0_SHAPE:
        raise ValueError(f"unexpected second view shape {shape1!r}, expected {OUT0_SHAPE}")
    return mm_190, seeds


def _source_has_inductor_random() -> bool:
    return "inductor_random" in REPRO_PATH.read_text()


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 1024}, num_warps=8, num_stages=3),
        ],
        key=["total"],
    )
    @triton.jit
    def _seeded_relu_dropout_side_mask_kernel(
        x_ptr,
        seeds_ptr,
        out0_ptr,
        out1_ptr,
        total: tl.constexpr,
        seed_index: tl.constexpr,
        dropout_p: tl.constexpr,
        dropout_scale: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < total

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        relu = tl.maximum(x, 0.0)
        seed = tl.load(seeds_ptr + seed_index)
        random = tl.rand(seed, offsets.to(tl.uint32))
        keep = random > dropout_p
        dropped = keep.to(tl.float32) * relu * dropout_scale
        non_positive = relu <= 0.0

        tl.store(out0_ptr + offsets, dropped, mask=mask)
        tl.store(out1_ptr + offsets, non_positive, mask=mask)


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full Repro.forward scope with exact output count, dtype, shape, and stride."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_seeded_relu_dropout_side_mask.py")

    mm_190, seeds = _validate_inputs(inputs)
    out0 = torch.empty_strided(
        OUT0_SHAPE,
        OUT0_STRIDE,
        device=mm_190.device,
        dtype=torch.float32,
    )
    out1 = torch.empty_strided(
        VIEW3_SHAPE,
        VIEW3_STRIDE,
        device=mm_190.device,
        dtype=torch.bool,
    )

    grid = lambda meta: (triton.cdiv(NUMEL, meta["BLOCK_N"]),)
    _seeded_relu_dropout_side_mask_kernel[grid](
        mm_190,
        seeds,
        out0,
        out1,
        total=NUMEL,
        seed_index=SEED_INDEX,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
    )
    return out0, out1


def _normalize_outputs(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        tensors: list[torch.Tensor] = []
        for item in value:
            tensors.extend(_normalize_outputs(item))
        return tensors
    return []


@torch.no_grad()
def _check_layout_and_deterministic_values(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
) -> bool:
    eager = _normalize_outputs(instance(*inputs))
    oracle = _normalize_outputs(oracle_forward(inputs))
    torch.cuda.synchronize()

    if len(eager) != len(oracle):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(oracle)} outputs, "
            f"eager produces {len(eager)}"
        )
        return False

    ok = True
    for index, (expected, actual) in enumerate(zip(eager, oracle)):
        layout_ok = (
            expected.shape == actual.shape
            and expected.dtype == actual.dtype
            and expected.stride() == actual.stride()
        )
        print(
            f"  output {index} layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(actual.shape)} stride={actual.stride()} dtype={actual.dtype})"
        )
        ok = ok and layout_ok

    bool_ok = bool(torch.equal(eager[1], oracle[1]))
    print(f"  output 1 deterministic values: {'PASS' if bool_ok else 'FAIL'} (exact bool mask)")
    ok = ok and bool_ok

    if not EXACT_STOCHASTIC_EQUALITY:
        finite = bool(torch.isfinite(oracle[0]).all().item())
        active_fraction = float((oracle[0] != 0).float().mean().item())
        print(
            "  output 0 stochastic values: SKIP exact compare "
            f"(finite={finite} active_fraction={active_fraction:.4f})"
        )
        ok = ok and finite and 0.40 < active_fraction < 0.50

    return ok


def main() -> None:
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

    if has_stochastic_ops(REPRO_PATH) or _source_has_inductor_random():
        print(
            f"NOTE: {REPRO_ID} contains prims.inductor_random; "
            "the deterministic bool output is checked exactly and the float "
            "dropout output is auto-skipped by the shared checker"
        )
    if not EXACT_STOCHASTIC_EQUALITY:
        print("NOTE: true_floor=False for exact eager stochastic values; timing remains full-scope")

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
        ok = _check_layout_and_deterministic_values(instance, inputs) and ok
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
        if not EXACT_STOCHASTIC_EQUALITY:
            print("diagnosis_only: not_true_floor because exact eager stochastic equality is not established")


if __name__ == "__main__":
    main()
