"""Oracle for pointwise_6b84ffe7cd3b.

Gap diagnosis (classification: BANDWIDTH_BOUND): this diagnosis-only oracle computes the complete f32[256,1280] hard-swish epilogue and stochastic dropout mask in one Triton pointwise kernel using the same tl.rand(seed, flat_offset) formula that compiled Inductor emits, whereas Inductor already lowers this repro to one fused pointwise Triton kernel plus seed generation; Inductor cannot materially improve this today with scheduler fusion because there is no remaining graph break or unfused producer/consumer work and the cost is the bandwidth/RNG-bound pointwise pass itself; the required change is BANDWIDTH_BOUND: no compiler fusion change is indicated unless a lower-level RNG/codegen improvement reduces the one-kernel pointwise cost. This file is diagnosis-only/not a true correctness floor because eager prims.inductor_random does not expose an exact reproducible equality contract for a hand-written Triton oracle, so --check validates hard-swish under a neutral mask, stochastic mask domain/scale/statistics, and exact agreement with compiled tl.rand semantics without claiming exact eager equality.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl

from oracle_harness import (
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

SHAPE = (256, 1280)
N_ELEMENTS = 256 * 1280
KEEP_PROB = 0.8
DROPOUT_SCALE = 1.0 / KEEP_PROB
CLASSIFICATION = "BANDWIDTH_BOUND"
TRUE_FLOOR = False

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning=True", {"coordinate_descent_tuning": True}),
    (
        "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,"
        "coordinate_descent_tuning=True,benchmark_combo_kernel=True,"
        "triton.multi_kernel=3",
        {
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": True,
            "coordinate_descent_tuning": True,
            "benchmark_combo_kernel": True,
            "triton.multi_kernel": 3,
        },
    ),
]


def get_inputs():
    """Load inputs from the repro's make_inputs."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create Repro() for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


@triton.jit
def _hardswish_dropout_kernel(
    addmm_ptr,
    seed_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    use_dropout: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(addmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    shifted = x + 3.0
    clipped = tl.minimum(tl.maximum(shifted, 0.0), 6.0)
    y = x * clipped * 0.16666666666666666

    if use_dropout:
        seed = tl.load(seed_ptr)
        random_values = tl.rand(seed, offsets.to(tl.uint32))
        keep = random_values < 0.8
        y = y * keep.to(tl.float32) * 1.25

    tl.store(out_ptr + offsets, y, mask=mask)


def _validate_inputs(inputs: tuple[object, ...] | list[object]) -> torch.Tensor:
    if len(inputs) != 1:
        raise ValueError(f"expected one input, got {len(inputs)}")
    addmm = inputs[0]
    if not isinstance(addmm, torch.Tensor):
        raise TypeError(f"expected tensor input, got {type(addmm)!r}")
    if not addmm.is_cuda:
        raise RuntimeError("CUDA input is required for the Triton oracle")
    if addmm.dtype != torch.float32:
        raise TypeError(f"expected float32 input, got {addmm.dtype}")
    if tuple(addmm.shape) != SHAPE:
        raise ValueError(f"expected shape {SHAPE}, got {tuple(addmm.shape)}")
    if tuple(addmm.stride()) != (1280, 1):
        raise ValueError(f"expected contiguous stride (1280, 1), got {tuple(addmm.stride())}")
    return addmm


def _launch(
    addmm: torch.Tensor,
    out: torch.Tensor,
    *,
    seed: torch.Tensor | None,
    use_dropout: bool,
    block: int,
) -> torch.Tensor:
    if seed is None:
        seed = torch.empty((1,), device=addmm.device, dtype=torch.int64)
    grid = (triton.cdiv(N_ELEMENTS, block),)
    _hardswish_dropout_kernel[grid](
        addmm,
        seed,
        out,
        n_elements=N_ELEMENTS,
        use_dropout=use_dropout,
        BLOCK=block,
        num_warps=4,
    )
    return out


def _make_seed(device: torch.device) -> torch.Tensor:
    seed = torch.empty((1,), device=device, dtype=torch.int64)
    torch.ops.aten.randint.low_out(
        -9223372036854775808,
        9223372036854775807,
        [1],
        out=seed,
    )
    return seed


def oracle_forward(inputs):
    """Compute the full Repro.forward output: hard-swish followed by dropout."""
    addmm = _validate_inputs(inputs)
    out = torch.empty_like(addmm)
    seed = _make_seed(addmm.device)
    return _launch(addmm, out, seed=seed, use_dropout=True, block=1024)


def _neutral_forward(inputs) -> torch.Tensor:
    addmm = _validate_inputs(inputs)
    out = torch.empty_like(addmm)
    return _launch(addmm, out, seed=None, use_dropout=False, block=1024)


def _reference_hardswish(addmm: torch.Tensor) -> torch.Tensor:
    return addmm * torch.clamp(addmm + 3.0, min=0.0, max=6.0) * (1.0 / 6.0)


def _allclose_mask(actual: torch.Tensor, expected: torch.Tensor, rtol: float, atol: float) -> torch.Tensor:
    return (actual - expected).abs() <= (atol + rtol * expected.abs())


def _compile_with_config(config: dict[str, object], warmup: int):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    instance = get_repro_instance()
    inputs = get_inputs()
    with inductor_config.patch(config):
        compiled = torch.compile(instance)
        with torch.no_grad():
            for _ in range(max(1, warmup)):
                compiled(*inputs)
            torch.cuda.synchronize()
    return compiled


@torch.no_grad()
def run_check(rtol: float, atol: float) -> bool:
    print(f"Checking {REPRO_ID}...")
    print(
        "  note: diagnosis-only/not true floor; exact eager equality is not "
        "claimed for prims.inductor_random"
    )
    inputs = tuple(get_inputs())
    addmm = _validate_inputs(inputs)
    ok = True

    expected_hardswish = _reference_hardswish(addmm)
    neutral = _neutral_forward(inputs)
    torch.cuda.synchronize()
    neutral_close = torch.allclose(neutral, expected_hardswish, rtol=rtol, atol=atol)
    neutral_max_diff = (neutral - expected_hardswish).abs().max().item()
    print(
        f"  neutral hard-swish: {'PASS' if neutral_close else 'FAIL'} "
        f"(shape={list(neutral.shape)} dtype={neutral.dtype} max_diff={neutral_max_diff:.2e})"
    )
    ok = ok and bool(neutral_close)

    stochastic = oracle_forward(inputs)
    torch.cuda.synchronize()
    scaled = expected_hardswish * DROPOUT_SCALE
    kept = _allclose_mask(stochastic, scaled, rtol=rtol, atol=atol)
    dropped = stochastic.abs() <= atol
    nonzero_base = expected_hardswish.abs() > atol
    domain_ok = bool(torch.all((kept | dropped) | ~nonzero_base).item())
    keep_count = int((kept & nonzero_base).sum().item())
    nonzero_count = int(nonzero_base.sum().item())
    keep_ratio = keep_count / max(1, nonzero_count)
    stats_ok = 0.79 <= keep_ratio <= 0.81
    print(
        f"  dropout mask domain/scale: {'PASS' if domain_ok else 'FAIL'} "
        f"(values are 0 or hard_swish*1.25)"
    )
    print(
        f"  dropout keep ratio: {'PASS' if stats_ok else 'FAIL'} "
        f"(kept={keep_count} nonzero={nonzero_count} ratio={keep_ratio:.6f})"
    )
    ok = ok and domain_ok and stats_ok

    compiled = _compile_with_config({}, warmup=1)
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(addmm.device)
    expected_compiled = compiled(*inputs)
    torch.cuda.synchronize()
    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state(cuda_state, addmm.device)
    actual_compiled_rng = oracle_forward(inputs)
    torch.cuda.synchronize()
    compiled_close = torch.allclose(
        actual_compiled_rng,
        expected_compiled,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )
    compiled_max_diff = (actual_compiled_rng - expected_compiled).abs().max().item()
    print(
        f"  compiled tl.rand semantics: {'PASS' if compiled_close else 'FAIL'} "
        f"(max_diff={compiled_max_diff:.2e})"
    )
    ok = ok and bool(compiled_close)

    print(
        "  coverage: full hard-swish add/clamp_min/clamp_max/mul/div plus "
        "stochastic dropout mask, keep probability 0.8, scale 1.25"
    )
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


@torch.no_grad()
def _bench_callable(fn, warmup: int, rep: int) -> float:
    from triton.testing import do_bench

    fn()
    torch.cuda.synchronize()
    return float(do_bench(fn, warmup=warmup, rep=rep, return_mode="min") * 1000.0)


@torch.no_grad()
def run_bench(warmup: int, rep: int) -> dict[str, object]:
    print(f"Benchmarking {REPRO_ID}...")
    inputs = tuple(get_inputs())
    _validate_inputs(inputs)

    holder: list[torch.Tensor | None] = [None]
    oracle_us = _bench_callable(lambda: holder.__setitem__(0, oracle_forward(inputs)), warmup, rep)

    compile_results: list[dict[str, object]] = []
    for label, config in COMPILE_CONFIGS:
        try:
            compiled = _compile_with_config(config, warmup=5)
            us = _bench_callable(lambda: holder.__setitem__(0, compiled(*inputs)), warmup, rep)
            compile_results.append({"label": label, "us": us})
            print(f"  torch.compile {label}: {us:.3f} us")
        except Exception as exc:
            compile_results.append({"label": label, "error": str(exc)})
            print(f"  torch.compile {label}: FAILED ({exc})")

    successful_compile = [float(result["us"]) for result in compile_results if "us" in result]
    compile_us = min(successful_compile) if successful_compile else float("nan")
    ratio = compile_us / oracle_us if compile_us == compile_us and oracle_us > 0 else 0.0
    status = "GOOD" if ratio > 1.05 else "BAD_ORACLE" if ratio < 0.95 else "AT_FLOOR"
    true_floor = bool(TRUE_FLOOR and status == "GOOD")
    result = {
        "repro_id": REPRO_ID,
        "oracle_us": round(oracle_us, 3),
        "compile_us": round(compile_us, 3) if compile_us == compile_us else None,
        "ratio": round(ratio, 3),
        "status": status,
        "classification": CLASSIFICATION,
        "true_floor": true_floor,
        "diagnosis_only": not true_floor,
        "compile_results": [
            {**item, "us": round(float(item["us"]), 3)} if "us" in item else item
            for item in compile_results
        ],
    }
    print(json.dumps(result, sort_keys=True))
    if not true_floor:
        print("diagnosis_only: not a true floor because exact eager RNG equality is not claimed")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Oracle for {REPRO_ID}")
    parser.add_argument("--check", action="store_true", help="run correctness diagnostics")
    parser.add_argument("--bench", action="store_true", help="benchmark oracle vs required compile configs")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=200)
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check and not run_check(args.rtol, args.atol):
        return 1
    if args.bench:
        run_bench(args.warmup, args.rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
