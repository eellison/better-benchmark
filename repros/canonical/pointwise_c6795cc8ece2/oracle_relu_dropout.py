"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete VGG16 train ReLU/dropout return in one Triton pointwise kernel by generating the dropout mask inline, scaling kept ReLU activations, and deriving the `le(relu, 0)` output directly from the input sign, whereas Inductor currently emits one fused pointwise kernel but still leaves the ReLU-threshold predicate in its unsimplified form inside the stochastic epilogue; Inductor cannot do this today because its algebraic simplifier does not canonicalize `le(relu(x), 0)` to `le(x, 0)` across a shared stochastic pointwise producer; the fix is ALGEBRAIC_ELIMINATION: add a ReLU-threshold simplification for zero-threshold comparisons before pointwise RNG codegen."""
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
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"
REPRO_ID = REPRO_DIR.name
SHAPE = (128, 4096)
N_ELEMENTS = 128 * 4096
DROPOUT_P = 0.5
DROPOUT_SCALE = 2.0
CLASSIFICATION = "ALGEBRAIC_ELIMINATION"
TRUE_FLOOR = False
BLOCK_SIZE = 512


def get_inputs() -> list[object]:
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    return _harness_get_repro_instance(REPRO_DIR)


@triton.jit
def _relu_dropout_kernel(
    addmm_ptr,
    seed_ptr,
    out_ptr,
    le_ptr,
    n_elements: tl.constexpr,
    use_dropout: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(addmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    zero = tl.full([BLOCK], 0.0, tl.float32)
    relu = tl.maximum(x, zero)

    if use_dropout:
        seed = tl.load(seed_ptr)
        random_values = tl.rand(seed, offsets.to(tl.uint32))
        keep = random_values > 0.5
        out = keep.to(tl.float32) * relu * 2.0
    else:
        out = relu * 2.0

    le = x <= 0.0
    tl.store(out_ptr + offsets, out, mask=mask)
    tl.store(le_ptr + offsets, le, mask=mask)


def _validate_inputs(inputs: tuple[object, ...] | list[object]) -> torch.Tensor:
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects one input, got {len(inputs)}")
    addmm = inputs[0]
    if not isinstance(addmm, torch.Tensor):
        raise TypeError(f"expected tensor input, got {type(addmm)!r}")
    if not addmm.is_cuda:
        raise RuntimeError("CUDA input is required for the Triton oracle")
    if addmm.dtype != torch.float32:
        raise TypeError(f"expected float32 input, got {addmm.dtype}")
    if tuple(addmm.shape) != SHAPE:
        raise ValueError(f"expected shape {SHAPE}, got {tuple(addmm.shape)}")
    if tuple(addmm.stride()) != (4096, 1):
        raise ValueError(f"expected contiguous stride (4096, 1), got {tuple(addmm.stride())}")
    return addmm


def _make_seed(device: torch.device) -> torch.Tensor:
    seed = torch.empty((1,), device=device, dtype=torch.int64)
    torch.ops.aten.randint.low_out(
        -9223372036854775808,
        9223372036854775807,
        [1],
        out=seed,
    )
    return seed


def _launch(
    addmm: torch.Tensor,
    out: torch.Tensor,
    le: torch.Tensor,
    *,
    seed: torch.Tensor,
    use_dropout: bool,
    block: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    grid = (triton.cdiv(N_ELEMENTS, block),)
    _relu_dropout_kernel[grid](
        addmm,
        seed,
        out,
        le,
        n_elements=N_ELEMENTS,
        use_dropout=use_dropout,
        BLOCK=block,
        num_warps=4,
    )
    return out, le


def oracle_forward(inputs: tuple[object, ...] | list[object]) -> tuple[torch.Tensor, torch.Tensor]:
    addmm = _validate_inputs(inputs)
    out = torch.empty_like(addmm)
    le = torch.empty(SHAPE, device=addmm.device, dtype=torch.bool)
    seed = _make_seed(addmm.device)
    return _launch(addmm, out, le, seed=seed, use_dropout=True, block=BLOCK_SIZE)


def _neutral_forward(inputs: tuple[object, ...] | list[object]) -> tuple[torch.Tensor, torch.Tensor]:
    addmm = _validate_inputs(inputs)
    out = torch.empty_like(addmm)
    le = torch.empty(SHAPE, device=addmm.device, dtype=torch.bool)
    seed = torch.empty((1,), device=addmm.device, dtype=torch.int64)
    return _launch(addmm, out, le, seed=seed, use_dropout=False, block=BLOCK_SIZE)


def _reference_relu_scaled(addmm: torch.Tensor) -> torch.Tensor:
    return torch.relu(addmm) * DROPOUT_SCALE


def _reference_le(addmm: torch.Tensor) -> torch.Tensor:
    return torch.relu(addmm) <= 0


def _allclose_mask(actual: torch.Tensor, expected: torch.Tensor, rtol: float, atol: float) -> torch.Tensor:
    return (actual - expected).abs() <= (atol + rtol * expected.abs())


def _compile_model(warmup: int) -> torch.nn.Module:
    import torch._dynamo

    torch._dynamo.reset()
    compiled = torch.compile(get_repro_instance())
    inputs = tuple(get_inputs())
    with torch.no_grad():
        for _ in range(max(1, warmup)):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


@torch.no_grad()
def run_check(rtol: float, atol: float) -> bool:
    print(f"Checking {REPRO_ID}...")
    print(
        "  note: not_true_floor; exact eager prims.inductor_random identity is "
        "not claimed, so stochastic output is checked by domain/statistics and "
        "compiled tl.rand agreement"
    )

    inputs = tuple(get_inputs())
    addmm = _validate_inputs(inputs)
    ok = True

    expected_scaled = _reference_relu_scaled(addmm)
    expected_le = _reference_le(addmm)

    neutral_out, neutral_le = _neutral_forward(inputs)
    torch.cuda.synchronize()
    neutral_close = torch.allclose(neutral_out, expected_scaled, rtol=rtol, atol=atol)
    neutral_max_diff = (neutral_out - expected_scaled).abs().max().item()
    neutral_le_ok = torch.equal(neutral_le, expected_le)
    print(
        f"  neutral relu*2: {'PASS' if neutral_close else 'FAIL'} "
        f"(shape={list(neutral_out.shape)} dtype={neutral_out.dtype} max_diff={neutral_max_diff:.2e})"
    )
    print(
        f"  le(relu, 0) output: {'PASS' if neutral_le_ok else 'FAIL'} "
        f"(shape={list(neutral_le.shape)} dtype={neutral_le.dtype})"
    )
    ok = ok and bool(neutral_close) and bool(neutral_le_ok)

    stochastic_out, stochastic_le = oracle_forward(inputs)
    torch.cuda.synchronize()
    le_ok = torch.equal(stochastic_le, expected_le)
    kept = _allclose_mask(stochastic_out, expected_scaled, rtol=rtol, atol=atol)
    dropped = stochastic_out.abs() <= atol
    positive_base = expected_scaled.abs() > atol
    domain_ok = bool(torch.all((kept | dropped) | ~positive_base).item())
    keep_count = int((kept & positive_base).sum().item())
    positive_count = int(positive_base.sum().item())
    keep_ratio = keep_count / max(1, positive_count)
    stats_ok = 0.48 <= keep_ratio <= 0.52
    print(
        f"  stochastic dropout domain/scale: {'PASS' if domain_ok else 'FAIL'} "
        "(values are 0 or relu(addmm)*2)"
    )
    print(
        f"  stochastic keep ratio: {'PASS' if stats_ok else 'FAIL'} "
        f"(kept={keep_count} positive={positive_count} ratio={keep_ratio:.6f})"
    )
    print(f"  stochastic le output: {'PASS' if le_ok else 'FAIL'}")
    ok = ok and domain_ok and stats_ok and le_ok

    compiled = _compile_model(warmup=1)
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(addmm.device)
    expected_compiled = compiled(*inputs)
    torch.cuda.synchronize()
    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state(cuda_state, addmm.device)
    actual_compiled_rng = oracle_forward(inputs)
    torch.cuda.synchronize()

    compiled_out_ok = torch.allclose(
        actual_compiled_rng[0],
        expected_compiled[0],
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )
    compiled_le_ok = torch.equal(actual_compiled_rng[1], expected_compiled[1])
    compiled_max_diff = (actual_compiled_rng[0] - expected_compiled[0]).abs().max().item()
    print(
        f"  compiled tl.rand output: {'PASS' if compiled_out_ok else 'FAIL'} "
        f"(max_diff={compiled_max_diff:.2e})"
    )
    print(f"  compiled le output: {'PASS' if compiled_le_ok else 'FAIL'}")
    ok = ok and bool(compiled_out_ok) and bool(compiled_le_ok)

    print(
        "  coverage: full Repro.forward return with ReLU, seed-index-0 "
        "stochastic dropout threshold 0.5, scale 2.0, and bool le output"
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

    holder: list[object | None] = [None]
    oracle_us = _bench_callable(lambda: holder.__setitem__(0, oracle_forward(inputs)), warmup, rep)

    compiled = _compile_model(warmup=5)
    compile_us = _bench_callable(lambda: holder.__setitem__(0, compiled(*inputs)), warmup, rep)

    ratio = compile_us / oracle_us if oracle_us > 0 else 0.0
    status = "GOOD" if ratio > 1.05 else "BAD_ORACLE" if ratio < 0.95 else "AT_FLOOR"
    result = {
        "repro_id": REPRO_ID,
        "oracle_us": round(oracle_us, 3),
        "compile_us": round(compile_us, 3),
        "ratio": round(ratio, 3),
        "status": status,
        "classification": CLASSIFICATION,
        "true_floor": bool(TRUE_FLOOR and status == "GOOD"),
        "floor_status": "not_true_floor",
    }
    print(f"  oracle inline tl.rand full return: {oracle_us:.3f} us")
    print(f"  torch.compile current: {compile_us:.3f} us")
    print(json.dumps(result, sort_keys=True))
    return result


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

    inputs = _harness_get_inputs(REPRO_DIR)
    instance = _harness_get_repro_instance(REPRO_DIR)

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
        status = "PASS" if ok else "FAIL"
        print(f"Correctness: {status}")
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
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
