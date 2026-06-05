"""Full-scope oracle for pointwise_2596a9303ec6.

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the
complete ReLU, view, Inductor `tl.rand(seed, flat_offset)` dropout, 1.25 scale,
and `relu <= 0` return in one Triton pointwise kernel after the required seed
creation, whereas Inductor already emits one fused pointwise kernel for the
same isolated dense read plus two dense writes; Inductor cannot materially do
less work for this capture without changing the stochastic seed/output
contract, so the fix is BANDWIDTH_BOUND: treat this as an at-floor pointwise
materialization. Because Repro.forward creates its random seed internally, this
file validates exact Inductor RNG semantics only for a supplied seed and keeps
the full-repro benchmark diagnosis-only rather than claiming a true canonical
floor.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
    triton = None
    tl = None

from oracle_harness import (
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

M = 512
N = 1280
NUMEL = M * N
INPUT_SHAPE = (M, N, 1, 1)
INPUT_STRIDE = (N, 1, 1, 1)
OUT0_SHAPE = (M, N)
OUT0_STRIDE = (N, 1)
OUT1_SHAPE = (M, N, 1, 1)
OUT1_STRIDE = (N, 1, 1, 1)
DROPOUT_P = 0.2
DROPOUT_SCALE = 1.25
CLASSIFICATION = "BANDWIDTH_BOUND"
EXACT_EAGER_RNG_MATCHED = False
SEED_LOW = -(2**63)
SEED_HIGH = 2**63 - 1

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


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _relu_dropout_kernel(
        input_ptr,
        seed_ptr,
        out0_ptr,
        out1_ptr,
        input_stride0: tl.constexpr,
        input_stride1: tl.constexpr,
        total: tl.constexpr,
        width: tl.constexpr,
        dropout_p: tl.constexpr,
        dropout_scale: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total
        rows = offsets // width
        cols = offsets - rows * width

        values = tl.load(
            input_ptr + rows * input_stride0 + cols * input_stride1,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        relu = tl.maximum(values, 0.0)

        seed = tl.load(seed_ptr)
        random_values = tl.rand(seed, offsets.to(tl.uint32))
        keep = random_values > dropout_p
        dropped = tl.where(keep, relu * dropout_scale, 0.0)
        non_positive = relu <= 0.0

        tl.store(out0_ptr + offsets, dropped, mask=mask)
        tl.store(out1_ptr + offsets, non_positive, mask=mask)


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, object]:
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects two inputs, got {len(inputs)}")
    convolution_94, shape_param = inputs
    if not isinstance(convolution_94, torch.Tensor):
        raise TypeError(f"expected tensor input 0, got {type(convolution_94)!r}")
    if tuple(convolution_94.shape) != INPUT_SHAPE:
        raise ValueError(
            f"unexpected input shape {tuple(convolution_94.shape)}, expected {INPUT_SHAPE}"
        )
    if tuple(convolution_94.stride()) != INPUT_STRIDE:
        raise ValueError(
            f"unexpected input stride {tuple(convolution_94.stride())}, expected {INPUT_STRIDE}"
        )
    if convolution_94.dtype != torch.float32:
        raise TypeError(f"unexpected input dtype {convolution_94.dtype}, expected fp32")
    if not convolution_94.is_cuda:
        raise ValueError("CUDA input is required for the Triton oracle")
    if tuple(int(dim) for dim in shape_param) != OUT0_SHAPE:
        raise ValueError(f"unexpected view shape param {shape_param}, expected {OUT0_SHAPE}")
    return convolution_94, shape_param


def _make_inductor_seed(device: torch.device) -> torch.Tensor:
    seed = torch.empty_strided((1,), (1,), device=device, dtype=torch.int64)
    torch.ops.aten.randint.low_out(SEED_LOW, SEED_HIGH, [1], out=seed)
    return seed


def _make_outputs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    out0 = torch.empty_strided(
        OUT0_SHAPE,
        OUT0_STRIDE,
        device=device,
        dtype=torch.float32,
    )
    out1 = torch.empty_strided(
        OUT1_SHAPE,
        OUT1_STRIDE,
        device=device,
        dtype=torch.bool,
    )
    return out0, out1


def _launch_oracle_with_seed(
    convolution_94: torch.Tensor,
    seed: torch.Tensor,
    out0: torch.Tensor,
    out1: torch.Tensor,
    *,
    block_size: int,
    num_warps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_relu_dropout.py")
    if tuple(seed.shape) != (1,) or seed.dtype != torch.int64 or not seed.is_cuda:
        raise ValueError("seed must be a CUDA int64 tensor with shape [1]")
    if tuple(out0.shape) != OUT0_SHAPE or tuple(out0.stride()) != OUT0_STRIDE:
        raise ValueError(f"out0 must have shape {OUT0_SHAPE} and stride {OUT0_STRIDE}")
    if tuple(out1.shape) != OUT1_SHAPE or tuple(out1.stride()) != OUT1_STRIDE:
        raise ValueError(f"out1 must have shape {OUT1_SHAPE} and stride {OUT1_STRIDE}")
    if out0.dtype != torch.float32 or out1.dtype != torch.bool:
        raise TypeError(f"unexpected output dtypes {out0.dtype}, {out1.dtype}")

    grid = (triton.cdiv(NUMEL, block_size),)
    _relu_dropout_kernel[grid](
        convolution_94,
        seed,
        out0,
        out1,
        input_stride0=convolution_94.stride(0),
        input_stride1=convolution_94.stride(1),
        total=NUMEL,
        width=N,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out0, out1


def oracle_forward_with_seed(
    inputs: tuple[Any, ...] | list[Any],
    seed: torch.Tensor,
    *,
    block_size: int = 512,
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full pointwise computation with a supplied Inductor seed tensor."""
    convolution_94, _shape_param_0 = _validate_inputs(inputs)
    out0, out1 = _make_outputs(convolution_94.device)
    return _launch_oracle_with_seed(
        convolution_94,
        seed,
        out0,
        out1,
        block_size=block_size,
        num_warps=num_warps,
    )


def oracle_forward(
    inputs: tuple[Any, ...] | list[Any],
    *,
    block_size: int = 512,
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the complete Repro.forward scope with Inductor-style seed creation."""
    convolution_94, _shape_param_0 = _validate_inputs(inputs)
    seed = _make_inductor_seed(convolution_94.device)
    out0, out1 = _make_outputs(convolution_94.device)
    return _launch_oracle_with_seed(
        convolution_94,
        seed,
        out0,
        out1,
        block_size=block_size,
        num_warps=num_warps,
    )


class _SeededReference(torch.nn.Module):
    def forward(self, convolution_94: torch.Tensor, seed: torch.Tensor):
        relu_default = torch.ops.aten.relu.default(convolution_94)
        view_default = torch.ops.aten.view.default(relu_default, [M, N])
        lookup_seed = torch.ops.prims.inductor_lookup_seed.default(seed, 0)
        random_values = torch.ops.prims.inductor_random.default([M, N], lookup_seed, "rand")
        keep = torch.ops.aten.gt.Scalar(random_values, DROPOUT_P)
        dropped = torch.ops.aten.mul.Tensor(keep, view_default)
        scaled = torch.ops.aten.mul.Tensor(dropped, DROPOUT_SCALE)
        non_positive = torch.ops.aten.le.Scalar(relu_default, 0)
        return scaled, non_positive


def _as_tuple(value: object) -> tuple[object, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    max_abs = torch.nan_to_num(diff, nan=0.0).max().item()
    rel = diff / expected.float().abs().clamp_min(1e-8)
    max_rel = torch.nan_to_num(rel, nan=0.0).max().item()
    return max_abs, max_rel


def _metadata_ok(got: torch.Tensor, ref: torch.Tensor) -> bool:
    return got.shape == ref.shape and got.dtype == ref.dtype and got.stride() == ref.stride()


def _compile_seeded_reference(
    convolution_94: torch.Tensor,
    seed: torch.Tensor,
) -> torch.nn.Module:
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    model = _SeededReference().cuda()
    with inductor_config.patch({"coordinate_descent_tuning": True}):
        compiled = torch.compile(model)
        compiled(convolution_94, seed)
        torch.cuda.synchronize()
    return compiled


def _check_seeded_compiled_semantics(
    inputs: tuple[Any, ...] | list[Any],
    *,
    block_size: int,
    num_warps: int,
    rtol: float,
    atol: float,
) -> bool:
    convolution_94, _shape_param_0 = _validate_inputs(inputs)
    seed = torch.tensor([0x1234ABCD], device=convolution_94.device, dtype=torch.int64)
    compiled = _compile_seeded_reference(convolution_94, seed)

    with torch.no_grad():
        expected = _as_tuple(compiled(convolution_94, seed))
        actual = _as_tuple(
            oracle_forward_with_seed(
                inputs,
                seed,
                block_size=block_size,
                num_warps=num_warps,
            )
        )
        torch.cuda.synchronize()

    ok = True
    for idx, (got_item, ref_item) in enumerate(zip(actual, expected)):
        if not isinstance(got_item, torch.Tensor) or not isinstance(ref_item, torch.Tensor):
            item_ok = got_item == ref_item
            print(f"  seeded output {idx}: non-tensor equal={item_ok}")
            ok = ok and bool(item_ok)
            continue

        metadata = _metadata_ok(got_item, ref_item)
        if got_item.dtype == torch.bool:
            value_ok = torch.equal(got_item, ref_item)
            max_abs = 0.0 if value_ok else 1.0
            max_rel = max_abs
        else:
            max_abs, max_rel = _max_diff(got_item, ref_item)
            value_ok = torch.allclose(
                got_item.float(),
                ref_item.float(),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )
        item_ok = metadata and value_ok
        ok = ok and bool(item_ok)
        print(
            f"  seeded output {idx}: {'PASS' if item_ok else 'FAIL'} "
            f"(shape={list(got_item.shape)} dtype={got_item.dtype} "
            f"stride={list(got_item.stride())} max_abs={max_abs:.6e} "
            f"max_rel={max_rel:.6e} values={value_ok} metadata={metadata})"
        )
    print("  seeded RNG check: compared against compiled Inductor tl.rand(seed, offset)")
    return bool(ok)


def _check_eager_deterministic_outputs(
    inputs: tuple[Any, ...] | list[Any],
    *,
    block_size: int,
    num_warps: int,
) -> bool:
    instance = get_repro_instance().cuda()
    with torch.no_grad():
        eager = _as_tuple(instance(*inputs))
        actual = _as_tuple(
            oracle_forward(inputs, block_size=block_size, num_warps=num_warps)
        )
        torch.cuda.synchronize()

    if len(actual) != len(eager):
        print(f"  SCOPE_MISMATCH: oracle={len(actual)} eager={len(eager)}")
        return False

    ok = True
    for idx, (got_item, ref_item) in enumerate(zip(actual, eager)):
        if not isinstance(got_item, torch.Tensor) or not isinstance(ref_item, torch.Tensor):
            item_ok = got_item == ref_item
            print(f"  eager output {idx}: non-tensor equal={item_ok}")
            ok = ok and bool(item_ok)
            continue

        metadata = _metadata_ok(got_item, ref_item)
        if idx == 0:
            print(
                f"  eager output {idx}: SKIP stochastic values "
                f"(shape={list(got_item.shape)} dtype={got_item.dtype} "
                f"stride={list(got_item.stride())} metadata={metadata})"
            )
            ok = ok and metadata
            continue

        value_ok = torch.equal(got_item, ref_item)
        item_ok = metadata and value_ok
        ok = ok and bool(item_ok)
        print(
            f"  eager output {idx}: {'PASS' if item_ok else 'FAIL'} "
            f"(exact bool, shape={list(got_item.shape)} dtype={got_item.dtype} "
            f"stride={list(got_item.stride())} metadata={metadata})"
        )
    return bool(ok)


def _check_stochastic_domain(
    inputs: tuple[Any, ...] | list[Any],
    *,
    block_size: int,
    num_warps: int,
) -> bool:
    convolution_94, _shape_param_0 = _validate_inputs(inputs)
    with torch.no_grad():
        out0, out1 = oracle_forward(inputs, block_size=block_size, num_warps=num_warps)
        torch.cuda.synchronize()
        relu = torch.relu(convolution_94).view(OUT0_SHAPE)
        scaled = relu * DROPOUT_SCALE
        expected_non_positive = (relu.view(OUT1_SHAPE) <= 0)

        out0_zero = out0 == 0
        out0_scaled = out0 == scaled
        valid_values = torch.logical_or(out0_zero, out0_scaled).all().item()
        deterministic_zero = (out0[relu <= 0] == 0).all().item()
        le_ok = torch.equal(out1, expected_non_positive)

        positive = relu > 0
        positive_count = int(positive.sum().item())
        kept_count = int(torch.logical_and(positive, out0_scaled).sum().item())
        keep_rate = kept_count / max(positive_count, 1)
        keep_rate_ok = 0.775 <= keep_rate <= 0.825

    ok = bool(valid_values and deterministic_zero and le_ok and keep_rate_ok)
    print(
        f"  stochastic domain: {'PASS' if ok else 'FAIL'} "
        f"(values_zero_or_scaled={valid_values} relu_nonpositive_zero={deterministic_zero} "
        f"le_matches_relu={le_ok} positive_keep_rate={keep_rate:.6f})"
    )
    print(
        "  note: eager prims.inductor_random ignores the seed tensor, so this is "
        "not a bit-exact eager RNG comparison"
    )
    return ok


def run_check(
    *,
    block_size: int,
    num_warps: int,
    rtol: float,
    atol: float,
) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")
    inputs = tuple(get_inputs())
    _validate_inputs(inputs)

    print(f"Checking {REPRO_ID}...")
    eager_ok = _check_eager_deterministic_outputs(
        inputs,
        block_size=block_size,
        num_warps=num_warps,
    )
    seeded_ok = _check_seeded_compiled_semantics(
        inputs,
        block_size=block_size,
        num_warps=num_warps,
        rtol=rtol,
        atol=atol,
    )
    domain_ok = _check_stochastic_domain(
        inputs,
        block_size=block_size,
        num_warps=num_warps,
    )
    ok = eager_ok and seeded_ok and domain_ok
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def _try_capture_graph(fn: object):
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        torch.cuda.synchronize()
        return graph
    except Exception:
        return None


def _time_graph_or_fn(
    graph: torch.cuda.CUDAGraph | None,
    fn: object,
    *,
    warmup: int,
    rep: int,
) -> float:
    from triton.testing import do_bench

    if graph is not None:
        ms = do_bench(lambda: graph.replay(), warmup=warmup, rep=rep, return_mode="min")
    else:
        ms = do_bench(fn, warmup=warmup, rep=rep, return_mode="min")
    return ms * 1000.0


def _bench_cuda_graph(fn: object, *, warmup: int, rep: int) -> float:
    for _ in range(max(1, warmup)):
        fn()
    torch.cuda.synchronize()
    graph = _try_capture_graph(fn)
    return _time_graph_or_fn(graph, fn, warmup=warmup, rep=rep)


def _compile_with_config(
    inputs: tuple[Any, ...],
    config: dict[str, object],
    warmup: int,
) -> object:
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    model = get_repro_instance().cuda()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        for _ in range(max(1, warmup)):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def run_bench(
    *,
    block_size: int,
    num_warps: int,
    warmup: int,
    rep: int,
    no_compile: bool,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    inputs = tuple(get_inputs())
    convolution_94, _shape_param_0 = _validate_inputs(inputs)
    seed = torch.tensor([0x1234ABCD], device=convolution_94.device, dtype=torch.int64)
    seeded_out0, seeded_out1 = _make_outputs(convolution_94.device)

    logical_bytes = NUMEL * (4 + 4 + 1)
    print(f"Benchmarking {REPRO_ID}...")
    print(
        f"oracle shape: input=f32{INPUT_SHAPE} output0=f32{OUT0_SHAPE} "
        f"output1=bool{OUT1_SHAPE}"
    )
    print(
        f"oracle tiling: block_size={block_size} num_warps={num_warps} "
        f"logical_read_write_bytes={logical_bytes / 1e6:.3f} MB"
    )

    holder: list[Any] = [None]
    with torch.no_grad():
        seeded_kernel_us = _bench_cuda_graph(
            lambda: holder.__setitem__(
                0,
                _launch_oracle_with_seed(
                    convolution_94,
                    seed,
                    seeded_out0,
                    seeded_out1,
                    block_size=block_size,
                    num_warps=num_warps,
                ),
            ),
            warmup=warmup,
            rep=rep,
        )
        full_oracle_us = _bench_cuda_graph(
            lambda: holder.__setitem__(
                0,
                oracle_forward(inputs, block_size=block_size, num_warps=num_warps),
            ),
            warmup=warmup,
            rep=rep,
        )

    seeded_bw = logical_bytes / (seeded_kernel_us * 1e-6) / 1e12
    full_bw = logical_bytes / (full_oracle_us * 1e-6) / 1e12
    print(
        f"oracle seeded Triton kernel: {seeded_kernel_us:.3f} us "
        f"({seeded_bw:.3f} TB/s logical bytes)"
    )
    print(
        f"oracle full forward including seed creation: {full_oracle_us:.3f} us "
        f"({full_bw:.3f} TB/s logical bytes)"
    )

    compile_results: list[dict[str, object]] = []
    if not no_compile:
        compiled_holder: list[Any] = [None]
        for label, config in COMPILE_CONFIGS:
            try:
                compiled = _compile_with_config(inputs, config, warmup)
                us = _bench_cuda_graph(
                    lambda: compiled_holder.__setitem__(0, compiled(*inputs)),
                    warmup=warmup,
                    rep=rep,
                )
                compile_results.append({"label": label, "us": us})
                print(f"torch.compile {label}: {us:.3f} us")
            except Exception as exc:
                compile_results.append({"label": label, "error": str(exc)})
                print(f"torch.compile {label}: FAILED ({exc})")

    successful_compile_us = [
        float(result["us"]) for result in compile_results if "us" in result
    ]
    best_compile_us = min(successful_compile_us) if successful_compile_us else None
    beats_required_compile = (
        best_compile_us is not None
        and full_oracle_us < best_compile_us
        and len(successful_compile_us) == len(COMPILE_CONFIGS)
    )
    true_floor = bool(beats_required_compile and EXACT_EAGER_RNG_MATCHED)
    status = "GOOD" if true_floor else "DIAGNOSIS_ONLY"
    if best_compile_us is not None:
        print(f"best_required_compile_us={best_compile_us:.3f}")
    print(f"beats_required_compile={beats_required_compile}")
    print(f"exact_eager_rng_matched={EXACT_EAGER_RNG_MATCHED}")
    print(f"true_floor={true_floor}")
    if not true_floor:
        print(
            "diagnosis_only: stochastic eager RNG is not bit-exactly matched, "
            "so this benchmark does not claim a true canonical floor"
        )

    result = {
        "repro_id": REPRO_ID,
        "classification": CLASSIFICATION,
        "oracle_seeded_kernel_us": seeded_kernel_us,
        "oracle_us": full_oracle_us,
        "compile_results": compile_results,
        "best_required_compile_us": best_compile_us,
        "beats_required_compile": beats_required_compile,
        "exact_eager_rng_matched": EXACT_EAGER_RNG_MATCHED,
        "true_floor": true_floor,
        "status": status,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="benchmark warmup iterations")
    parser.add_argument("--rep", type=int, default=50, help="benchmark repetitions")
    parser.add_argument("--block-size", type=int, default=512, help="Triton elements per program")
    parser.add_argument("--num-warps", type=int, default=4, help="Triton warps per program")
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="skip torch.compile baselines for the requested configs",
    )
    parser.add_argument("--show-hw", action="store_true", help="print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if args.block_size <= 0 or args.block_size & (args.block_size - 1):
        raise ValueError("--block-size must be a positive power of two")

    if not args.check and not args.bench:
        args.check = args.bench = True

    if args.check and not run_check(
        block_size=args.block_size,
        num_warps=args.num_warps,
        rtol=args.rtol,
        atol=args.atol,
    ):
        sys.exit(1)

    if args.bench:
        run_bench(
            block_size=args.block_size,
            num_warps=args.num_warps,
            warmup=args.warmup,
            rep=args.rep,
            no_compile=args.no_compile,
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
