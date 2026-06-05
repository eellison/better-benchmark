"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete GhostNet head pointwise region from Repro.forward in one Triton pass, including f32[512,1280,1,1] ReLU, reshape to f32[512,1280], Inductor-style tl.rand(seed, flat_offset) dropout with threshold > 0.2 and scale 1.25, and the bool le(relu <= 0) output, whereas Inductor already lowers this as a simple fused pointwise random/dropout kernel plus the same deterministic mask work; the remaining difference is dominated by reading the input and writing one fp32 plus one bool output, so the fix is BANDWIDTH_BOUND. Because Repro.forward creates its random seed internally, this file validates exact Inductor RNG semantics for a supplied seed and uses a scoped stochastic-domain check for the full repro, so its benchmark is diagnosis-only rather than a true canonical floor against eager RNG semantics."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 512
CHANNELS = 1280
N_ELEMENTS = BATCH * CHANNELS
INPUT_SHAPE = (BATCH, CHANNELS, 1, 1)
DROPOUT_SHAPE = (BATCH, CHANNELS)
LE_SHAPE = (BATCH, CHANNELS, 1, 1)
DROP_THRESHOLD = 0.2
DROP_SCALE = 1.25
CLASSIFICATION = "BANDWIDTH_BOUND"
FULL_SCOPE_EXACT_RNG = False

DEFAULT_BLOCK = 256
DEFAULT_NUM_WARPS = 4

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


@triton.jit
def _relu_dropout_le_kernel(
    input_ptr,
    seeds_ptr,
    dropout_ptr,
    le_ptr,
    n_elements: tl.constexpr,
    block_size: tl.constexpr,
    drop_threshold: tl.constexpr,
    drop_scale: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    relu = tl.where(x < 0.0, 0.0, x)

    seed = tl.load(seeds_ptr)
    random = tl.rand(seed, offsets.to(tl.uint32))
    keep = random > drop_threshold
    dropout = relu * keep.to(tl.float32) * drop_scale

    tl.store(dropout_ptr + offsets, dropout, mask=mask)
    tl.store(le_ptr + offsets, relu <= 0.0, mask=mask)


def _load_repro_module() -> Any:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _as_tuple(value: object) -> tuple[object, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)


def _make_inputs(module: Any, seed: int) -> tuple[Any, ...]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    inputs = module.make_inputs()
    moved = []
    for item in inputs:
        if isinstance(item, torch.Tensor):
            moved.append(item.cuda())
        else:
            moved.append(item)
    return tuple(moved)


def get_inputs() -> tuple[Any, ...]:
    return _make_inputs(_load_repro_module(), seed=1234)


def get_repro_instance() -> torch.nn.Module:
    return _load_repro_module().Repro().eval()


def _validate_shape_param(actual: object) -> None:
    if tuple(int(dim) for dim in actual) != DROPOUT_SHAPE:
        raise ValueError(
            f"_shape_param_0 mismatch: expected {DROPOUT_SHAPE}, got {tuple(actual)}"
        )


def _validate_inputs(convolution_94: torch.Tensor, _shape_param_0: object) -> None:
    if not convolution_94.is_cuda:
        raise RuntimeError("CUDA input is required")
    if convolution_94.dtype != torch.float32:
        raise TypeError(f"expected fp32 input, got {convolution_94.dtype}")
    if tuple(convolution_94.shape) != INPUT_SHAPE:
        raise ValueError(f"unexpected input shape: {tuple(convolution_94.shape)}")
    if tuple(convolution_94.stride()) != (CHANNELS, 1, 1, 1):
        raise ValueError(f"unexpected input stride: {tuple(convolution_94.stride())}")
    _validate_shape_param(_shape_param_0)


def _validate_seeds(seeds: torch.Tensor) -> None:
    if not seeds.is_cuda or seeds.dtype != torch.int64 or tuple(seeds.shape) != (1,):
        raise ValueError(
            "seeds must be a CUDA int64 tensor with shape [1], "
            f"got shape={tuple(seeds.shape)} dtype={seeds.dtype} device={seeds.device}"
        )


def _make_outputs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    dropout = torch.empty(DROPOUT_SHAPE, device=device, dtype=torch.float32)
    le = torch.empty(LE_SHAPE, device=device, dtype=torch.bool)
    return dropout, le


def _launch_oracle(
    convolution_94: torch.Tensor,
    seeds: torch.Tensor,
    dropout: torch.Tensor,
    le: torch.Tensor,
    *,
    block_size: int,
    num_warps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_seeds(seeds)
    if tuple(dropout.shape) != DROPOUT_SHAPE or dropout.dtype != torch.float32:
        raise ValueError("dropout output must be CUDA fp32[512,1280]")
    if tuple(le.shape) != LE_SHAPE or le.dtype != torch.bool:
        raise ValueError("le output must be CUDA bool[512,1280,1,1]")
    if not dropout.is_cuda or not le.is_cuda:
        raise RuntimeError("CUDA outputs are required")

    grid = (triton.cdiv(N_ELEMENTS, block_size),)
    _relu_dropout_le_kernel[grid](
        convolution_94,
        seeds,
        dropout,
        le,
        n_elements=N_ELEMENTS,
        block_size=block_size,
        drop_threshold=DROP_THRESHOLD,
        drop_scale=DROP_SCALE,
        num_warps=num_warps,
    )
    return dropout, le


def _oracle_with_seeds(
    inputs: tuple[Any, ...],
    seeds: torch.Tensor,
    *,
    block_size: int = DEFAULT_BLOCK,
    num_warps: int = DEFAULT_NUM_WARPS,
    outputs: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")
    convolution_94, _shape_param_0 = inputs
    _validate_inputs(convolution_94, _shape_param_0)
    if outputs is None:
        outputs = _make_outputs(convolution_94.device)
    return _launch_oracle(
        convolution_94,
        seeds,
        outputs[0],
        outputs[1],
        block_size=block_size,
        num_warps=num_warps,
    )


def oracle_forward(inputs: tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the complete Repro.forward scope with a fused Triton pointwise kernel."""
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")
    convolution_94, _shape_param_0 = inputs
    _validate_inputs(convolution_94, _shape_param_0)
    seeds = torch.ops.prims.inductor_seeds.default(1, convolution_94.device)
    return _oracle_with_seeds(inputs, seeds)


class _SeededReference(torch.nn.Module):
    def forward(
        self,
        convolution_94: torch.Tensor,
        seeds: torch.Tensor,
        _shape_param_0: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        relu_default = torch.ops.aten.relu.default(convolution_94)
        reshape_default = torch.ops.aten.reshape.default(relu_default, _shape_param_0)
        lookup_seed = torch.ops.prims.inductor_lookup_seed.default(seeds, 0)
        random = torch.ops.prims.inductor_random.default(
            list(DROPOUT_SHAPE),
            lookup_seed,
            "rand",
        )
        keep = torch.ops.aten.gt.Scalar(random, DROP_THRESHOLD)
        dropped = torch.ops.aten.mul.Tensor(keep, reshape_default)
        scaled = torch.ops.aten.mul.Tensor(dropped, DROP_SCALE)
        le = torch.ops.aten.le.Scalar(relu_default, 0)
        return scaled, le


def _compile_seeded_reference(
    convolution_94: torch.Tensor,
    seeds: torch.Tensor,
    shape_param: object,
) -> torch.nn.Module:
    import torch._dynamo

    torch._dynamo.reset()
    model = _SeededReference().cuda().eval()
    compiled = torch.compile(model)
    for _ in range(3):
        compiled(convolution_94, seeds, shape_param)
    torch.cuda.synchronize()
    return compiled


def _compile_repro_with_config(
    module: Any,
    inputs: tuple[Any, ...],
    config: dict[str, object],
    warmup: int,
) -> Callable[..., object]:
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    model = module.Repro().cuda().eval()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        for _ in range(max(1, warmup)):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    abs_diff = (actual_f - expected_f).abs()
    max_abs = abs_diff.max().item()
    denom = expected_f.abs().clamp_min(1e-12)
    max_rel = (abs_diff / denom).max().item()
    return max_abs, max_rel


def _compare_outputs(
    actual: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, ...],
    *,
    rtol: float,
    atol: float,
) -> bool:
    if len(actual) != len(expected):
        print(f"output arity mismatch: oracle={len(actual)} ref={len(expected)}")
        return False

    ok = True
    for idx, (got, ref) in enumerate(zip(actual, expected)):
        metadata_ok = (
            got.shape == ref.shape
            and got.dtype == ref.dtype
            and got.stride() == ref.stride()
        )
        if got.dtype == torch.bool:
            value_ok = torch.equal(got, ref)
            max_abs = 0.0 if value_ok else 1.0
            max_rel = max_abs
        else:
            value_ok = torch.allclose(
                got.float(),
                ref.float(),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )
            max_abs, max_rel = _max_diff(got, ref)
        item_ok = metadata_ok and value_ok
        ok = ok and bool(item_ok)
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} "
            f"stride={list(got.stride())} ref_stride={list(ref.stride())} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"values={value_ok} metadata={metadata_ok}"
        )
    return bool(ok)


def _validate_dropout_domain(
    convolution_94: torch.Tensor,
    dropout: torch.Tensor,
    *,
    keep_tolerance: float,
) -> bool:
    relu = torch.relu(convolution_94).reshape(DROPOUT_SHAPE)
    scaled = relu * DROP_SCALE
    positive = relu > 0
    zero = dropout == 0
    scaled_match = torch.isclose(dropout, scaled, rtol=0.0, atol=0.0)
    valid = torch.where(positive, zero | scaled_match, zero)
    domain_ok = bool(valid.all().item())

    positive_count = int(positive.sum().item())
    kept_count = int((positive & scaled_match).sum().item())
    keep_fraction = kept_count / positive_count if positive_count else 0.0
    keep_ok = abs(keep_fraction - (1.0 - DROP_THRESHOLD)) <= keep_tolerance

    print(
        "stochastic-domain output[0]: "
        f"positive={positive_count} kept={kept_count} "
        f"keep_fraction={keep_fraction:.6f} expected={1.0 - DROP_THRESHOLD:.6f} "
        f"tolerance={keep_tolerance:.6f} values={domain_ok} distribution={keep_ok}"
    )
    return bool(domain_ok and keep_ok)


def run_check(
    block_size: int,
    num_warps: int,
    rtol: float,
    atol: float,
    keep_tolerance: float,
) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    module = _load_repro_module()
    inputs = _make_inputs(module, seed=1234)
    _validate_inputs(*inputs)
    convolution_94, shape_param = inputs

    fixed_seeds = torch.tensor([0x1234_5678_9ABC_DEF], device=convolution_94.device, dtype=torch.int64)
    seeded_ref = _compile_seeded_reference(convolution_94, fixed_seeds, shape_param)

    with torch.no_grad():
        expected_seeded = _as_tuple(seeded_ref(convolution_94, fixed_seeds, shape_param))
        actual_seeded = _as_tuple(
            _oracle_with_seeds(
                inputs,
                fixed_seeds,
                block_size=block_size,
                num_warps=num_warps,
            )
        )
        torch.cuda.synchronize()

    print(
        "exact-seed check: comparing oracle tl.rand(seed, flat_offset) against "
        "a torch.compile reference where the seed tensor is an explicit input"
    )
    exact_ok = _compare_outputs(actual_seeded, expected_seeded, rtol=rtol, atol=atol)

    repro = module.Repro().cuda().eval()
    with torch.no_grad():
        eager = _as_tuple(repro(*inputs))
        full_oracle = _as_tuple(
            _oracle_with_seeds(
                inputs,
                torch.ops.prims.inductor_seeds.default(1, convolution_94.device),
                block_size=block_size,
                num_warps=num_warps,
            )
        )
        torch.cuda.synchronize()

    print(
        "full-repro stochastic check: dropout seed is internal to Repro.forward, "
        "so output[0] is checked by domain/distribution and output[1] is exact"
    )
    metadata_ok = (
        full_oracle[0].shape == eager[0].shape
        and full_oracle[0].dtype == eager[0].dtype
        and full_oracle[0].stride() == eager[0].stride()
    )
    print(
        f"stochastic-domain output[0] metadata: shape={list(full_oracle[0].shape)} "
        f"dtype={full_oracle[0].dtype} stride={list(full_oracle[0].stride())} "
        f"ref_stride={list(eager[0].stride())} metadata={metadata_ok}"
    )
    domain_ok = _validate_dropout_domain(
        convolution_94,
        full_oracle[0],
        keep_tolerance=keep_tolerance,
    )
    le_ok = _compare_outputs((full_oracle[1],), (eager[1],), rtol=rtol, atol=atol)

    ok = bool(exact_ok and metadata_ok and domain_ok and le_ok)
    print(f"full_scope_exact_rng={FULL_SCOPE_EXACT_RNG}")
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def _try_capture_graph(fn: Callable[[], object]) -> torch.cuda.CUDAGraph | None:
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        torch.cuda.synchronize()
        return graph
    except Exception:
        return None


def _bench_graph_or_fn(
    graph: torch.cuda.CUDAGraph | None,
    fn: Callable[[], object],
    *,
    warmup: int,
    rep: int,
) -> float:
    from triton.testing import do_bench

    if graph is not None:
        return do_bench(lambda: graph.replay(), warmup=warmup, rep=rep, return_mode="min") * 1000.0
    return do_bench(fn, warmup=warmup, rep=rep, return_mode="min") * 1000.0


def _bench_oracle(
    inputs: tuple[Any, ...],
    *,
    block_size: int,
    num_warps: int,
    warmup: int,
    rep: int,
) -> float:
    convolution_94, _shape_param_0 = inputs
    outputs = _make_outputs(convolution_94.device)
    holder: list[torch.Tensor | None] = [None]

    def run() -> object:
        seeds = torch.ops.prims.inductor_seeds.default(1, convolution_94.device)
        result = _launch_oracle(
            convolution_94,
            seeds,
            outputs[0],
            outputs[1],
            block_size=block_size,
            num_warps=num_warps,
        )
        holder[0] = result[0]
        return result

    for _ in range(5):
        run()
    torch.cuda.synchronize()
    graph = _try_capture_graph(run)
    return _bench_graph_or_fn(graph, run, warmup=warmup, rep=rep)


def _bench_compiled(
    compiled: Callable[..., object],
    inputs: tuple[Any, ...],
    *,
    warmup: int,
    rep: int,
) -> float:
    holder: list[object | None] = [None]

    def run() -> object:
        holder[0] = compiled(*inputs)
        return holder[0]

    for _ in range(5):
        run()
    torch.cuda.synchronize()
    graph = _try_capture_graph(run)
    return _bench_graph_or_fn(graph, run, warmup=warmup, rep=rep)


def run_bench(
    block_size: int,
    num_warps: int,
    warmup: int,
    rep: int,
    no_compile: bool,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    module = _load_repro_module()
    inputs = _make_inputs(module, seed=4321)
    _validate_inputs(*inputs)
    convolution_94, _shape_param_0 = inputs

    logical_bytes = N_ELEMENTS * (4 + 4 + 1)
    print(
        "oracle shape: "
        f"input=f32{INPUT_SHAPE} stride={(CHANNELS, 1, 1, 1)} "
        f"dropout=f32{DROPOUT_SHAPE} le=bool{LE_SHAPE}"
    )
    print(
        "oracle tiling: "
        f"n={N_ELEMENTS} block_size={block_size} num_warps={num_warps} "
        f"logical_read_write_bytes={logical_bytes / 1e6:.3f} MB"
    )

    with torch.no_grad():
        oracle_us = _bench_oracle(
            inputs,
            block_size=block_size,
            num_warps=num_warps,
            warmup=warmup,
            rep=rep,
        )

    oracle_bw = logical_bytes / (oracle_us * 1e-6) / 1e12
    print(
        "oracle full-scope relu/dropout/le: "
        f"{oracle_us:.3f} us ({oracle_bw:.3f} TB/s logical bytes)"
    )
    print(f"oracle_us={oracle_us:.3f}")

    compile_results: list[dict[str, object]] = []
    if not no_compile:
        print(
            "torch.compile timings cover the same Repro.forward scope: ReLU, "
            "reshape, internal Inductor seed creation, random mask, dropout "
            "scale, and le mask"
        )
        for label, config in COMPILE_CONFIGS:
            try:
                compiled = _compile_repro_with_config(module, inputs, config, warmup)
                us = _bench_compiled(compiled, inputs, warmup=warmup, rep=rep)
                compile_results.append({"label": label, "us": us})
                print(f"torch.compile {label}: {us:.3f} us")
            except Exception as exc:
                compile_results.append({"label": label, "error": str(exc)})
                print(f"torch.compile {label}: FAILED ({exc})")

    successful_compile_us = [
        float(result["us"]) for result in compile_results if "us" in result
    ]
    best_required_compile_us = min(successful_compile_us) if successful_compile_us else None
    beats_required_compile = (
        len(successful_compile_us) == len(COMPILE_CONFIGS)
        and best_required_compile_us is not None
        and oracle_us < best_required_compile_us
    )
    true_floor = False

    if best_required_compile_us is not None:
        print(f"best_required_compile_us={best_required_compile_us:.3f}")
    print(f"beats_required_compile={beats_required_compile}")
    print(f"full_scope_exact_rng={FULL_SCOPE_EXACT_RNG}")
    print(f"true_floor={true_floor}")
    print(
        "diagnosis_only: internal stochastic seed prevents an exact eager "
        "full-Repro RNG equality check"
    )

    result = {
        "repro_id": REPRO_ID,
        "classification": CLASSIFICATION,
        "oracle_us": oracle_us,
        "compile_results": compile_results,
        "best_required_compile_us": best_required_compile_us,
        "beats_required_compile": beats_required_compile,
        "full_scope_exact_rng": FULL_SCOPE_EXACT_RNG,
        "true_floor": true_floor,
        "status": "DIAGNOSIS_ONLY",
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="benchmark warmup iterations")
    parser.add_argument("--rep", type=int, default=50, help="benchmark repetitions")
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK)
    parser.add_argument("--num-warps", type=int, default=DEFAULT_NUM_WARPS)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument(
        "--keep-tolerance",
        type=float,
        default=0.01,
        help="allowed absolute deviation from expected dropout keep fraction",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="skip torch.compile baselines for the requested configs",
    )
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check and not run_check(
        block_size=args.block_size,
        num_warps=args.num_warps,
        rtol=args.rtol,
        atol=args.atol,
        keep_tolerance=args.keep_tolerance,
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
