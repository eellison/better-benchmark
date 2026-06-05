"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete GhostNet head pointwise region from Repro.forward in one Triton pass, including f32[512,1280,1,1] ReLU, reshape to f32[512,1280], Inductor-style tl.rand(seed, flat_offset) dropout with threshold > 0.2 and scale 1.25, and the bool le(relu <= 0) output, whereas Inductor already lowers this as a simple fused pointwise random/dropout kernel plus the same deterministic mask work; the remaining difference is dominated by reading the input and writing one fp32 plus one bool output, so the fix is BANDWIDTH_BOUND. Because Repro.forward creates its random seed internally, this file validates exact Inductor RNG semantics for a supplied seed and uses a scoped stochastic-domain check for the full repro, so its benchmark is diagnosis-only rather than a true canonical floor against eager RNG semantics."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

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
