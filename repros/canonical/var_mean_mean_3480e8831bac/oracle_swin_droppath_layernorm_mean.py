"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Swin drop-path residual add, hidden-size-1024 LayerNorm, live rsqrt/1024 side output, and final 7x7 spatial mean directly from the source tensors, whereas Inductor currently schedules the stochastic drop-path producer, row var_mean normalization epilogue, affine output, and downstream spatial mean as generic regions with avoidable normalized-activation traffic; Inductor cannot do this today because its normalization scheduler does not sink a fixed spatial-mean reduction into the row-normalization plan while also threading an Inductor RNG producer and a reused inverse-standard-deviation side output; the fix is SCHEDULER_FUSION: extend the norm-template scheduler to fuse stochastic residual producers and fixed small spatial-mean consumers into one planned LayerNorm lowering with explicit side-output stores."""
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

BATCH = 128
SPATIAL = 49
HEIGHT = 7
WIDTH = 7
HIDDEN = 1024
ROWS = BATCH * SPATIAL
KEEP_PROB = 0.8999999985098839
DROPPATH_SCALE = 1.0 / KEEP_PROB
EPS = 1.0e-5
SEED_INDEX = 45
BLOCK_H = 1024
SPATIAL_CHUNKS = 7
CHUNK_SPATIAL = 7

ADDMAT_SHAPE = (ROWS, HIDDEN)
VIEW_SHAPE = (BATCH, SPATIAL, HIDDEN)
NORM_SHAPE = (BATCH, HEIGHT, WIDTH, HIDDEN)
OUT_SHAPE = (BATCH, HIDDEN)
OUT_STRIDE = (HIDDEN, 1)
SIDE_SHAPE = (BATCH, HEIGHT, WIDTH, 1)
SIDE_STRIDE = (SPATIAL, WIDTH, 1, 1)


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _swin_spatial_chunk_kernel(
        addmm_ptr,
        seeds_ptr,
        residual_ptr,
        side_ptr,
        partial_ptr,
        hidden: tl.constexpr,
        spatial: tl.constexpr,
        chunk_spatial: tl.constexpr,
        keep_prob: tl.constexpr,
        droppath_scale: tl.constexpr,
        eps: tl.constexpr,
        seed_index: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        batch = tl.program_id(0)
        chunk = tl.program_id(1)
        cols = tl.arange(0, BLOCK)
        mask = cols < hidden

        seed = tl.load(seeds_ptr + seed_index)
        keep = tl.rand(seed, batch.to(tl.uint32)) < keep_prob
        scale = tl.where(keep, droppath_scale, 0.0)
        spatial_acc = tl.zeros((BLOCK,), tl.float32)

        for p_offset in tl.range(0, chunk_spatial):
            p = chunk * chunk_spatial + p_offset
            row = batch * spatial + p
            offsets = row * hidden + cols
            x = (
                tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
                + tl.load(addmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32) * scale
            )
            x = tl.where(mask, x, 0.0)

            mean = tl.sum(x, axis=0) / hidden
            centered = x - mean
            variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / hidden
            inv_std = tl.rsqrt(variance + eps)

            spatial_acc += centered * inv_std
            tl.store(side_ptr + row, inv_std / hidden)

        partial_offsets = (batch * (spatial // chunk_spatial) + chunk) * hidden + cols
        tl.store(partial_ptr + partial_offsets, spatial_acc, mask=mask)


    @triton.jit
    def _swin_finalize_spatial_mean_kernel(
        partial_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        spatial: tl.constexpr,
        spatial_chunks: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        batch = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < hidden

        spatial_acc = tl.zeros((BLOCK,), tl.float32)
        for chunk in tl.range(0, spatial_chunks):
            partial_offsets = (batch * spatial_chunks + chunk) * hidden + cols
            spatial_acc += tl.load(partial_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        pooled = (spatial_acc / spatial) * weight + bias
        tl.store(out_ptr + batch * hidden + cols, pooled, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    stride: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if stride is not None and tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects seven inputs, got {len(inputs)}")

    addmm = _require_tensor("addmm_95", inputs[0], ADDMAT_SHAPE, torch.float32, (HIDDEN, 1))
    seeds = _require_tensor("inductor_seeds_default", inputs[1], (46,), torch.int64, (1,))
    residual = _require_tensor("view_652", inputs[2], VIEW_SHAPE, torch.float32, (SPATIAL * HIDDEN, HIDDEN, 1))
    weight = _require_tensor("primals_362", inputs[3], (HIDDEN,), torch.float32, (1,))
    bias = _require_tensor("primals_363", inputs[4], (HIDDEN,), torch.float32, (1,))

    if _shape_tuple(inputs[5]) != VIEW_SHAPE:
        raise ValueError(f"unexpected first reshape shape parameter: {inputs[5]!r}")
    if _shape_tuple(inputs[6]) != NORM_SHAPE:
        raise ValueError(f"unexpected second reshape shape parameter: {inputs[6]!r}")

    device = addmm.device
    if any(t.device != device for t in (seeds, residual, weight, bias)):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return addmm, seeds, residual, weight, bias


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full Repro.forward scope with a fused Swin tail kernel."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_swin_droppath_layernorm_mean.py")

    addmm, seeds, residual, weight, bias = _validate_inputs(inputs)
    out = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=addmm.device, dtype=torch.float32)
    side = torch.empty_strided(SIDE_SHAPE, SIDE_STRIDE, device=addmm.device, dtype=torch.float32)
    partial = torch.empty_strided(
        (BATCH, SPATIAL_CHUNKS, HIDDEN),
        (SPATIAL_CHUNKS * HIDDEN, HIDDEN, 1),
        device=addmm.device,
        dtype=torch.float32,
    )

    _swin_spatial_chunk_kernel[(BATCH, SPATIAL_CHUNKS)](
        addmm,
        seeds,
        residual,
        side,
        partial,
        hidden=HIDDEN,
        spatial=SPATIAL,
        chunk_spatial=CHUNK_SPATIAL,
        keep_prob=KEEP_PROB,
        droppath_scale=DROPPATH_SCALE,
        eps=EPS,
        seed_index=SEED_INDEX,
        BLOCK=BLOCK_H,
        num_warps=8,
        num_stages=3,
    )
    _swin_finalize_spatial_mean_kernel[(BATCH,)](
        partial,
        weight,
        bias,
        out,
        hidden=HIDDEN,
        spatial=SPATIAL,
        spatial_chunks=SPATIAL_CHUNKS,
        BLOCK=BLOCK_H,
        num_warps=8,
        num_stages=3,
    )
    return out, side


def _flatten_tensor_outputs(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        result: list[torch.Tensor] = []
        for item in value:
            result.extend(_flatten_tensor_outputs(item))
        return result
    return []


def _check_metadata(instance: torch.nn.Module, inputs: list[Any] | tuple[Any, ...]) -> bool:
    with torch.no_grad():
        expected = _flatten_tensor_outputs(instance(*inputs))
        actual = _flatten_tensor_outputs(oracle_forward(inputs))
        torch.cuda.synchronize()

    ok = len(expected) == len(actual)
    if not ok:
        print(f"  metadata: FAIL output_count oracle={len(actual)} eager={len(expected)}")
        return False

    for index, (eager, oracle) in enumerate(zip(expected, actual)):
        same = (
            eager.shape == oracle.shape
            and eager.dtype == oracle.dtype
            and eager.stride() == oracle.stride()
        )
        print(
            f"  output {index} metadata: {'PASS' if same else 'FAIL'} "
            f"(shape={list(oracle.shape)} dtype={oracle.dtype} stride={oracle.stride()})"
        )
        ok = ok and same
    return ok


def _has_inductor_random() -> bool:
    return "inductor_random" in REPRO_PATH.read_text()


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
        help="Disable auto-detection and skipping of stochastic outputs",
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

    if has_stochastic_ops(REPRO_PATH) or _has_inductor_random():
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")
        print("NOTE: exact eager stochastic equality is not established; true_floor=no")

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
        if not args.no_skip_stochastic:
            ok = _check_metadata(instance, inputs) and ok
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
