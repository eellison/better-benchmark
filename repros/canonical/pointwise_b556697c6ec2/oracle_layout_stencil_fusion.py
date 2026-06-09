"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full Longformer query/key/value bias add plus overlapping as_strided chunk clone by writing the final `[72, 512, 64]` strided output storage directly in one Triton kernel, whereas Inductor must reason through a view/permute/view/as_strided/unsqueeze/permute/clone/view/permute chain and may leave generic layout-indexing overhead around the materializing clone; Inductor cannot do this today because its scheduler does not canonicalize this fixed sliding-window chunk layout as a direct indexed store from the biased `[1024, 2, 12, 64]` logical tensor; the fix is SCHEDULER_FUSION: recognize the Longformer overlapping chunk clone pattern and fuse producer pointwise work into a specialized direct-layout copy kernel."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - keeps py_compile useful.
    triton = None
    tl = None

from oracle_harness import (
    oracle_impl,
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

MM_SHAPE = (2048, 768)
BIAS_SHAPE = (768,)
OUT_SHAPE = (72, 512, 64)
OUT_STRIDE = (32768, 1, 512)
NUM_CHUNKS = 72
BLOCK_R = 16
BLOCK_D = 64


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create Repro() for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _biased_longformer_chunk_kernel(
        mm_ptr,
        bias_ptr,
        out_ptr,
        BLOCK_R_CONST: tl.constexpr,
        BLOCK_D_CONST: tl.constexpr,
    ):
        chunk = tl.program_id(0)
        token_block = tl.program_id(1)
        qkv_head = chunk // 3
        window = chunk - qkv_head * 3
        batch = qkv_head // 12
        head = qkv_head - batch * 12

        chunk_token = token_block * BLOCK_R_CONST + tl.arange(0, BLOCK_R_CONST)
        head_dim = tl.arange(0, BLOCK_D_CONST)
        token = window * 256 + chunk_token
        hidden = head * 64 + head_dim
        mm_offsets = (token[:, None] * 2 + batch) * 768 + hidden[None, :]
        out_offsets = chunk * 32768 + head_dim[None, :] * 512 + chunk_token[:, None]

        values = tl.load(mm_ptr + mm_offsets).to(tl.float32)
        bias = tl.load(bias_ptr + hidden).to(tl.float32)
        tl.store(out_ptr + out_offsets, values + bias)


def _check_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    mm_45, arg182_1, *shape_params = inputs
    if not isinstance(mm_45, torch.Tensor) or not isinstance(arg182_1, torch.Tensor):
        raise TypeError(f"{REPRO_ID} expects tensor data inputs")
    if tuple(mm_45.shape) != MM_SHAPE:
        raise ValueError(f"{REPRO_ID} expects mm_45 shape {MM_SHAPE}, got {tuple(mm_45.shape)}")
    if tuple(arg182_1.shape) != BIAS_SHAPE:
        raise ValueError(f"{REPRO_ID} expects arg182_1 shape {BIAS_SHAPE}, got {tuple(arg182_1.shape)}")
    if mm_45.dtype is not torch.float32 or arg182_1.dtype is not torch.float32:
        raise ValueError(f"{REPRO_ID} expects f32 tensors, got {mm_45.dtype} and {arg182_1.dtype}")
    if not mm_45.is_cuda or not arg182_1.is_cuda:
        raise ValueError(f"{REPRO_ID} expects CUDA tensors")
    if not mm_45.is_contiguous() or not arg182_1.is_contiguous():
        raise ValueError(
            f"{REPRO_ID} expects contiguous tensors, got {mm_45.stride()} and {arg182_1.stride()}"
        )

    expected_shape_params = (
        [1024, 2, 768],
        [1024, 2, 12, 64],
        [24, 1024, 64],
        [24, 2, 512, 64],
        [72, 64, 512],
    )
    for idx, (actual, expected) in enumerate(zip(shape_params, expected_shape_params, strict=True)):
        if list(actual) != expected:
            raise ValueError(f"{REPRO_ID} shape param {idx} expected {expected}, got {list(actual)}")

    return mm_45, arg182_1


@oracle_impl(hardware="H100", shapes="(T([2048, 768], f32), T([768], f32), S([1024, 2, 768]), S([1024, 2, 12, 64]), S([24, 1024, 64]), S([24, 2, 512, 64]), S([72, 64, 512]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward scope with the final output stride."""
    mm_45, arg182_1 = _check_inputs(inputs)

    out = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=mm_45.device, dtype=torch.float32)
    grid = (NUM_CHUNKS, 512 // BLOCK_R)
    _biased_longformer_chunk_kernel[grid](
        mm_45,
        arg182_1,
        out,
        BLOCK_R_CONST=BLOCK_R,
        BLOCK_D_CONST=BLOCK_D,
        num_warps=8,
    )
    return out


def _check_layout(instance: torch.nn.Module, inputs: list[Any] | tuple[Any, ...]) -> bool:
    with torch.no_grad():
        eager_out = instance(*inputs)
        oracle_out = oracle_forward(inputs)
        torch.cuda.synchronize()

    ok = (
        tuple(oracle_out.shape) == tuple(eager_out.shape) == OUT_SHAPE
        and oracle_out.dtype == eager_out.dtype == torch.float32
        and oracle_out.stride() == eager_out.stride() == OUT_STRIDE
        and oracle_out.storage_offset() == eager_out.storage_offset() == 0
    )
    print(
        f"  output 0 layout: {'PASS' if ok else 'FAIL'} "
        f"(shape={list(oracle_out.shape)} stride={oracle_out.stride()} "
        f"dtype={oracle_out.dtype} storage_offset={oracle_out.storage_offset()})"
    )
    return ok


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
        ok = ok and _check_layout(instance, inputs)
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
