"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GhostNet head pointwise scope in one Triton pass, including f32[512,1280,1,1] ReLU, reshape to f32[512,1280], Inductor random dropout with threshold > 0.2 and scale 1.25, and the bool le(relu <= 0) side output, whereas Inductor currently lowers the same stochastic pointwise graph through its generic random/dropout pointwise schedule with internal seed generation and generic multi-output indexing; Inductor cannot do this today because scheduler/codegen does not specialize this fixed contiguous ReLU/dropout/le sibling-output pattern into a single shape-specific stochastic pointwise template whose RNG offsets and output layouts are known statically; the fix is SCHEDULER_FUSION: add a guarded pointwise scheduler path that recognizes fixed-layout ReLU feeding both dropout and le, fuses the stochastic epilogue and bool side output, and emits the direct flat-index Triton loop. Exact stochastic equality against eager Repro.forward cannot be established because Repro.forward creates the Inductor dropout seed internally, so this oracle is diagnostic/not_true_floor and relies on the harness stochastic-output skip for the dropout tensor."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import torch
import torch._inductor.inductor_prims  # noqa: F401

try:
    import triton
    import triton.language as tl
except ImportError:
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

BATCH = 512
CHANNELS = 1280
N_ELEMENTS = BATCH * CHANNELS
INPUT_SHAPE = (BATCH, CHANNELS, 1, 1)
DROPOUT_SHAPE = (BATCH, CHANNELS)
LE_SHAPE = (BATCH, CHANNELS, 1, 1)
CONTIGUOUS_4D_STRIDE = (CHANNELS, 1, 1, 1)
DROP_THRESHOLD = 0.2
DROP_SCALE = 1.25
BLOCK_SIZE = 1024
NUM_WARPS = 4
CLASSIFICATION = "SCHEDULER_FUSION"
TRUE_FLOOR = False


def get_inputs():
    """Load inputs from the repro's make_inputs."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _as_shape_tuple(shape_param: object) -> tuple[int, ...]:
    if not isinstance(shape_param, Sequence):
        raise TypeError(f"_shape_param_0 must be a sequence, got {type(shape_param)!r}")
    return tuple(int(dim) for dim in shape_param)


def _validate_inputs(convolution_94: torch.Tensor, shape_param: object) -> None:
    if not convolution_94.is_cuda:
        raise RuntimeError("CUDA input is required")
    if convolution_94.dtype is not torch.float32:
        raise TypeError(f"expected fp32 input, got {convolution_94.dtype}")
    if tuple(convolution_94.shape) != INPUT_SHAPE:
        raise ValueError(f"unexpected input shape: {tuple(convolution_94.shape)}")
    if tuple(convolution_94.stride()) != CONTIGUOUS_4D_STRIDE:
        raise ValueError(f"unexpected input stride: {tuple(convolution_94.stride())}")
    if _as_shape_tuple(shape_param) != DROPOUT_SHAPE:
        raise ValueError(
            f"_shape_param_0 mismatch: expected {DROPOUT_SHAPE}, "
            f"got {_as_shape_tuple(shape_param)}"
        )


def _make_outputs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    dropout = torch.empty(DROPOUT_SHAPE, device=device, dtype=torch.float32)
    le = torch.empty(LE_SHAPE, device=device, dtype=torch.bool)
    return dropout, le


if triton is not None:

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


@oracle_impl(hardware="H100", shapes="(T([512, 1280, 1, 1], f32), S([512, 1280]))")
def oracle_forward(inputs):
    """Run the full Repro.forward scope with a fused pointwise Triton kernel."""
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    convolution_94, shape_param = inputs
    _validate_inputs(convolution_94, shape_param)

    dropout, le = _make_outputs(convolution_94.device)
    seeds = torch.ops.prims.inductor_seeds.default(1, convolution_94.device)

    grid = (triton.cdiv(N_ELEMENTS, BLOCK_SIZE),)
    _relu_dropout_le_kernel[grid](
        convolution_94,
        seeds,
        dropout,
        le,
        n_elements=N_ELEMENTS,
        block_size=BLOCK_SIZE,
        drop_threshold=DROP_THRESHOLD,
        drop_scale=DROP_SCALE,
        num_warps=NUM_WARPS,
    )
    return dropout, le


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

    print(
        "NOTE: exact stochastic equality is not established for output 0; "
        "this oracle is diagnostic/not_true_floor."
    )
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
