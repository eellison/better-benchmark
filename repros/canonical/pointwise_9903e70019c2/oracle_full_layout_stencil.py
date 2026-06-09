"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle fuses the complete fp16 divide-by-8 producer, Longformer sliding-window as_strided clone, and final contiguous view into one Triton materialization kernel, whereas Inductor treats the pointwise producer and overlapping layout clone as separate scheduling regions; Inductor cannot do this today because clone/as_strided materialization is a scheduler fusion barrier even when the overlapping stencil is affine and the consumer is only a view; the fix is SCHEDULER_FUSION: allow pointwise producers to fuse into affine overlapping layout materialization kernels and store the requested final contiguous layout directly."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

from oracle_harness import (
    oracle_impl,  # noqa: E402
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


SEQ = 4096
HIDDEN = 768
HEADS = 12
HEAD_DIM = 64
WINDOWS = 15
WINDOW_SIZE = 512
WINDOW_STEP = 256
OUT_SHAPE = (HEADS * WINDOWS, WINDOW_SIZE, HEAD_DIM)
OUT_STRIDE = (WINDOW_SIZE * HEAD_DIM, HEAD_DIM, 1)
NUMEL = HEADS * WINDOWS * WINDOW_SIZE * HEAD_DIM


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 512}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_N": 1024}, num_warps=8, num_stages=3),
        ],
        key=["N_ELEMENTS"],
    )
    @triton.jit
    def _layout_stencil_div_kernel(
        input_ptr,
        out_ptr,
        N_ELEMENTS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        head_dim: tl.constexpr,
        window_size: tl.constexpr,
        windows: tl.constexpr,
        window_step: tl.constexpr,
        hidden: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N_ELEMENTS

        dim = offsets % head_dim
        tmp = offsets // head_dim
        pos = tmp % window_size
        out_n = tmp // window_size
        window = out_n % windows
        head = out_n // windows

        source_seq = window * window_step + pos
        source_feature = head * head_dim + dim
        source_offset = source_seq * hidden + source_feature

        values = tl.load(input_ptr + source_offset, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, values * 0.125, mask=mask)


def _expect_inputs(inputs: list[object] | tuple[object, ...]) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_full_layout_stencil.py")
    if len(inputs) != 6:
        raise ValueError(f"expected 6 inputs, got {len(inputs)}")

    addmm_66 = inputs[0]
    if not isinstance(addmm_66, torch.Tensor):
        raise TypeError("expected tensor input addmm_66")
    if addmm_66.device.type != "cuda":
        raise RuntimeError("oracle_full_layout_stencil.py expects CUDA input tensors")
    if addmm_66.dtype != torch.float16:
        raise ValueError(f"expected float16 input, got {addmm_66.dtype}")
    if tuple(addmm_66.shape) != (SEQ, HIDDEN) or tuple(addmm_66.stride()) != (HIDDEN, 1):
        raise ValueError(
            f"unexpected addmm_66 layout: shape={tuple(addmm_66.shape)} "
            f"stride={tuple(addmm_66.stride())}"
        )

    expected_shapes = (
        [4096, 1, 768],
        [4096, 1, 12, 64],
        [12, 4096, 64],
        [12, 8, 512, 64],
        [180, 512, 64],
    )
    for index, expected in enumerate(expected_shapes, start=1):
        if list(inputs[index]) != expected:
            raise ValueError(f"unexpected shape parameter {index}: {inputs[index]!r}")

    return addmm_66


@oracle_impl(hardware="H100", shapes="(T([4096, 768], f16), S([4096, 1, 768]), S([4096, 1, 12, 64]), S([12, 4096, 64]), S([12, 8, 512, 64]), S([180, 512, 64]))")
def oracle_forward(inputs):
    """Run the full Repro.forward scope with the same output shape and stride."""
    addmm_66 = _expect_inputs(inputs)
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=addmm_66.device,
        dtype=torch.float16,
    )

    grid = lambda meta: (triton.cdiv(NUMEL, meta["BLOCK_N"]),)
    _layout_stencil_div_kernel[grid](
        addmm_66,
        out,
        N_ELEMENTS=NUMEL,
        head_dim=HEAD_DIM,
        window_size=WINDOW_SIZE,
        windows=WINDOWS,
        window_step=WINDOW_STEP,
        hidden=HIDDEN,
    )
    return out


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
