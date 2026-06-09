"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the ReLU over f32[512,1280,1,1], metadata-only view to [512,1280], Inductor-style generated-seed `tl.rand` dropout threshold `> 0.2` with scale `1.25`, and `relu <= 0` bool side output in one dense Triton pointwise kernel, whereas Inductor currently lowers the decomposed relu/view/inductor_random/gt/mul/le graph through its generic stochastic pointwise path with seed generation as a separate runtime concern rather than a canonical dropout-with-side-mask epilogue; Inductor cannot do this today because the scheduler/codegen does not model seeded random generation plus a deterministic sibling boolean store as one reusable multi-output pointwise fusion pattern; the fix is SCHEDULER_FUSION: add a stochastic multi-output pointwise fusion rule that threads the generated Inductor RNG seed through codegen and emits dropout plus side-mask stores from the shared ReLU producer. Exact stochastic equality is not established for the internally generated eager seed, so this oracle is diagnostic/not_true_floor."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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
EXACT_STOCHASTIC_EQUALITY = False


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _relu_dropout_kernel(
        x_ptr,
        seed_ptr,
        out0_ptr,
        out1_ptr,
        total: tl.constexpr,
        dropout_p: tl.constexpr,
        dropout_scale: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < total
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        relu = tl.maximum(x, 0.0)
        seed = tl.load(seed_ptr)
        keep = tl.rand(seed, offsets.to(tl.uint32)) > dropout_p
        dropped = tl.where(keep, relu * dropout_scale, 0.0)
        non_positive = relu <= 0.0
        tl.store(out0_ptr + offsets, dropped, mask=mask)
        tl.store(out1_ptr + offsets, non_positive, mask=mask)


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> torch.Tensor:
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects two inputs, got {len(inputs)}")
    x, shape_param = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected tensor input 0, got {type(x)!r}")
    if tuple(x.shape) != INPUT_SHAPE:
        raise ValueError(f"unexpected input shape {tuple(x.shape)}, expected {INPUT_SHAPE}")
    if tuple(x.stride()) != INPUT_STRIDE:
        raise ValueError(f"unexpected input stride {tuple(x.stride())}, expected {INPUT_STRIDE}")
    if x.dtype != torch.float32:
        raise TypeError(f"unexpected input dtype {x.dtype}, expected torch.float32")
    if tuple(int(dim) for dim in shape_param) != OUT0_SHAPE:
        raise ValueError(f"unexpected view shape {shape_param}, expected {OUT0_SHAPE}")
    return x


@oracle_impl(hardware="H100", shapes="(T([512, 1280, 1, 1], f32), S([512, 1280]))")
def oracle_forward(inputs: tuple[Any, ...] | list[Any]):
    """Run the full ReLU/dropout/side-mask scope."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    x = _validate_inputs(inputs)
    seed = torch.ops.prims.inductor_seeds.default(1, device=x.device)
    out0 = torch.empty_strided(
        OUT0_SHAPE,
        OUT0_STRIDE,
        device=x.device,
        dtype=torch.float32,
    )
    out1 = torch.empty_strided(
        OUT1_SHAPE,
        OUT1_STRIDE,
        device=x.device,
        dtype=torch.bool,
    )
    block_n = 512
    grid = (triton.cdiv(NUMEL, block_n),)
    _relu_dropout_kernel[grid](
        x,
        seed,
        out0,
        out1,
        total=NUMEL,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return out0, out1


def _has_inductor_random() -> bool:
    return "inductor_random" in REPRO_PATH.read_text()


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

    if has_stochastic_ops(REPRO_PATH) or _has_inductor_random():
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")
    if not EXACT_STOCHASTIC_EQUALITY:
        print(
            "NOTE: exact stochastic equality is not established for the internally "
            "generated Inductor seed; true_floor=False"
        )

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
        if not EXACT_STOCHASTIC_EQUALITY:
            print("diagnosis_only: not_true_floor because exact stochastic equality is not established")


if __name__ == "__main__":
    main()
