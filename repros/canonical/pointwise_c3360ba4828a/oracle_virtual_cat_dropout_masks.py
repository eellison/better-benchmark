"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete SqueezeNet two-input ReLU, virtual channel-cat, seed-index-0 Inductor dropout mask and scale, and both ReLU backward-mask sibling outputs in one Triton pointwise kernel without materializing either ReLU activation or the concatenated f32[512,512,13,13] intermediate, whereas Inductor currently lowers the ReLU/cat/inductor_random/dropout/mask tuple through generic stochastic pointwise scheduling with avoidable virtual-layout and sibling-output work; Inductor cannot do this today because the scheduler/codegen does not keep a static channel concat as a virtual multi-source producer while threading generated Inductor RNG and shared ReLU predicates into all sibling stores; the fix is SCHEDULER_FUSION: teach pointwise scheduling to fuse static concat producers, seeded dropout epilogues, and deterministic sibling masks into one multi-output layout-aware kernel. Exact stochastic equality is not established when the dropout output is skipped, so this is not_true_floor."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401 - registers prims RNG ops

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

N_BATCH = 512
IN_CHANNELS = 256
OUT_CHANNELS = 512
H = 13
W = 13
HW = H * W
INPUT_SHAPE = (N_BATCH, IN_CHANNELS, H, W)
INPUT_STRIDE = (IN_CHANNELS * HW, HW, W, 1)
OUTPUT_SHAPE = (N_BATCH, OUT_CHANNELS, H, W)
OUTPUT_STRIDE = (OUT_CHANNELS * HW, HW, W, 1)
INPUT_NUMEL = N_BATCH * IN_CHANNELS * HW
OUTPUT_NUMEL = N_BATCH * OUT_CHANNELS * HW
SEED_COUNT = 1
DROPOUT_P = 0.5
DROPOUT_SCALE = 2.0
BLOCK_N = 1024

if triton is not None:

    @triton.jit
    def _paired_virtual_cat_dropout_masks_kernel(
        x0_ptr,
        x1_ptr,
        seed_ptr,
        dropout_ptr,
        mask_x1_ptr,
        mask_x0_ptr,
        input_total: tl.constexpr,
        input_batch_stride: tl.constexpr,
        output_batch_stride: tl.constexpr,
        channel_cat_offset: tl.constexpr,
        dropout_p: tl.constexpr,
        dropout_scale: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        active = offsets < input_total

        batch = offsets // input_batch_stride
        batch_inner = offsets - batch * input_batch_stride
        out_x0_offsets = batch * output_batch_stride + batch_inner
        out_x1_offsets = out_x0_offsets + channel_cat_offset

        x0 = tl.load(x0_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        x1 = tl.load(x1_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        relu0 = tl.where(x0 <= 0.0, 0.0, x0)
        relu1 = tl.where(x1 <= 0.0, 0.0, x1)

        seed = tl.load(seed_ptr)
        keep0 = tl.rand(seed, out_x0_offsets.to(tl.uint32)) > dropout_p
        keep1 = tl.rand(seed, out_x1_offsets.to(tl.uint32)) > dropout_p
        dropped0 = relu0 * keep0.to(tl.float32) * dropout_scale
        dropped1 = relu1 * keep1.to(tl.float32) * dropout_scale

        tl.store(dropout_ptr + out_x0_offsets, dropped0, mask=active)
        tl.store(dropout_ptr + out_x1_offsets, dropped1, mask=active)
        tl.store(mask_x0_ptr + offsets, x0 <= 0.0, mask=active)
        tl.store(mask_x1_ptr + offsets, x1 <= 0.0, mask=active)


def _require_input(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must be torch.float32, got {value.dtype}")
    if tuple(value.shape) != INPUT_SHAPE:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {INPUT_SHAPE}")
    if tuple(value.stride()) != INPUT_STRIDE:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {INPUT_STRIDE}")
    if not value.is_cuda:
        raise RuntimeError("CUDA inputs are required for this Triton oracle")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects two inputs, got {len(inputs)}")
    x0 = _require_input("convolution_23", inputs[0])
    x1 = _require_input("convolution_24", inputs[1])
    if x0.device != x1.device:
        raise RuntimeError(f"inputs must be on the same CUDA device, got {x0.device} and {x1.device}")
    return x0, x1


def oracle_forward(inputs):
    """Run the full virtual-cat ReLU dropout plus both sibling mask outputs."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    x0, x1 = _validate_inputs(inputs)
    seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device=x0.device)
    dropout = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=x0.device,
        dtype=torch.float32,
    )
    mask_x1 = torch.empty_strided(
        INPUT_SHAPE,
        INPUT_STRIDE,
        device=x0.device,
        dtype=torch.bool,
    )
    mask_x0 = torch.empty_strided(
        INPUT_SHAPE,
        INPUT_STRIDE,
        device=x0.device,
        dtype=torch.bool,
    )

    _paired_virtual_cat_dropout_masks_kernel[(triton.cdiv(INPUT_NUMEL, BLOCK_N),)](
        x0,
        x1,
        seeds,
        dropout,
        mask_x1,
        mask_x0,
        input_total=INPUT_NUMEL,
        input_batch_stride=IN_CHANNELS * HW,
        output_batch_stride=OUT_CHANNELS * HW,
        channel_cat_offset=IN_CHANNELS * HW,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=4,
    )
    return dropout, mask_x1, mask_x0


# --- CLI entry point ---
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

    # Handle --show-hw early
    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    # Default: run both --check and --bench
    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    # Report if stochastic ops detected in source
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
            # The shared harness owns timing so graph capture, GPU locking, and
            # interleaved oracle/compile measurement stay intact.
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
