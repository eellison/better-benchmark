"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete two-input ReLU, virtual channel-cat, low-memory 3x3 stride-2 ceil-mode maxpool-with-offsets, and both ReLU backward masks without materializing either ReLU activation or the concatenated f32[512,256,27,27] tensor, whereas Inductor currently schedules the ReLU/cat/stencil/mask graph as generic producer-consumer work with avoidable activation and cat traffic; Inductor cannot do this today because its scheduler does not treat fixed channel concatenation as a virtual multi-source layout that can feed a stencil reduction while also preserving sibling mask consumers; the fix is SCHEDULER_FUSION: teach Inductor to inline static cat producers into low-memory maxpool stencils and route shared ReLU predicates to sibling outputs."""
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
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None

from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_shape_key,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"
CLASSIFICATION = "SCHEDULER_FUSION"

N_BATCH = 512
IN_CHANNELS = 128
OUT_CHANNELS = 256
H_IN = 27
W_IN = 27
H_OUT = 13
W_OUT = 13
INPUT_HW = H_IN * W_IN
OUTPUT_HW = H_OUT * W_OUT
INPUT_NUMEL = N_BATCH * IN_CHANNELS * INPUT_HW
OUTPUT_NUMEL = N_BATCH * OUT_CHANNELS * OUTPUT_HW
INPUT_SHAPE = (N_BATCH, IN_CHANNELS, H_IN, W_IN)
INPUT_STRIDE = (IN_CHANNELS * INPUT_HW, INPUT_HW, W_IN, 1)
OUTPUT_SHAPE = (N_BATCH, OUT_CHANNELS, H_OUT, W_OUT)
OUTPUT_STRIDE = (OUT_CHANNELS * OUTPUT_HW, OUTPUT_HW, W_OUT, 1)
MASK_BLOCK = 1024
POOL_BLOCK = 256


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _relu_le_masks_kernel(
        x0_ptr,
        x1_ptr,
        mask_x1_ptr,
        mask_x0_ptr,
        total: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        active = offsets < total

        x0 = tl.load(x0_ptr + offsets, mask=active, other=0.0)
        x1 = tl.load(x1_ptr + offsets, mask=active, other=0.0)
        tl.store(mask_x1_ptr + offsets, x1 <= 0.0, mask=active)
        tl.store(mask_x0_ptr + offsets, x0 <= 0.0, mask=active)

    @triton.jit
    def _virtual_cat_relu_maxpool_kernel(
        x0_ptr,
        x1_ptr,
        values_ptr,
        offsets_ptr,
        total: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        active = offsets < total

        out_w = offsets % 13
        tmp = offsets // 13
        out_h = tmp % 13
        tmp = tmp // 13
        cat_channel = tmp % 256
        batch = tmp // 256

        use_x1 = cat_channel >= 128
        source_channel = cat_channel % 128
        source_base = (batch * 128 + source_channel) * 729

        best = tl.full([block_size], -float("inf"), dtype=tl.float32)
        best_offset = tl.zeros([block_size], dtype=tl.int32)

        for kh in tl.static_range(0, 3):
            in_h = out_h * 2 + kh
            for kw in tl.static_range(0, 3):
                in_w = out_w * 2 + kw
                input_offset = source_base + in_h * 27 + in_w
                x0 = tl.load(
                    x0_ptr + input_offset,
                    mask=active & ~use_x1,
                    other=-float("inf"),
                )
                x1 = tl.load(
                    x1_ptr + input_offset,
                    mask=active & use_x1,
                    other=-float("inf"),
                )
                raw = tl.where(use_x1, x1, x0)
                relu = tl.where(raw != raw, raw, tl.maximum(raw, 0.0))
                take = active & ((relu > best) | (relu != relu))
                best = tl.where(take, relu, best)
                best_offset = tl.where(take, kh * 3 + kw, best_offset)

        tl.store(values_ptr + offsets, best, mask=active)
        tl.store(offsets_ptr + offsets, best_offset.to(tl.int8), mask=active)


def _require_f32_tensor(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != INPUT_SHAPE:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {INPUT_SHAPE}")
    if tuple(value.stride()) != INPUT_STRIDE:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {INPUT_STRIDE}")
    if value.dtype is not torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects two inputs, got {len(inputs)}")
    x0 = _require_f32_tensor("convolution_11", inputs[0])
    x1 = _require_f32_tensor("convolution_12", inputs[1])
    if x0.device != x1.device:
        raise ValueError(f"inputs must be on the same device, got {x0.device} and {x1.device}")
    return x0, x1


def _torch_oracle(x0: torch.Tensor, x1: torch.Tensor) -> tuple[torch.Tensor, ...]:
    relu0 = torch.relu(x0)
    relu1 = torch.relu(x1)
    cat = torch.cat([relu0, relu1], dim=1)
    values, offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(
        cat,
        [3, 3],
        [2, 2],
        [0, 0],
        [1, 1],
        True,
    )
    return values, offsets, relu1 <= 0, relu0 <= 0


@oracle_impl(hardware="H100", shapes="(T([512, 128, 27, 27], f32), T([512, 128, 27, 27], f32))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    """Run the full Repro.forward scope.

    The maxpool kernel reads directly from the two input tensors according to
    the virtual cat channel range. The mask kernel emits the two boolean
    `relu(input) <= 0` outputs in the same order as the captured repro.
    """
    x0, x1 = _validate_inputs(inputs)
    if not x0.is_cuda:
        return _torch_oracle(x0, x1)
    if triton is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    values = torch.empty_strided(OUTPUT_SHAPE, OUTPUT_STRIDE, device=x0.device, dtype=torch.float32)
    offsets = torch.empty_strided(OUTPUT_SHAPE, OUTPUT_STRIDE, device=x0.device, dtype=torch.int8)
    mask_x1 = torch.empty_strided(INPUT_SHAPE, INPUT_STRIDE, device=x0.device, dtype=torch.bool)
    mask_x0 = torch.empty_strided(INPUT_SHAPE, INPUT_STRIDE, device=x0.device, dtype=torch.bool)

    _relu_le_masks_kernel[(triton.cdiv(INPUT_NUMEL, MASK_BLOCK),)](
        x0,
        x1,
        mask_x1,
        mask_x0,
        total=INPUT_NUMEL,
        block_size=MASK_BLOCK,
        num_warps=4,
        num_stages=4,
    )
    _virtual_cat_relu_maxpool_kernel[(triton.cdiv(OUTPUT_NUMEL, POOL_BLOCK),)](
        x0,
        x1,
        values,
        offsets,
        total=OUTPUT_NUMEL,
        block_size=POOL_BLOCK,
        num_warps=8,
        num_stages=4,
    )
    return values, offsets, mask_x1, mask_x0


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
