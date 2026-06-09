"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GPT-J training two-branch RoPE graph, including both strided input permutes, arg583/arg584 coefficient expansion semantics, pairwise slice_scatter rotate-half on the first 64 dims, preservation of the remaining 192 dims, and the two final transposed f32[4096,128] view returns in one Triton pointwise materialization, whereas Inductor lowers the decomposed slice/select/slice_scatter/add/view/permute graph through generic pointwise/layout scheduling around duplicated branches; Inductor cannot do this today because scheduler fusion does not recognize the duplicated RoPE rotate-half plus tail-copy assembly as one direct write into the final contiguous backing layouts for both sibling outputs; the fix is SCHEDULER_FUSION: teach Inductor to fuse repeated GPT-J RoPE rotate-half and slice_scatter concat patterns directly into the returned transposed-view backing storage."""
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
    oracle_impl,
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

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_N": 2048}, num_warps=8, num_stages=2),
        ],
        key=["N"],
    )
    @triton.jit
    def _gptj_rope_pair_kernel(
        x0_ptr,
        x1_ptr,
        coeff_rotate_ptr,
        coeff_direct_ptr,
        out0_base_ptr,
        out1_base_ptr,
        N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ROTARY_DIM: tl.constexpr,
        ROW_STRIDE: tl.constexpr,
        COEFF_STRIDE: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N

        row = offsets // ROW_STRIDE
        dim = offsets - row * ROW_STRIDE
        dim = dim % HEAD_DIM
        pair = dim // 2
        rotary = dim < ROTARY_DIM
        odd = (dim % 2) == 1
        rotate_offsets = offsets + tl.where(odd, -1, 1)
        rotate_sign = tl.where(odd, -1.0, 1.0)

        coeff_offsets = row * COEFF_STRIDE + pair
        coeff_rotate = tl.load(
            coeff_rotate_ptr + coeff_offsets,
            mask=mask & rotary,
            other=0.0,
        ).to(tl.float32)
        coeff_direct = tl.load(
            coeff_direct_ptr + coeff_offsets,
            mask=mask & rotary,
            other=0.0,
        ).to(tl.float32)

        x0 = tl.load(x0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x0_rot = tl.load(
            x0_ptr + rotate_offsets,
            mask=mask & rotary,
            other=0.0,
        ).to(tl.float32)
        x1_rot = tl.load(
            x1_ptr + rotate_offsets,
            mask=mask & rotary,
            other=0.0,
        ).to(tl.float32)

        y0_rotary = x0_rot * rotate_sign * coeff_rotate + x0 * coeff_direct
        y1_rotary = x1_rot * rotate_sign * coeff_rotate + x1 * coeff_direct
        y0 = tl.where(rotary, y0_rotary, x0)
        y1 = tl.where(rotary, y1_rotary, x1)

        # Repro returns the getitem_1 branch first, then the getitem branch.
        tl.store(out0_base_ptr + offsets, y1, mask=mask)
        tl.store(out1_base_ptr + offsets, y0, mask=mask)


INPUT_SHAPE = (1, 16, 128, 256)
INPUT_STRIDE = (524288, 256, 4096, 1)
COEFF_SHAPE = (1, 128, 1, 32, 1)
COEFF_STRIDE = (4096, 32, 32, 1, 1)
SHAPE_PARAMS = (
    (1, 128, 1, 32, 2),
    (1, 128, 1, 64),
    (1, 128, 16, 32, 2),
    (1, 128, 1, 32, 2),
    (1, 128, 1, 64),
    (1, 128, 16, 32, 2),
    (1, 128, 4096),
    (1, 128, 4096),
    (128, 4096),
    (128, 4096),
)
BASE_SHAPE = (1, 128, 16, 256)
BASE_STRIDE = (524288, 4096, 256, 1)
OUTPUT_SHAPE = (4096, 128)
OUTPUT_STRIDE = (1, 4096)
SEQ = 128
HEADS = 16
HEAD_DIM = 256
ROTARY_DIM = 64
COEFFS_PER_ROW = 32
NUMEL = SEQ * HEADS * HEAD_DIM


def _shape_tuple(value, name):
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable shape, got {type(value)!r}") from exc


def _validate_tensor(value, shape, stride, dtype, name):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} shape {tuple(value.shape)} does not match expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} stride {tuple(value.stride())} does not match expected {stride}")
    if value.storage_offset() != 0:
        raise ValueError(f"{name} storage_offset {value.storage_offset()} does not match expected 0")
    if value.dtype is not dtype:
        raise TypeError(f"{name} dtype {value.dtype} does not match expected {dtype}")
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    return value


def _validate_inputs(inputs):
    if len(inputs) != 14:
        raise ValueError(f"{REPRO_ID} expects 14 inputs, got {len(inputs)}")

    x0 = _validate_tensor(inputs[0], INPUT_SHAPE, INPUT_STRIDE, torch.float32, "getitem")
    x1 = _validate_tensor(inputs[1], INPUT_SHAPE, INPUT_STRIDE, torch.float32, "getitem_1")
    coeff_rotate = _validate_tensor(
        inputs[2],
        COEFF_SHAPE,
        COEFF_STRIDE,
        torch.float32,
        "arg583_1",
    )
    coeff_direct = _validate_tensor(
        inputs[3],
        COEFF_SHAPE,
        COEFF_STRIDE,
        torch.float32,
        "arg584_1",
    )

    for i, expected in enumerate(SHAPE_PARAMS, start=4):
        actual = _shape_tuple(inputs[i], f"_shape_param_{i - 4}")
        if actual != expected:
            raise ValueError(f"_shape_param_{i - 4} {actual} does not match expected {expected}")

    return x0, x1, coeff_rotate, coeff_direct


@oracle_impl(hardware="H100", shapes="(T([1, 16, 128, 256], f32, stride=(524288, 256, 4096, 1)), T([1, 16, 128, 256], f32, stride=(524288, 256, 4096, 1)), T([1, 128, 1, 32, 1], f32), T([1, 128, 1, 32, 1], f32), S([1, 128, 1, 32, 2]), S([1, 128, 1, 64]), S([1, 128, 16, 32, 2]), S([1, 128, 1, 32, 2]), S([1, 128, 1, 64]), S([1, 128, 16, 32, 2]), S([1, 128, 4096]), S([1, 128, 4096]), S([128, 4096]), S([128, 4096]))")
def oracle_forward(inputs):
    """Run the full GPT-J two-branch RoPE repro scope."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_gptj_rope_pair.py")

    x0, x1, coeff_rotate, coeff_direct = _validate_inputs(inputs)

    out0_base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=x0.device,
        dtype=torch.float32,
    )
    out1_base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=x0.device,
        dtype=torch.float32,
    )

    grid = lambda meta: (triton.cdiv(NUMEL, meta["BLOCK_N"]),)
    _gptj_rope_pair_kernel[grid](
        x0,
        x1,
        coeff_rotate,
        coeff_direct,
        out0_base,
        out1_base,
        N=NUMEL,
        HEAD_DIM=HEAD_DIM,
        ROTARY_DIM=ROTARY_DIM,
        ROW_STRIDE=HEADS * HEAD_DIM,
        COEFF_STRIDE=COEFFS_PER_ROW,
    )

    out0 = out0_base.view(128, 4096).permute(1, 0)
    out1 = out1_base.view(128, 4096).permute(1, 0)
    return (out0, out1)


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
