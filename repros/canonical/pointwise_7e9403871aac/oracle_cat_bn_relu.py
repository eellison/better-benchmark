"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full virtual channel concat of [64,512,7,7], [64,32,7,7], and [64,32,7,7] inputs, applies the fp32 BN-inference affine from fp16 parameters, performs the explicit fp16 cast, and writes the final ReLU output in one Triton pointwise kernel without materializing the concat, whereas Inductor currently schedules the cat/materialization boundary separately from the downstream BN affine/ReLU pointwise work for this captured graph; Inductor cannot do this today because its scheduler does not keep multi-input channel concat as a virtual producer across the channel-dependent fp32 affine consumer and epilogue; the fix is SCHEDULER_FUSION: allow pointwise consumers with per-channel side inputs to inline a concat producer by emitting guarded source selection in the fused loop nest."""
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


N = 64
C0 = 512
C1 = 32
C2 = 32
C_OUT = C0 + C1 + C2
H = 7
W = 7
HW = H * W
EPS = 1.0e-5
BLOCK_C = 8
BLOCK_HW = 64

if triton is not None:

    @triton.jit
    def oracle_kernel(
        x0_ptr,
        x1_ptr,
        x2_ptr,
        mean_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        BLOCK_C_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
        C0_: tl.constexpr,
        C1_: tl.constexpr,
        C2_: tl.constexpr,
        C_OUT_: tl.constexpr,
        HW_: tl.constexpr,
        EPS_: tl.constexpr,
    ):
        batch = tl.program_id(0)
        channels = tl.program_id(1) * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        spatial = tl.arange(0, BLOCK_HW_)
        valid_c = channels < C_OUT_
        valid_hw = spatial < HW_
        valid = valid_c[:, None] & valid_hw[None, :]

        in_x0 = channels < C0_
        in_x1 = (channels >= C0_) & (channels < (C0_ + C1_))
        in_x2 = ~(in_x0 | in_x1)
        ch0 = channels
        ch1 = tl.where(in_x1, channels - C0_, 0)
        ch2 = tl.where(channels >= (C0_ + C1_), channels - C0_ - C1_, 0)

        x0_offsets = (batch * C0_ + ch0[:, None]) * HW_ + spatial[None, :]
        x1_offsets = (batch * C1_ + ch1[:, None]) * HW_ + spatial[None, :]
        x2_offsets = (batch * C2_ + ch2[:, None]) * HW_ + spatial[None, :]

        x0 = tl.load(x0_ptr + x0_offsets, mask=valid & in_x0[:, None], other=0.0).to(tl.float32)
        x1 = tl.load(x1_ptr + x1_offsets, mask=valid & in_x1[:, None], other=0.0).to(tl.float32)
        x2 = tl.load(x2_ptr + x2_offsets, mask=valid & in_x2[:, None], other=0.0).to(tl.float32)
        x = x0 + x1 + x2

        mean = tl.load(mean_ptr + channels, mask=valid_c, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + channels, mask=valid_c, other=1.0).to(tl.float32)
        weight = tl.load(weight_ptr + channels, mask=valid_c, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channels, mask=valid_c, other=0.0).to(tl.float32)

        y = (x - mean[:, None]) * (1.0 / tl.sqrt(var[:, None] + EPS_))
        y = y * weight[:, None] + bias[:, None]
        y_h = y.to(tl.float16)
        zero_h = tl.full((BLOCK_C_, BLOCK_HW_), 0.0, tl.float16)
        relu_h = tl.where(y_h < zero_h, zero_h, y_h)
        out_offsets = (batch * C_OUT_ + channels[:, None]) * HW_ + spatial[None, :]
        tl.store(out_ptr + out_offsets, relu_h, mask=valid)


def _require_f16_tensor(name, value, shape, stride):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float16:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float16")
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(inputs):
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")
    x0, x1, x2, mean, var, weight, bias = inputs
    x0 = _require_f16_tensor("avg_pool2d_2", x0, (N, C0, H, W), (C0 * HW, HW, W, 1))
    x1 = _require_f16_tensor("convolution_89", x1, (N, C1, H, W), (C1 * HW, HW, W, 1))
    x2 = _require_f16_tensor("convolution_91", x2, (N, C2, H, W), (C2 * HW, HW, W, 1))
    mean = _require_f16_tensor("arg461_1", mean, (C_OUT,), (1,))
    var = _require_f16_tensor("arg462_1", var, (C_OUT,), (1,))
    weight = _require_f16_tensor("arg463_1", weight, (C_OUT,), (1,))
    bias = _require_f16_tensor("arg464_1", bias, (C_OUT,), (1,))
    if any(t.device != x0.device for t in (x1, x2, mean, var, weight, bias)):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return x0, x1, x2, mean, var, weight, bias


def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_cat_bn_relu.py")

    x0, x1, x2, mean, var, weight, bias = _validate_inputs(inputs)
    output = torch.empty_strided(
        (N, C_OUT, H, W),
        (C_OUT * HW, HW, W, 1),
        device=x0.device,
        dtype=torch.float16,
    )
    grid = (N, triton.cdiv(C_OUT, BLOCK_C))
    oracle_kernel[grid](
        x0,
        x1,
        x2,
        mean,
        var,
        weight,
        bias,
        output,
        BLOCK_C_=BLOCK_C,
        BLOCK_HW_=BLOCK_HW,
        C0_=C0,
        C1_=C1,
        C2_=C2,
        C_OUT_=C_OUT,
        HW_=HW,
        EPS_=EPS,
        num_warps=4,
    )
    return output


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
