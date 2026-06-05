"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full captured cat -> BN-inference affine -> fp16 cast -> ReLU scope in one Triton kernel, reading the eight concat inputs as a virtual channel layout and storing only the final contiguous fp16[64,736,7,7] tensor, whereas Inductor currently materializes the fixed channel cat before the downstream normalization pointwise kernel; Inductor cannot do this today because its scheduler does not model aten.cat as a virtual multi-source producer that can be inlined into elementwise consumers with per-channel source selection; the fix is SCHEDULER_FUSION: allow fixed-shape channel concat producers to feed fused pointwise consumers directly instead of forcing a dense concatenated intermediate."""
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
# Do not add oracle-local sys.path or REPO_ROOT import hacks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)

BATCH = 64
C0 = 512
CSEG = 32
NSEG = 7
CHANNELS = C0 + NSEG * CSEG
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
EPS = 1.0e-5
OUT_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
OUT_STRIDE = (CHANNELS * HW, HW, WIDTH, 1)
BLOCK_ROWS = 16
BLOCK_HW = 64


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _cat_bn_relu_kernel(
        in0,
        in1,
        in2,
        in3,
        in4,
        in5,
        in6,
        in7,
        mean_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        BLOCK_ROWS_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
        eps: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_ROWS_ + tl.arange(0, BLOCK_ROWS_)
        hw_offsets = tl.arange(0, BLOCK_HW_)

        n_offsets = row_offsets // 736
        c_offsets = row_offsets - n_offsets * 736
        valid_rows = row_offsets < 47104
        valid_hw = hw_offsets < 49
        valid = valid_rows[:, None] & valid_hw[None, :]

        out_offsets = row_offsets[:, None] * 49 + hw_offsets[None, :]

        c0_offsets = c_offsets
        in0_offsets = (n_offsets[:, None] * 512 + c0_offsets[:, None]) * 49 + hw_offsets[None, :]
        src_ptr = in0 + in0_offsets

        c1_offsets = c_offsets - 512
        small1_offsets = (n_offsets[:, None] * 32 + c1_offsets[:, None]) * 49 + hw_offsets[None, :]
        use1 = (c_offsets >= 512) & (c_offsets < 544)
        src_ptr = tl.where(use1[:, None], in1 + small1_offsets, src_ptr)

        c2_offsets = c_offsets - 544
        small2_offsets = (n_offsets[:, None] * 32 + c2_offsets[:, None]) * 49 + hw_offsets[None, :]
        use2 = (c_offsets >= 544) & (c_offsets < 576)
        src_ptr = tl.where(use2[:, None], in2 + small2_offsets, src_ptr)

        c3_offsets = c_offsets - 576
        small3_offsets = (n_offsets[:, None] * 32 + c3_offsets[:, None]) * 49 + hw_offsets[None, :]
        use3 = (c_offsets >= 576) & (c_offsets < 608)
        src_ptr = tl.where(use3[:, None], in3 + small3_offsets, src_ptr)

        c4_offsets = c_offsets - 608
        small4_offsets = (n_offsets[:, None] * 32 + c4_offsets[:, None]) * 49 + hw_offsets[None, :]
        use4 = (c_offsets >= 608) & (c_offsets < 640)
        src_ptr = tl.where(use4[:, None], in4 + small4_offsets, src_ptr)

        c5_offsets = c_offsets - 640
        small5_offsets = (n_offsets[:, None] * 32 + c5_offsets[:, None]) * 49 + hw_offsets[None, :]
        use5 = (c_offsets >= 640) & (c_offsets < 672)
        src_ptr = tl.where(use5[:, None], in5 + small5_offsets, src_ptr)

        c6_offsets = c_offsets - 672
        small6_offsets = (n_offsets[:, None] * 32 + c6_offsets[:, None]) * 49 + hw_offsets[None, :]
        use6 = (c_offsets >= 672) & (c_offsets < 704)
        src_ptr = tl.where(use6[:, None], in6 + small6_offsets, src_ptr)

        c7_offsets = c_offsets - 704
        small7_offsets = (n_offsets[:, None] * 32 + c7_offsets[:, None]) * 49 + hw_offsets[None, :]
        use7 = c_offsets >= 704
        src_ptr = tl.where(use7[:, None], in7 + small7_offsets, src_ptr)

        x = tl.load(src_ptr, mask=valid, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c_offsets, mask=valid_rows, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + c_offsets, mask=valid_rows, other=1.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=valid_rows, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=valid_rows, other=0.0).to(tl.float32)

        y = (x - mean[:, None]) * (1.0 / tl.sqrt(var[:, None] + eps)) * weight[:, None] + bias[:, None]
        y = tl.where(y != y, y, tl.maximum(y, 0.0))
        tl.store(out_ptr + out_offsets, y, mask=valid)


def _require_tensor(
    name: str,
    value: object,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype is not torch.float16:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float16")
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(inputs: tuple[object, ...] | list[object]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 12:
        raise ValueError(f"{REPRO_ID} expects 12 inputs, got {len(inputs)}")

    in0 = _require_tensor("avg_pool2d_2", inputs[0], (BATCH, C0, HEIGHT, WIDTH), (C0 * HW, HW, WIDTH, 1))
    small_shape = (BATCH, CSEG, HEIGHT, WIDTH)
    small_stride = (CSEG * HW, HW, WIDTH, 1)
    small_inputs = tuple(
        _require_tensor(f"convolution_{89 + i * 2}", inputs[i + 1], small_shape, small_stride)
        for i in range(NSEG)
    )
    params = tuple(
        _require_tensor(name, value, (CHANNELS,), (1,))
        for name, value in zip(("arg511_1", "arg512_1", "arg513_1", "arg514_1"), inputs[8:12])
    )

    device = in0.device
    for tensor in (*small_inputs, *params):
        if tensor.device != device:
            raise ValueError("all tensor inputs must be on the same CUDA device")

    return (in0, *small_inputs, *params)


def oracle_forward(inputs):
    """Run the full Repro.forward computation with virtual cat fusion."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_cat_bn_relu.py")

    (
        in0,
        in1,
        in2,
        in3,
        in4,
        in5,
        in6,
        in7,
        mean,
        var,
        weight,
        bias,
    ) = _validate_inputs(inputs)

    out = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=in0.device, dtype=torch.float16)
    grid = (triton.cdiv(BATCH * CHANNELS, BLOCK_ROWS),)
    _cat_bn_relu_kernel[grid](
        in0,
        in1,
        in2,
        in3,
        in4,
        in5,
        in6,
        in7,
        mean,
        var,
        weight,
        bias,
        out,
        BLOCK_ROWS_=BLOCK_ROWS,
        BLOCK_HW_=BLOCK_HW,
        eps=EPS,
        num_warps=8,
        num_stages=3,
    )
    return out


def _check_oracle_nan_equal(
    instance: torch.nn.Module,
    inputs: list[object],
    *,
    atol: float,
    rtol: float,
) -> bool:
    with torch.no_grad():
        eager = instance(*inputs)
        oracle_out = oracle_forward(inputs)
        torch.cuda.synchronize()

    if not isinstance(eager, torch.Tensor) or not isinstance(oracle_out, torch.Tensor):
        print("  SCOPE_MISMATCH: eager and oracle outputs must both be tensors")
        return False
    if tuple(oracle_out.shape) != OUT_SHAPE or tuple(eager.shape) != OUT_SHAPE:
        print(f"  output 0: SCOPE_MISMATCH shape oracle={list(oracle_out.shape)} eager={list(eager.shape)}")
        return False
    if oracle_out.dtype is not torch.float16 or eager.dtype is not torch.float16:
        print(f"  output 0: SCOPE_MISMATCH dtype oracle={oracle_out.dtype} eager={eager.dtype}")
        return False
    if tuple(oracle_out.stride()) != OUT_STRIDE or tuple(eager.stride()) != OUT_STRIDE:
        print(f"  output 0: SCOPE_MISMATCH stride oracle={oracle_out.stride()} eager={eager.stride()}")
        return False

    eager_f32 = eager.float()
    oracle_f32 = oracle_out.float()
    eager_nan = torch.isnan(eager_f32)
    oracle_nan = torch.isnan(oracle_f32)
    nan_ok = torch.equal(eager_nan, oracle_nan)
    finite = ~eager_nan
    if finite.any():
        max_diff = (eager_f32[finite] - oracle_f32[finite]).abs().max().item()
        values_ok = torch.allclose(eager_f32[finite], oracle_f32[finite], atol=atol, rtol=rtol)
    else:
        max_diff = 0.0
        values_ok = True

    ok = nan_ok and values_ok
    print(
        f"  output 0: {'PASS' if ok else 'FAIL'} "
        f"(shape={list(eager.shape)} dtype={eager.dtype} stride={eager.stride()} "
        f"max_finite_diff={max_diff:.2e} nan_count={int(eager_nan.sum().item())})"
    )
    return ok


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
        ok = _check_oracle_nan_equal(instance, inputs, atol=args.atol, rtol=args.rtol)
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
            # All timing must go through bench_oracle(). Direct do_bench or
            # compiled(*inputs) timing includes dispatch overhead and can invent
            # fake gaps for fast kernels.
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
