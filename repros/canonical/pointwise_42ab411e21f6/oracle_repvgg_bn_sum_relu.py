"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the full three-branch RepVGG BN-inference affine sum plus ReLU by folding each branch to channel-only scale/shift values, reusing them across spatial tiles, and masking activation loads for channels whose folded coefficients already force NaN output, whereas Inductor currently lowers the decomposed unsqueeze/sub/sqrt/reciprocal/mul/add graph as a generic pointwise fusion that recomputes channel-only normalization work and still reads all activation inputs; Inductor cannot do this today because its algebraic simplifier/codegen does not canonicalize repeated BN-inference branches into hoisted per-channel FMA coefficients with dead activation-load elimination before pointwise scheduling; the fix is ALGEBRAIC_ELIMINATION: add a BN-affine folding pass that produces per-channel scale/shift coefficients for sibling branches, preserves NaN semantics, and emits the final sum/ReLU as tiled FMAs with masked dead loads."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


EPS = 1.0e-5


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
            triton.Config({"BLOCK_C": 1, "BLOCK_HW": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 2, "BLOCK_HW": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 4, "BLOCK_HW": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 8, "BLOCK_HW": 256}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_C": 4, "BLOCK_HW": 128}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 8, "BLOCK_HW": 128}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 16, "BLOCK_HW": 128}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_C": 16, "BLOCK_HW": 64}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_C": 32, "BLOCK_HW": 32}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_C": 32, "BLOCK_HW": 64}, num_warps=8, num_stages=3),
        ],
        key=["C", "H", "W", "S_C", "S_H", "S_W"],
    )
    @triton.jit
    def _repvgg_bn_sum_relu_kernel(
        mean0_ptr,
        x0_ptr,
        var0_ptr,
        weight0_ptr,
        bias0_ptr,
        mean1_ptr,
        x1_ptr,
        var1_ptr,
        weight1_ptr,
        bias1_ptr,
        mean2_ptr,
        x2_ptr,
        var2_ptr,
        weight2_ptr,
        bias2_ptr,
        out_ptr,
        C: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        S_N: tl.constexpr,
        S_C: tl.constexpr,
        S_H: tl.constexpr,
        S_W: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_HW: tl.constexpr,
        eps: tl.constexpr,
    ):
        n = tl.program_id(0)
        c_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        hw_offsets = tl.program_id(2) * BLOCK_HW + tl.arange(0, BLOCK_HW)

        h_offsets = hw_offsets // W
        w_offsets = hw_offsets - h_offsets * W
        hw_mask = hw_offsets < H * W
        c_mask = c_offsets < C
        mask = c_mask[:, None] & hw_mask[None, :]
        offsets = (
            n * S_N
            + c_offsets[:, None] * S_C
            + h_offsets[None, :] * S_H
            + w_offsets[None, :] * S_W
        )

        mean0 = tl.load(mean0_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        var0 = tl.load(var0_ptr + c_offsets, mask=c_mask, other=1.0).to(tl.float32)
        weight0 = tl.load(weight0_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        bias0 = tl.load(bias0_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

        mean1 = tl.load(mean1_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        var1 = tl.load(var1_ptr + c_offsets, mask=c_mask, other=1.0).to(tl.float32)
        weight1 = tl.load(weight1_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        bias1 = tl.load(bias1_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

        mean2 = tl.load(mean2_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        var2 = tl.load(var2_ptr + c_offsets, mask=c_mask, other=1.0).to(tl.float32)
        weight2 = tl.load(weight2_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        bias2 = tl.load(bias2_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

        scale0 = weight0 * tl.rsqrt(var0 + eps)
        scale1 = weight1 * tl.rsqrt(var1 + eps)
        scale2 = weight2 * tl.rsqrt(var2 + eps)
        shift0 = bias0 - mean0 * scale0
        shift1 = bias1 - mean1 * scale1
        shift2 = bias2 - mean2 * scale2
        shift = shift0 + shift1 + shift2
        invalid_channel = (
            (scale0 != scale0)
            | (scale1 != scale1)
            | (scale2 != scale2)
            | (shift != shift)
        )
        load_mask = mask & (~invalid_channel[:, None])
        x0 = tl.load(x0_ptr + offsets, mask=load_mask, other=0.0).to(tl.float32)
        x1 = tl.load(x1_ptr + offsets, mask=load_mask, other=0.0).to(tl.float32)
        x2 = tl.load(x2_ptr + offsets, mask=load_mask, other=0.0).to(tl.float32)

        y = x0 * scale0[:, None] + x1 * scale1[:, None] + x2 * scale2[:, None] + shift[:, None]
        relu = tl.where((y > 0.0) | (y != y), y, 0.0)
        tl.store(out_ptr + offsets, relu, mask=mask)


def _require_f32_tensor(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 15:
        raise ValueError(f"{REPRO_ID} expects 15 inputs, got {len(inputs)}")

    tensors = tuple(
        _require_f32_tensor(name, value)
        for name, value in zip(
            (
                "arg270_1",
                "convolution_40",
                "arg271_1",
                "arg272_1",
                "arg273_1",
                "arg275_1",
                "convolution_41",
                "arg276_1",
                "arg277_1",
                "arg278_1",
                "arg265_1",
                "relu_19",
                "arg266_1",
                "arg267_1",
                "arg268_1",
            ),
            inputs,
        )
    )
    (
        mean0,
        x0,
        var0,
        weight0,
        bias0,
        mean1,
        x1,
        var1,
        weight1,
        bias1,
        mean2,
        x2,
        var2,
        weight2,
        bias2,
    ) = tensors

    if x0.ndim != 4:
        raise ValueError(f"convolution_40 must be rank 4, got shape={tuple(x0.shape)}")
    if tuple(x1.shape) != tuple(x0.shape) or tuple(x2.shape) != tuple(x0.shape):
        raise ValueError("the three activation inputs must have identical shapes")
    if tuple(x1.stride()) != tuple(x0.stride()) or tuple(x2.stride()) != tuple(x0.stride()):
        raise ValueError("the three activation inputs must have identical strides")

    _, channels, height, width = x0.shape
    if channels <= 0 or height <= 0 or width <= 0:
        raise ValueError(f"invalid activation shape {tuple(x0.shape)}")
    channel_shape = (channels,)
    for name, tensor in (
        ("arg270_1", mean0),
        ("arg271_1", var0),
        ("arg272_1", weight0),
        ("arg273_1", bias0),
        ("arg275_1", mean1),
        ("arg276_1", var1),
        ("arg277_1", weight1),
        ("arg278_1", bias1),
        ("arg265_1", mean2),
        ("arg266_1", var2),
        ("arg267_1", weight2),
        ("arg268_1", bias2),
    ):
        if tuple(tensor.shape) != channel_shape:
            raise ValueError(f"{name} shape must be {channel_shape}, got {tuple(tensor.shape)}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    device = x0.device
    if any(t.device != device for t in tensors):
        raise ValueError("all tensor inputs must be on the same device")
    return tensors


def _torch_oracle(inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    (
        mean0,
        x0,
        var0,
        weight0,
        bias0,
        mean1,
        x1,
        var1,
        weight1,
        bias1,
        mean2,
        x2,
        var2,
        weight2,
        bias2,
    ) = inputs

    def branch(x, mean, var, weight, bias):
        inv_std = torch.reciprocal(torch.sqrt(var + EPS))
        return (x - mean[None, :, None, None]) * inv_std[None, :, None, None] * weight[None, :, None, None] + bias[None, :, None, None]

    out = torch.relu(
        branch(x0, mean0, var0, weight0, bias0)
        + branch(x1, mean1, var1, weight1, bias1)
        + branch(x2, mean2, var2, weight2, bias2)
    )
    if tuple(out.stride()) == tuple(x0.stride()):
        return out
    strided = torch.empty_strided(tuple(x0.shape), tuple(x0.stride()), device=x0.device, dtype=torch.float32)
    strided.copy_(out)
    return strided


@oracle_impl(hardware="H100", shapes="(T([384], f32), T([128, 384, 14, 14], f32), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([128, 384, 14, 14], f32), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([128, 384, 14, 14], f32), T([384], f32), T([384], f32), T([384], f32))")
def oracle_forward(inputs):
    """Run the full Repro.forward three-branch BN-affine sum plus ReLU scope."""
    tensors = _validate_inputs(inputs)
    x0 = tensors[1]
    if not x0.is_cuda:
        return _torch_oracle(tensors)
    if triton is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    batch, channels, height, width = x0.shape
    output = torch.empty_strided(
        tuple(x0.shape),
        tuple(x0.stride()),
        device=x0.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (
        batch,
        triton.cdiv(channels, meta["BLOCK_C"]),
        triton.cdiv(height * width, meta["BLOCK_HW"]),
    )
    _repvgg_bn_sum_relu_kernel[grid](
        tensors[0],
        tensors[1],
        tensors[2],
        tensors[3],
        tensors[4],
        tensors[5],
        tensors[6],
        tensors[7],
        tensors[8],
        tensors[9],
        tensors[10],
        tensors[11],
        tensors[12],
        tensors[13],
        tensors[14],
        output,
        C=channels,
        H=height,
        W=width,
        S_N=x0.stride(0),
        S_C=x0.stride(1),
        S_H=x0.stride(2),
        S_W=x0.stride(3),
        eps=EPS,
    )
    return output


def _normalize_outputs(outputs: Any) -> tuple[Any, ...]:
    if isinstance(outputs, (tuple, list)):
        return tuple(outputs)
    return (outputs,)


def _check_oracle_nan_aware(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
    *,
    atol: float,
    rtol: float,
) -> bool:
    with torch.no_grad():
        eager = instance(*inputs)
        actual = oracle_forward(inputs)
        if any(isinstance(t, torch.Tensor) and t.is_cuda for t in inputs):
            torch.cuda.synchronize()

    eager_list = _normalize_outputs(eager)
    actual_list = _normalize_outputs(actual)
    if len(eager_list) != len(actual_list):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(actual_list)} outputs, "
            f"eager produces {len(eager_list)}"
        )
        return False

    all_pass = True
    for i, (expected, observed) in enumerate(zip(eager_list, actual_list)):
        if not isinstance(expected, torch.Tensor) or not isinstance(observed, torch.Tensor):
            ok = expected == observed
            print(f"  output {i}: {'PASS' if ok else 'FAIL'} (non-tensor)")
            all_pass = all_pass and bool(ok)
            continue

        metadata_ok = (
            expected.shape == observed.shape
            and expected.dtype == observed.dtype
            and expected.stride() == observed.stride()
            and expected.device == observed.device
        )
        if not metadata_ok:
            print(
                f"  output {i}: SCOPE_MISMATCH "
                f"oracle=(shape={list(observed.shape)} stride={observed.stride()} dtype={observed.dtype}) "
                f"eager=(shape={list(expected.shape)} stride={expected.stride()} dtype={expected.dtype})"
            )
            all_pass = False
            continue

        if not expected.is_floating_point():
            ok = torch.equal(expected, observed)
            print(f"  output {i}: {'PASS' if ok else 'FAIL'} (exact, dtype={expected.dtype})")
            all_pass = all_pass and ok
            continue

        expected_f32 = expected.float()
        observed_f32 = observed.float()
        expected_nan = torch.isnan(expected_f32)
        observed_nan = torch.isnan(observed_f32)
        nan_mask_ok = torch.equal(expected_nan, observed_nan)
        finite = ~(expected_nan | observed_nan)
        if finite.any():
            max_diff = (expected_f32[finite] - observed_f32[finite]).abs().max().item()
            finite_ok = torch.allclose(expected_f32[finite], observed_f32[finite], atol=atol, rtol=rtol)
        else:
            max_diff = 0.0
            finite_ok = True

        ok = nan_mask_ok and finite_ok
        print(
            f"  output {i}: {'PASS' if ok else 'FAIL'} "
            f"(shape={list(expected.shape)} dtype={expected.dtype} stride={expected.stride()} "
            f"max_finite_diff={max_diff:.2e} nan_count={int(expected_nan.sum().item())})"
        )
        all_pass = all_pass and ok

    return all_pass


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
        ok = _check_oracle_nan_aware(
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
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
