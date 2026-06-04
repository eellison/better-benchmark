"""
Oracle for pointwise_a14dcfc06344

Gap diagnosis:
  Classification: SCHEDULER_FUSION
  What oracle does differently: fuses each branch's BN-affine + ReLU producer directly into the 3x3 avg-pool stencil while writing the final concatenated channel layout.
  What Inductor change would fix: teach the scheduler to fuse layout-producing cat operands into same-channel stencil consumers and assign final-layout writes to the stencil tiles.
"""
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

# Import shared oracle infrastructure (installed via pip install -e .)
from oracle_harness import (
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    bench_oracle,
    bench_oracle_all_shapes,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def get_hardware_info():
    """Get GPU hardware properties, tolerating torch property name drift."""
    props = torch.cuda.get_device_properties(0)
    shared_mem = getattr(
        props,
        "max_shared_memory_per_multiprocessor",
        getattr(props, "shared_memory_per_multiprocessor", None),
    )
    return {
        "name": props.name,
        "sm_major": props.major,
        "sm_minor": props.minor,
        "num_sms": props.multi_processor_count,
        "shared_mem_per_sm": shared_mem,
        "total_mem_gb": props.total_memory / 1e9,
    }


def _normalize_outputs(out):
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        result = []
        for item in out:
            result.extend(_normalize_outputs(item))
        return result
    return []


def check_oracle_equal_nan(oracle_forward_fn, instance, inputs, *, atol, rtol):
    """Correctness check for deterministic NaNs from random BN variance inputs."""
    with torch.no_grad():
        eager = instance(*inputs)
        oracle_out = oracle_forward_fn(inputs)

    eager_list = _normalize_outputs(eager)
    oracle_list = _normalize_outputs(oracle_out)
    if len(oracle_list) != len(eager_list):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(oracle_list)} outputs, "
            f"eager produces {len(eager_list)}"
        )
        return False

    all_pass = True
    for idx, (expected, actual) in enumerate(zip(eager_list, oracle_list)):
        if expected.shape != actual.shape:
            print(
                f"  output {idx}: SCOPE_MISMATCH shape oracle={list(actual.shape)} "
                f"eager={list(expected.shape)}"
            )
            all_pass = False
            continue
        if expected.dtype != actual.dtype:
            print(
                f"  output {idx}: WARNING dtype mismatch "
                f"oracle={actual.dtype} eager={expected.dtype}"
            )

        if not expected.is_floating_point():
            ok = torch.equal(expected, actual)
            print(f"  output {idx}: {'PASS' if ok else 'FAIL'} (exact, dtype={expected.dtype})")
            all_pass &= ok
            continue

        expected_f32 = expected.float()
        actual_f32 = actual.float()
        finite = torch.isfinite(expected_f32) & torch.isfinite(actual_f32)
        if finite.any():
            max_diff = (expected_f32[finite] - actual_f32[finite]).abs().max().item()
        else:
            max_diff = 0.0
        nan_count = torch.isnan(expected_f32).sum().item()
        ok = torch.allclose(
            expected_f32,
            actual_f32,
            atol=atol,
            rtol=rtol,
            equal_nan=True,
        )
        status = "PASS" if ok else "FAIL"
        print(
            f"  output {idx}: {status} (shape={list(expected.shape)} "
            f"dtype={expected.dtype} finite_max_diff={max_diff:.2e} "
            f"nan_count={nan_count})"
        )
        all_pass &= ok

    return all_pass


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_C": 1, "BLOCK_HW": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 2, "BLOCK_HW": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 4, "BLOCK_HW": 512}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_C": 8, "BLOCK_HW": 512}, num_warps=8, num_stages=3),
        ],
        key=["C", "HW"],
    )
    @triton.jit
    def _bn_relu_avgpool_branch_kernel(
        input_ptr,
        mean_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        B: tl.constexpr,
        C: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        HW: tl.constexpr,
        C_OUT: tl.constexpr,
        BRANCH: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        """Fused BN-affine + ReLU + zero-padded 3x3 avg-pool for one cat branch."""
        batch = tl.program_id(0)
        c_block = tl.program_id(1)

        c_offsets = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        hw_offsets = tl.arange(0, BLOCK_HW)
        c = c_offsets[:, None]
        hw = hw_offsets[None, :]

        c_mask = c_offsets < C
        hw_mask = hw_offsets < HW
        out_mask = c_mask[:, None] & hw_mask[None, :]

        oh = hw_offsets // W
        ow = hw_offsets - oh * W

        mean = tl.load(mean_ptr + c_offsets, mask=c_mask, other=0.0)[:, None]
        var = tl.load(var_ptr + c_offsets, mask=c_mask, other=0.0)[:, None]
        weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0.0)[:, None]
        bias = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0)[:, None]
        inv_std = 1.0 / tl.sqrt(var + 0.001)

        acc = tl.zeros((BLOCK_C, BLOCK_HW), dtype=tl.float32)
        input_plane = batch * C * HW + c * HW

        for kh in tl.static_range(3):
            ih = oh + kh - 1
            h_valid = (ih >= 0) & (ih < H)
            for kw in tl.static_range(3):
                iw = ow + kw - 1
                w_valid = (iw >= 0) & (iw < W)
                valid = out_mask & h_valid[None, :] & w_valid[None, :]
                input_offsets = input_plane + ih[None, :] * W + iw[None, :]
                raw = tl.load(
                    input_ptr + input_offsets,
                    mask=valid,
                    other=0.0,
                    eviction_policy="evict_last",
                )
                val = (raw - mean) * inv_std
                val = val * weight + bias
                relu = tl.where(val != val, val, tl.maximum(val, 0.0))
                acc += tl.where(valid, relu, 0.0)

        output_offsets = (
            batch * C_OUT * HW
            + (BRANCH * C + c) * HW
            + hw
        )
        tl.store(output_ptr + output_offsets, acc * (1.0 / 9.0), mask=out_mask)


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
        raise RuntimeError("Triton is required for oracle_layout_stencil.py")
    if len(inputs) != 20:
        raise ValueError(f"expected 20 inputs, got {len(inputs)}")

    branches = (
        (inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]),
        (inputs[5], inputs[6], inputs[7], inputs[8], inputs[9]),
        (inputs[10], inputs[11], inputs[12], inputs[13], inputs[14]),
        (inputs[15], inputs[16], inputs[17], inputs[18], inputs[19]),
    )

    first_conv = branches[0][1]
    if tuple(first_conv.shape) != (128, 192, 17, 17):
        raise ValueError(f"unexpected convolution shape: {tuple(first_conv.shape)}")
    if first_conv.dtype != torch.float32:
        raise ValueError(f"unexpected convolution dtype: {first_conv.dtype}")
    if not first_conv.is_cuda:
        raise ValueError("oracle_layout_stencil.py expects CUDA inputs")

    batch, channels, height, width = first_conv.shape
    output_channels = channels * len(branches)
    output = torch.empty(
        (batch, output_channels, height, width),
        device=first_conv.device,
        dtype=torch.float32,
    )

    for branch_idx, (mean, conv, var, weight, bias) in enumerate(branches):
        if tuple(conv.shape) != (batch, channels, height, width):
            raise ValueError(f"branch {branch_idx} convolution shape mismatch: {tuple(conv.shape)}")
        for name, tensor in (
            ("mean", mean),
            ("var", var),
            ("weight", weight),
            ("bias", bias),
        ):
            if tuple(tensor.shape) != (channels,):
                raise ValueError(
                    f"branch {branch_idx} {name} shape mismatch: {tuple(tensor.shape)}"
                )
            if tensor.dtype != torch.float32:
                raise ValueError(f"branch {branch_idx} {name} dtype mismatch: {tensor.dtype}")

        grid = lambda meta: (batch, triton.cdiv(channels, meta["BLOCK_C"]))
        _bn_relu_avgpool_branch_kernel[grid](
            conv,
            mean,
            var,
            weight,
            bias,
            output,
            B=batch,
            C=channels,
            H=height,
            W=width,
            HW=height * width,
            C_OUT=output_channels,
            BRANCH=branch_idx,
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
        ok = check_oracle_equal_nan(
            oracle_forward,
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
