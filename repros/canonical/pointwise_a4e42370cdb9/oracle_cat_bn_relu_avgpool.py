"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full Inception branch BN-inference affine, ReLU, virtual channel concatenation, and padded 3x3 stride-1 avg_pool2d in branch-specialized Triton stencil kernels that write only the final f32[128,2048,8,8] output, whereas Inductor currently materializes the BN/ReLU branch outputs and fixed channel cats before a separate avg_pool2d consumer; Inductor cannot do this today because scheduler fusion does not push same-channel stencil consumers through virtual cat producers and their broadcast affine pointwise branches; the fix is SCHEDULER_FUSION: let fixed channel-cat operands with per-channel pointwise producers feed pooling stencil codegen directly and store the final concatenated output layout."""
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
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


BATCH = 128
HEIGHT = 8
WIDTH = 8
HW = HEIGHT * WIDTH
OUT_CHANNELS = 2048
EPS = 0.001
BLOCK_C = 2
BLOCK_HW = 64

BRANCHES = (
    # (input index mean, conv, var, weight, bias, channels, output channel offset)
    (0, 1, 2, 3, 4, 320, 0),
    (5, 6, 7, 8, 9, 384, 320),
    (10, 11, 12, 13, 14, 384, 704),
    (15, 16, 17, 18, 19, 384, 1088),
    (20, 21, 22, 23, 24, 384, 1472),
    (25, 26, 27, 28, 29, 192, 1856),
)
OUT_SHAPE = (BATCH, OUT_CHANNELS, HEIGHT, WIDTH)
OUT_STRIDE = (OUT_CHANNELS * HW, HW, WIDTH, 1)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _relu_preserve_nan(x):
        return tl.where((x > 0.0) | (x != x), x, 0.0)


    @triton.jit
    def _branch_bn_relu_avgpool_kernel(
        mean_ptr,
        conv_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        C: tl.constexpr,
        OUT_C_OFFSET: tl.constexpr,
        BLOCK_C_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
        eps: tl.constexpr,
    ):
        batch = tl.program_id(0)
        c_offsets = tl.program_id(1) * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        hw_offsets = tl.arange(0, BLOCK_HW_)

        c_mask = c_offsets < C
        hw_mask = hw_offsets < 64
        out_mask = c_mask[:, None] & hw_mask[None, :]

        oh = hw_offsets // 8
        ow = hw_offsets - oh * 8

        mean = tl.load(mean_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)[:, None]
        var = tl.load(var_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)[:, None]
        weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)[:, None]
        bias = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)[:, None]
        inv_std = 1.0 / tl.sqrt(var + eps)

        input_base = (batch * C + c_offsets[:, None]) * 64
        acc = tl.zeros((BLOCK_C_, BLOCK_HW_), dtype=tl.float32)

        for kh in tl.static_range(3):
            ih = oh + kh - 1
            h_valid = (ih >= 0) & (ih < 8)
            for kw in tl.static_range(3):
                iw = ow + kw - 1
                w_valid = (iw >= 0) & (iw < 8)
                valid = out_mask & h_valid[None, :] & w_valid[None, :]
                raw = tl.load(
                    conv_ptr + input_base + ih[None, :] * 8 + iw[None, :],
                    mask=valid,
                    other=0.0,
                    eviction_policy="evict_last",
                ).to(tl.float32)
                y = (raw - mean) * inv_std
                y = y * weight + bias
                acc += tl.where(valid, _relu_preserve_nan(y), 0.0)

        out_c = OUT_C_OFFSET + c_offsets[:, None]
        out_offsets = (batch * 2048 + out_c) * 64 + hw_offsets[None, :]
        tl.store(out_ptr + out_offsets, acc * (1.0 / 9.0), mask=out_mask)


def _require_f32_tensor(
    name: str,
    value: object,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    return value


def _validate_inputs(inputs: tuple[object, ...] | list[object]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 30:
        raise ValueError(f"{REPRO_ID} expects 30 inputs, got {len(inputs)}")

    checked: list[torch.Tensor] = []
    for branch_idx, (mean_i, conv_i, var_i, weight_i, bias_i, channels, _) in enumerate(BRANCHES):
        checked.append(
            _require_f32_tensor(
                f"branch{branch_idx}_mean",
                inputs[mean_i],
                (channels,),
                (1,),
            )
        )
        checked.append(
            _require_f32_tensor(
                f"branch{branch_idx}_conv",
                inputs[conv_i],
                (BATCH, channels, HEIGHT, WIDTH),
                (channels * HW, HW, WIDTH, 1),
            )
        )
        checked.append(
            _require_f32_tensor(
                f"branch{branch_idx}_var",
                inputs[var_i],
                (channels,),
                (1,),
            )
        )
        checked.append(
            _require_f32_tensor(
                f"branch{branch_idx}_weight",
                inputs[weight_i],
                (channels,),
                (1,),
            )
        )
        checked.append(
            _require_f32_tensor(
                f"branch{branch_idx}_bias",
                inputs[bias_i],
                (channels,),
                (1,),
            )
        )

    device = checked[0].device
    if any(tensor.device != device for tensor in checked):
        raise ValueError("all tensor inputs must be on the same device")
    return tuple(checked)


def _branch_torch(mean, conv, var, weight, bias):
    y = (conv - mean[None, :, None, None]) * torch.reciprocal(
        torch.sqrt(var[None, :, None, None] + EPS)
    )
    y = y * weight[None, :, None, None] + bias[None, :, None, None]
    return torch.relu(y)


def _torch_oracle(inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    outputs = []
    for base in range(0, len(inputs), 5):
        outputs.append(_branch_torch(*inputs[base:base + 5]))
    cat = torch.cat(outputs, dim=1)
    return torch.ops.aten.avg_pool2d.default(cat, [3, 3], [1, 1], [1, 1])


@oracle_impl(hardware="H100", shapes="(T([320], f32), T([128, 320, 8, 8], f32), T([320], f32), T([320], f32), T([320], f32), T([384], f32), T([128, 384, 8, 8], f32), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([128, 384, 8, 8], f32), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([128, 384, 8, 8], f32), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([128, 384, 8, 8], f32), T([384], f32), T([384], f32), T([384], f32), T([192], f32), T([128, 192, 8, 8], f32), T([192], f32), T([192], f32), T([192], f32))")
def oracle_forward(inputs):
    """Run the full branch BN/ReLU, virtual cat, and avg_pool2d computation."""
    checked = _validate_inputs(inputs)
    if not checked[0].is_cuda:
        return _torch_oracle(checked)
    if triton is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=checked[0].device,
        dtype=torch.float32,
    )
    for branch_idx, branch in enumerate(BRANCHES):
        _, _, _, _, _, channels, out_c_offset = branch
        base = branch_idx * 5
        grid = (BATCH, triton.cdiv(channels, BLOCK_C))
        _branch_bn_relu_avgpool_kernel[grid](
            checked[base],
            checked[base + 1],
            checked[base + 2],
            checked[base + 3],
            checked[base + 4],
            out,
            C=channels,
            OUT_C_OFFSET=out_c_offset,
            BLOCK_C_=BLOCK_C,
            BLOCK_HW_=BLOCK_HW,
            eps=EPS,
            num_warps=4,
            num_stages=3,
        )
    return out


def _normalize_outputs(out):
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        result = []
        for item in out:
            result.extend(_normalize_outputs(item))
        return result
    return []


def check_oracle_equal_nan(
    oracle_forward_fn,
    instance,
    inputs,
    *,
    atol: float,
    rtol: float,
    skip_stochastic: bool,
) -> bool:
    """Correctness check matching harness structure but accepting deterministic NaNs."""
    if skip_stochastic and has_stochastic_ops(REPRO_PATH):
        print("  stochastic output skipping requested, but this repro has no stochastic outputs")

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
        expected_nan = torch.isnan(expected_f32)
        actual_nan = torch.isnan(actual_f32)
        nan_mismatch = torch.logical_xor(expected_nan, actual_nan).sum().item()
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
            f"nan_count={expected_nan.sum().item()} nan_mismatch={nan_mismatch})"
        )
        all_pass &= ok

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
        ok = check_oracle_equal_nan(
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
