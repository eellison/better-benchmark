"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete LLaMA cache slice-update plus returned head-major clone scope with one Triton kernel that writes only `arg85_1[:32, 1:33, :, :]` from `mm_51.view(32, 32, 8, 64)` while directly materializing the fresh contiguous `[256, 33, 64]` output, whereas Inductor sees the captured `copy -> slice_scatter -> slice_scatter -> permute -> clone -> copy_` layout chain through generic slice-scatter materialization; Inductor cannot express this direct in-place slice update and derived head-layout clone as one guarded layout stencil today, so the fix is NEW_PATTERN: add a specialized slice-update/head-layout materialization pattern that elides unchanged-region copies and emits the final clone layout directly."""
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


# --- Oracle kernel(s) ---

BATCH = 32
ARG_BATCH = 64
ARG_SEQ = 1024
HEADS = 8
HEAD_DIM = 64
UPDATE_SEQ = 32
OUT_SEQ = 33
MM_HIDDEN = HEADS * HEAD_DIM

ARG_SHAPE = (ARG_BATCH, ARG_SEQ, HEADS, HEAD_DIM)
ARG_STRIDE = (ARG_SEQ * MM_HIDDEN, MM_HIDDEN, HEAD_DIM, 1)
MM_SHAPE = (BATCH * UPDATE_SEQ, MM_HIDDEN)
MM_STRIDE = (MM_HIDDEN, 1)
OUT0_SHAPE = (BATCH * HEADS, OUT_SEQ, HEAD_DIM)
OUT0_STRIDE = (OUT_SEQ * HEAD_DIM, HEAD_DIM, 1)
OUT0_NUMEL = BATCH * HEADS * OUT_SEQ * HEAD_DIM
CLASSIFICATION = "NEW_PATTERN"

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 2048}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_N": 4096}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_N": 8192}, num_warps=8, num_stages=3),
        ],
        key=["N"],
    )
    @triton.jit
    def _slice_update_head_layout_kernel(
        arg_ptr,
        mm_ptr,
        out_ptr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N

        dim = offsets % 64
        tmp = offsets // 64
        out_seq = tmp % 33
        batch_head = tmp // 33
        head = batch_head % 8
        batch = batch_head // 8

        update_value = out_seq > 0
        mm_offsets = ((batch * 32 + (out_seq - 1)) * 512) + head * 64 + dim
        arg_offsets = batch * 524288 + out_seq * 512 + head * 64 + dim

        mm_values = tl.load(mm_ptr + mm_offsets, mask=mask & update_value, other=0.0)
        seq0_values = tl.load(arg_ptr + arg_offsets, mask=mask & ~update_value, other=0.0)
        values = tl.where(update_value, mm_values, seq0_values)

        tl.store(out_ptr + offsets, values, mask=mask)
        tl.store(arg_ptr + arg_offsets, mm_values, mask=mask & update_value)


def _shape_tuple(value) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"expected shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _require_shape_arg(name: str, value, expected: tuple[int, ...]) -> None:
    actual = _shape_tuple(value)
    if actual != expected:
        raise ValueError(f"{name} must be {expected}, got {actual}")


def _require_tensor(
    name: str,
    value,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} shape must be {shape}, got {tuple(value.shape)}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} stride must be {stride}, got {tuple(value.stride())}")
    if value.dtype is not torch.float32:
        raise TypeError(f"{name} dtype must be torch.float32, got {value.dtype}")
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    return value


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
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects six inputs, got {len(inputs)}")

    arg85_1 = _require_tensor("arg85_1", inputs[0], ARG_SHAPE, ARG_STRIDE)
    mm_51 = _require_tensor("mm_51", inputs[1], MM_SHAPE, MM_STRIDE)
    _require_shape_arg("_shape_param_0", inputs[2], (BATCH, UPDATE_SEQ, MM_HIDDEN))
    _require_shape_arg("_shape_param_1", inputs[3], (BATCH, UPDATE_SEQ, HEADS, HEAD_DIM))
    _require_shape_arg("_shape_param_2", inputs[4], (BATCH, HEADS, OUT_SEQ, HEAD_DIM))
    _require_shape_arg("_shape_param_3", inputs[5], OUT0_SHAPE)

    output = torch.empty_strided(
        OUT0_SHAPE,
        OUT0_STRIDE,
        device=arg85_1.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (triton.cdiv(OUT0_NUMEL, meta["BLOCK_N"]),)
    _slice_update_head_layout_kernel[grid](
        arg85_1,
        mm_51,
        output,
        N=OUT0_NUMEL,
    )
    return output, arg85_1


def _check_output_contract(inputs, outputs) -> bool:
    if not isinstance(outputs, tuple) or len(outputs) != 2:
        print("  output contract: FAIL (expected two-tensor tuple)")
        return False

    output0, output1 = outputs
    output0_ok = (
        isinstance(output0, torch.Tensor)
        and tuple(output0.shape) == OUT0_SHAPE
        and tuple(output0.stride()) == OUT0_STRIDE
        and output0.dtype is torch.float32
        and output0.data_ptr() != inputs[0].data_ptr()
    )
    output1_ok = (
        isinstance(output1, torch.Tensor)
        and tuple(output1.shape) == ARG_SHAPE
        and tuple(output1.stride()) == ARG_STRIDE
        and output1.dtype is torch.float32
        and output1.data_ptr() == inputs[0].data_ptr()
    )
    print(
        f"  output 0 contract: {'PASS' if output0_ok else 'FAIL'} "
        f"(shape={list(output0.shape)} stride={output0.stride()} alias_input0="
        f"{output0.data_ptr() == inputs[0].data_ptr()})"
    )
    print(
        f"  output 1 contract: {'PASS' if output1_ok else 'FAIL'} "
        f"(shape={list(output1.shape)} stride={output1.stride()} alias_input0="
        f"{output1.data_ptr() == inputs[0].data_ptr()})"
    )
    return output0_ok and output1_ok


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
        with torch.no_grad():
            contract_outputs = oracle_forward(inputs)
            torch.cuda.synchronize()
        ok = ok and _check_output_contract(inputs, contract_outputs)
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
