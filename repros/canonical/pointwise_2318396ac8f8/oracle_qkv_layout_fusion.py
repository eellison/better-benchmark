"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Swin qkv layout materialization returned by Repro.forward in one Triton kernel, including the q scale, q/k/v unbind, k transpose, three contiguous clone/view outputs, and the exact [4096,49,32], [4096,32,49], [4096,49,32] output contract, whereas Inductor currently lowers the sibling clone/layout materializations as separate generic layout-copy work; Inductor cannot do this today because its scheduler does not fuse multiple users of an unbound view/permute producer when each user requires a different contiguous output layout; the fix is SCHEDULER_FUSION: teach layout-copy scheduling to emit a multi-output fused producer for sibling qkv materializations with per-output index maps."""
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
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


BATCH = 128
WINDOW = 49
QKV = 3
HEADS = 32
HEAD_DIM = 32
ROWS = BATCH * HEADS
IN_FEATURES = QKV * HEADS * HEAD_DIM
OUT_NUMEL = ROWS * WINDOW * HEAD_DIM
Q_SCALE = 0.1767766952966369


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _qkv_layout_kernel(
        addmm_ptr,
        q_out_ptr,
        k_out_ptr,
        v_out_ptr,
        BLOCK_S: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = row // 32
        head = row - batch * 32

        s = tl.arange(0, BLOCK_S)[:, None]
        d = tl.arange(0, BLOCK_D)[None, :]
        mask = (s < 49) & (d < 32)

        input_base = (batch * 49 + s) * 3072 + head * 32 + d
        q_vals = tl.load(addmm_ptr + input_base, mask=mask, other=0.0)
        k_vals = tl.load(addmm_ptr + input_base + 1024, mask=mask, other=0.0)
        v_vals = tl.load(addmm_ptr + input_base + 2048, mask=mask, other=0.0)

        qv_offsets = row * 49 * 32 + s * 32 + d
        k_offsets = row * 32 * 49 + d * 49 + s
        tl.store(q_out_ptr + qv_offsets, q_vals * 0.1767766952966369, mask=mask)
        tl.store(k_out_ptr + k_offsets, k_vals, mask=mask)
        tl.store(v_out_ptr + qv_offsets, v_vals, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _validate_shape_param(name: str, actual: Any, expected: tuple[int, ...]) -> None:
    got = _shape_tuple(actual)
    if got != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {got}")


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    if len(inputs) != 9:
        raise ValueError(f"{REPRO_ID} expects 9 inputs, got {len(inputs)}")

    addmm_92 = inputs[0]
    if not isinstance(addmm_92, torch.Tensor):
        raise TypeError(f"addmm_92 must be a tensor, got {type(addmm_92)!r}")
    if tuple(addmm_92.shape) != (BATCH * WINDOW, IN_FEATURES):
        raise ValueError(
            f"addmm_92 has shape {tuple(addmm_92.shape)}, "
            f"expected {(BATCH * WINDOW, IN_FEATURES)}"
        )
    if tuple(addmm_92.stride()) != (IN_FEATURES, 1):
        raise ValueError(
            f"addmm_92 has stride {tuple(addmm_92.stride())}, "
            f"expected {(IN_FEATURES, 1)}"
        )
    if addmm_92.dtype != torch.float32:
        raise TypeError(f"addmm_92 has dtype {addmm_92.dtype}, expected torch.float32")
    if not addmm_92.is_cuda:
        raise RuntimeError("addmm_92 must be a CUDA tensor for this Triton oracle")

    _validate_shape_param("_shape_param_0", inputs[1], (BATCH, WINDOW, IN_FEATURES))
    shape_1 = _shape_tuple(inputs[2])
    if shape_1[:4] != (BATCH, WINDOW, QKV, HEADS) or shape_1[4] not in (-1, HEAD_DIM):
        raise ValueError(
            "_shape_param_1 mismatch: expected "
            f"{(BATCH, WINDOW, QKV, HEADS, -1)} compatible, got {shape_1}"
        )
    _validate_shape_param("_shape_param_2", inputs[3], (BATCH, HEADS, WINDOW, HEAD_DIM))
    _validate_shape_param("_shape_param_3", inputs[4], (ROWS, WINDOW, HEAD_DIM))
    _validate_shape_param("_shape_param_4", inputs[5], (BATCH, HEADS, HEAD_DIM, WINDOW))
    _validate_shape_param("_shape_param_5", inputs[6], (ROWS, HEAD_DIM, WINDOW))
    _validate_shape_param("_shape_param_6", inputs[7], (BATCH, HEADS, WINDOW, HEAD_DIM))
    _validate_shape_param("_shape_param_7", inputs[8], (ROWS, WINDOW, HEAD_DIM))
    return addmm_92


@oracle_impl(hardware="H100", shapes="(T([6272, 3072], f32), S([128, 49, 3072]), S([128, 49, 3, 32, -1]), S([128, 32, 49, 32]), S([4096, 49, 32]), S([128, 32, 32, 49]), S([4096, 32, 49]), S([128, 32, 49, 32]), S([4096, 49, 32]))")
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

    addmm_92 = _validate_inputs(inputs)
    q_out = torch.empty((ROWS, WINDOW, HEAD_DIM), device=addmm_92.device, dtype=addmm_92.dtype)
    k_out = torch.empty((ROWS, HEAD_DIM, WINDOW), device=addmm_92.device, dtype=addmm_92.dtype)
    v_out = torch.empty((ROWS, WINDOW, HEAD_DIM), device=addmm_92.device, dtype=addmm_92.dtype)

    _qkv_layout_kernel[(ROWS,)](
        addmm_92,
        q_out,
        k_out,
        v_out,
        BLOCK_S=64,
        BLOCK_D=32,
        num_warps=4,
    )
    return (q_out, k_out, v_out)


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
