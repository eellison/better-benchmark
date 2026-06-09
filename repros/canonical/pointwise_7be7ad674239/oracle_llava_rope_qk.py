"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete LLaVA Q/K RoPE pointwise and layout scope, including both viewed-permuted matmul inputs, shared cos/sin coefficient broadcasts, rotate-half slice/neg/cat semantics, and both returned non-contiguous f16[1,32,512,128] outputs by directly loading the signed rotated mate at dim +/- 64 in one row-tiled multi-output Triton kernel, whereas Inductor already fuses the full graph into one generic pointwise kernel but preserves the decomposed rotate-half concat as predicated two-load/where logic for each Q/K branch; Inductor cannot do this today because its algebraic simplifier does not canonicalize cat([-x[..., half:], x[..., :half]]) through view/permute metadata into an affine rotated-index plus sign expression before pointwise codegen; the fix is ALGEBRAIC_ELIMINATION: add a guarded rotate-half canonicalization that rewrites those concat consumers to signed affine loads while preserving the final non-contiguous output strides."""
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

BATCH = 1
SEQ = 512
HEADS = 32
HEAD_DIM = 128
HIDDEN = HEADS * HEAD_DIM
OUT_SHAPE = (BATCH, HEADS, SEQ, HEAD_DIM)
OUT_STRIDE = (HEADS * SEQ * HEAD_DIM, HEAD_DIM, HEADS * HEAD_DIM, 1)
SHAPE_PARAMS = (
    (1, 512, 4096),
    (1, 512, -1, 128),
    (1, 512, 4096),
    (1, 512, -1, 128),
)

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


if triton is not None:

    @triton.jit
    def _llava_rope_qk_kernel(
        q_ptr,
        coeff0_ptr,
        coeff1_ptr,
        k_ptr,
        out_q_ptr,
        out_k_ptr,
        q_s0: tl.constexpr,
        q_s1: tl.constexpr,
        coeff0_s1: tl.constexpr,
        coeff0_s2: tl.constexpr,
        coeff1_s1: tl.constexpr,
        coeff1_s2: tl.constexpr,
        k_s0: tl.constexpr,
        k_s1: tl.constexpr,
        out_h_stride: tl.constexpr,
        out_seq_stride: tl.constexpr,
        N_ROWS: tl.constexpr,
        HEADS_: tl.constexpr,
        HEAD_DIM_: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        dims = tl.arange(0, BLOCK_D)
        row_mask = rows < N_ROWS
        elem_mask = row_mask[:, None] & (dims[None, :] < HEAD_DIM_)

        pos = rows // HEADS_
        head = rows - pos * HEADS_
        cols = head[:, None] * HEAD_DIM_ + dims[None, :]

        half_dim = HEAD_DIM_ // 2
        rot_dims = tl.where(dims < half_dim, dims + half_dim, dims - half_dim)
        rot_cols = head[:, None] * HEAD_DIM_ + rot_dims[None, :]
        rot_sign = tl.where(dims < half_dim, -1.0, 1.0)

        q_offsets = pos[:, None] * q_s0 + cols * q_s1
        q_rot_offsets = pos[:, None] * q_s0 + rot_cols * q_s1
        k_offsets = pos[:, None] * k_s0 + cols * k_s1
        k_rot_offsets = pos[:, None] * k_s0 + rot_cols * k_s1
        coeff0_offsets = pos[:, None] * coeff0_s1 + dims[None, :] * coeff0_s2
        coeff1_offsets = pos[:, None] * coeff1_s1 + dims[None, :] * coeff1_s2
        out_offsets = head[:, None] * out_h_stride + pos[:, None] * out_seq_stride + dims[None, :]

        coeff0 = tl.load(coeff0_ptr + coeff0_offsets, mask=elem_mask, other=0.0).to(tl.float32)
        coeff1 = tl.load(coeff1_ptr + coeff1_offsets, mask=elem_mask, other=0.0).to(tl.float32)

        q = tl.load(q_ptr + q_offsets, mask=elem_mask, other=0.0).to(tl.float32)
        q_rot = tl.load(q_ptr + q_rot_offsets, mask=elem_mask, other=0.0).to(tl.float32)
        q_out = q * coeff0 + q_rot * rot_sign[None, :] * coeff1
        tl.store(out_q_ptr + out_offsets, q_out, mask=elem_mask)

        k = tl.load(k_ptr + k_offsets, mask=elem_mask, other=0.0).to(tl.float32)
        k_rot = tl.load(k_ptr + k_rot_offsets, mask=elem_mask, other=0.0).to(tl.float32)
        k_out = k * coeff0 + k_rot * rot_sign[None, :] * coeff1
        tl.store(out_k_ptr + out_offsets, k_out, mask=elem_mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple, torch.Size)):
        raise TypeError(f"{REPRO_ID} expected a shape parameter, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _validate_tensor(
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{REPRO_ID} {name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{REPRO_ID} {name} expects shape {shape}, got {tuple(value.shape)}")
    if value.dtype is not dtype:
        raise TypeError(f"{REPRO_ID} {name} expects dtype {dtype}, got {value.dtype}")
    if not value.is_cuda:
        raise ValueError(f"{REPRO_ID} {name} must be a CUDA tensor")
    return value


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    q = _validate_tensor(inputs[0], (SEQ, HIDDEN), torch.float16, "mm_217")
    coeff0 = _validate_tensor(inputs[1], (BATCH, SEQ, HEAD_DIM), torch.float16, "convert_element_type_4")
    coeff1 = _validate_tensor(inputs[2], (BATCH, SEQ, HEAD_DIM), torch.float16, "convert_element_type_5")
    k = _validate_tensor(inputs[3], (SEQ, HIDDEN), torch.float16, "mm_218")

    devices = {q.device, coeff0.device, coeff1.device, k.device}
    if len(devices) != 1:
        raise ValueError(f"{REPRO_ID} tensor inputs must be on the same CUDA device")

    for index, expected in enumerate(SHAPE_PARAMS, start=4):
        actual = _shape_tuple(inputs[index])
        if actual != expected:
            raise ValueError(f"{REPRO_ID} _shape_param_{index - 4} expects {expected}, got {actual}")

    return q, coeff0, coeff1, k


@oracle_impl(hardware="H100", shapes="(T([512, 4096], f16), T([1, 512, 128], f16), T([1, 512, 128], f16), T([512, 4096], f16), S([1, 512, 4096]), S([1, 512, -1, 128]), S([1, 512, 4096]), S([1, 512, -1, 128]))")
def oracle_forward(inputs):
    """Run the exact full Repro.forward scope with one fused Triton kernel."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_llava_rope_qk.py")

    q, coeff0, coeff1, k = _validate_inputs(inputs)
    out_q = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=q.device, dtype=q.dtype)
    out_k = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=k.device, dtype=k.dtype)

    n_rows = SEQ * HEADS
    block_rows = 4
    _llava_rope_qk_kernel[(triton.cdiv(n_rows, block_rows),)](
        q,
        coeff0,
        coeff1,
        k,
        out_q,
        out_k,
        q_s0=q.stride(0),
        q_s1=q.stride(1),
        coeff0_s1=coeff0.stride(1),
        coeff0_s2=coeff0.stride(2),
        coeff1_s1=coeff1.stride(1),
        coeff1_s2=coeff1.stride(2),
        k_s0=k.stride(0),
        k_s1=k.stride(1),
        out_h_stride=out_q.stride(1),
        out_seq_stride=out_q.stride(2),
        N_ROWS=n_rows,
        HEADS_=HEADS,
        HEAD_DIM_=HEAD_DIM,
        BLOCK_ROWS=block_rows,
        BLOCK_D=HEAD_DIM,
        num_warps=4,
        num_stages=3,
    )
    return (out_q, out_k)


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
