"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle fuses the full two-branch Moondream RoPE rotate-half pointwise work, duplicated Q/K-like inputs, and final concat layout materialization into one Triton kernel that writes both contiguous f16[1,32,512,64] outputs, whereas Inductor currently emits separate scheduled pointwise and cat/materialization work around each view-permute branch; Inductor cannot do this today because scheduler fusion does not form one producer/consumer group through duplicated slice/neg/cat rotate-half subgraphs and concat output assembly boundaries; the fix is SCHEDULER_FUSION: teach Inductor to recognize and fuse repeated RoPE rotate-half plus tail-copy concat patterns directly into the final permuted output layout."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
    triton = None
    tl = None


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 1
SEQ = 512
HEADS = 32
HEAD_DIM = 64
ROTARY_DIM = 32
OUT_SHAPE = (BATCH, HEADS, SEQ, HEAD_DIM)
OUT_STRIDE = (HEADS * SEQ * HEAD_DIM, SEQ * HEAD_DIM, HEAD_DIM, 1)
CLASSIFICATION = "SCHEDULER_FUSION"


from oracle_harness import (
    oracle_impl,  # noqa: E402
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
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
    def _rope_pair_layout_kernel(
        x0_ptr,
        coeff0_ptr,
        coeff1_ptr,
        x1_ptr,
        out0_ptr,
        out1_ptr,
        x0_s0: tl.constexpr,
        x0_s1: tl.constexpr,
        c0_s1: tl.constexpr,
        c0_s2: tl.constexpr,
        c1_s1: tl.constexpr,
        c1_s2: tl.constexpr,
        x1_s0: tl.constexpr,
        x1_s1: tl.constexpr,
        N_ROWS: tl.constexpr,
        SEQ_: tl.constexpr,
        HEAD_DIM_: tl.constexpr,
        ROTARY_DIM_: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        dims = tl.arange(0, BLOCK_D)
        row_mask = rows < N_ROWS

        head = rows // SEQ_
        pos = rows - head * SEQ_
        col = head[:, None] * HEAD_DIM_ + dims[None, :]

        x0_offsets = pos[:, None] * x0_s0 + col * x0_s1
        x1_offsets = pos[:, None] * x1_s0 + col * x1_s1
        out_offsets = rows[:, None] * HEAD_DIM_ + dims[None, :]
        elem_mask = row_mask[:, None] & (dims[None, :] < HEAD_DIM_)

        rotary = dims < ROTARY_DIM_
        rot_dims = tl.where(dims < ROTARY_DIM_ // 2, dims + ROTARY_DIM_ // 2, dims - ROTARY_DIM_ // 2)
        rot_sign = tl.where(dims < ROTARY_DIM_ // 2, -1.0, 1.0)
        rot_col = head[:, None] * HEAD_DIM_ + rot_dims[None, :]

        c0 = tl.load(
            coeff0_ptr + pos[:, None] * c0_s1 + dims[None, :] * c0_s2,
            mask=row_mask[:, None] & rotary[None, :],
            other=0.0,
        ).to(tl.float32)
        c1 = tl.load(
            coeff1_ptr + pos[:, None] * c1_s1 + dims[None, :] * c1_s2,
            mask=row_mask[:, None] & rotary[None, :],
            other=0.0,
        ).to(tl.float32)

        x0 = tl.load(x0_ptr + x0_offsets, mask=elem_mask, other=0.0)
        x0_rot = tl.load(
            x0_ptr + pos[:, None] * x0_s0 + rot_col * x0_s1,
            mask=row_mask[:, None] & rotary[None, :],
            other=0.0,
        ).to(tl.float32)
        out0 = tl.where(rotary[None, :], x0.to(tl.float32) * c0 + x0_rot * rot_sign[None, :] * c1, x0)
        tl.store(out0_ptr + out_offsets, out0, mask=elem_mask)

        x1 = tl.load(x1_ptr + x1_offsets, mask=elem_mask, other=0.0)
        x1_rot = tl.load(
            x1_ptr + pos[:, None] * x1_s0 + rot_col * x1_s1,
            mask=row_mask[:, None] & rotary[None, :],
            other=0.0,
        ).to(tl.float32)
        out1 = tl.where(rotary[None, :], x1.to(tl.float32) * c0 + x1_rot * rot_sign[None, :] * c1, x1)
        tl.store(out1_ptr + out_offsets, out1, mask=elem_mask)


def _validate_tensor(value: object, shape: tuple[int, ...], dtype: torch.dtype, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} shape {tuple(value.shape)} does not match expected {shape}")
    if value.dtype is not dtype:
        raise TypeError(f"{name} dtype {value.dtype} does not match expected {dtype}")
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    return value


@oracle_impl(hardware="H100", shapes="(T([512, 2048], f16), T([1, 512, 32], f16), T([1, 512, 32], f16), T([512, 2048], f16), S([1, 512, 2048]), S([1, 512, -1, 64]), S([1, 512, 2048]), S([1, 512, -1, 64]))")
def oracle_forward(inputs):
    """Run the exact full Repro.forward scope with one fused Triton kernel."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_rope_pair_layout.py")
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    x0 = _validate_tensor(inputs[0], (SEQ, HEADS * HEAD_DIM), torch.float16, "addmm_138")
    coeff0 = _validate_tensor(inputs[1], (BATCH, SEQ, ROTARY_DIM), torch.float16, "convert_element_type_4")
    coeff1 = _validate_tensor(inputs[2], (BATCH, SEQ, ROTARY_DIM), torch.float16, "convert_element_type_5")
    x1 = _validate_tensor(inputs[3], (SEQ, HEADS * HEAD_DIM), torch.float16, "addmm_139")

    out0 = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=x0.device, dtype=torch.float16)
    out1 = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=x1.device, dtype=torch.float16)

    n_rows = HEADS * SEQ
    block_rows = 4
    _rope_pair_layout_kernel[(triton.cdiv(n_rows, block_rows),)](
        x0,
        coeff0,
        coeff1,
        x1,
        out0,
        out1,
        x0_s0=x0.stride(0),
        x0_s1=x0.stride(1),
        c0_s1=coeff0.stride(1),
        c0_s2=coeff0.stride(2),
        c1_s1=coeff1.stride(1),
        c1_s2=coeff1.stride(2),
        x1_s0=x1.stride(0),
        x1_s1=x1.stride(1),
        N_ROWS=n_rows,
        SEQ_=SEQ,
        HEAD_DIM_=HEAD_DIM,
        ROTARY_DIM_=ROTARY_DIM,
        BLOCK_ROWS=block_rows,
        BLOCK_D=HEAD_DIM,
        num_warps=4,
    )
    return (out0, out1)


def _check_layout(outputs: tuple[torch.Tensor, torch.Tensor]) -> bool:
    ok = True
    for i, output in enumerate(outputs):
        layout_ok = tuple(output.shape) == OUT_SHAPE and output.stride() == OUT_STRIDE
        print(
            f"  output {i} layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(output.shape)} stride={output.stride()})"
        )
        ok = ok and layout_ok
    return ok


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
        ok = check_oracle(
            oracle_forward,
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
            skip_stochastic=not args.no_skip_stochastic,
        )
        with torch.no_grad():
            layout_out = oracle_forward(inputs)
            torch.cuda.synchronize()
        ok = ok and _check_layout(layout_out)
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
