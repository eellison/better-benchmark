"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete inverse-permutation layout materialization by scattering each `bmm_2.view(8, 12, 4096, 64)[b, h, q, :]` vector directly to its final contiguous `[b * 4096 + getitem_3[b, h, q], h * 64:(h + 1) * 64]` output slice, whereas Inductor materializes the `scatter(iota)` inverse-index tensor, gathers through that tensor, then permutes/clones/views the result; Inductor cannot do this today because its scheduler treats the data-dependent `scatter.src` index construction and downstream `gather`/layout copy as separate memory effects instead of recognizing the permutation-inversion producer-consumer pair; the fix is SCHEDULER_FUSION: add a layout-indexing fusion that rewrites scatter-built inverse permutations feeding gather plus clone into a direct destination-layout scatter over the original permutation, with the surrounding view/permute materialization folded into the same kernel."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None

from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _direct_permutation_layout_kernel(
        bmm_ptr,
        permutation_ptr,
        out_ptr,
        Q: tl.constexpr,
        H: tl.constexpr,
        C: tl.constexpr,
        HC: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        bh = tl.program_id(0)
        q_block = tl.program_id(1)
        b = bh // H
        h = bh - b * H

        q_offsets = q_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
        c_offsets = tl.arange(0, BLOCK_C)
        q_mask = q_offsets < Q
        c_mask = c_offsets < C

        destination_q = tl.load(
            permutation_ptr + bh * Q + q_offsets,
            mask=q_mask,
            other=0,
        )
        source_offsets = (bh * Q + q_offsets[:, None]) * C + c_offsets[None, :]
        output_offsets = (b * Q + destination_q[:, None]) * HC + h * C + c_offsets[None, :]
        mask = q_mask[:, None] & c_mask[None, :]

        values = tl.load(bmm_ptr + source_offsets, mask=mask, other=0.0)
        tl.store(out_ptr + output_offsets, values, mask=mask)


def _shape_tuple(value: object) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _validate_inputs(inputs: tuple[object, ...] | list[object]) -> tuple[int, int, int, int, tuple[int, int]]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_layout_index_fusion.py")
    if len(inputs) != 8:
        raise ValueError(f"expected 8 inputs from make_inputs(), got {len(inputs)}")

    bmm_2, getitem_3, shape0, shape1, shape2, shape3, shape4, shape5 = inputs
    if not isinstance(bmm_2, torch.Tensor) or not isinstance(getitem_3, torch.Tensor):
        raise TypeError("expected tensor inputs for bmm_2 and getitem_3")
    if bmm_2.device.type != "cuda" or getitem_3.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if bmm_2.dtype != torch.float32 or getitem_3.dtype != torch.int64:
        raise TypeError("expected f32 bmm_2 and i64 getitem_3")
    if not bmm_2.is_contiguous() or not getitem_3.is_contiguous():
        raise ValueError("oracle expects the captured contiguous input layouts")

    view0 = _shape_tuple(shape0)
    view1 = _shape_tuple(shape1)
    scatter_shape = _shape_tuple(shape2)
    gather_shape = _shape_tuple(shape3)
    view3 = _shape_tuple(shape4)
    out_shape = _shape_tuple(shape5)
    if len(view1) != 4:
        raise ValueError(f"unexpected viewed input rank: {view1}")

    b, h, q, c = view1
    expected_numel = 1
    for dim in view1:
        expected_numel *= dim
    if bmm_2.numel() != expected_numel:
        raise ValueError(f"unexpected bmm_2 numel: {bmm_2.numel()} for view {view1}")
    if view0[:2] != (b, h) or view0[-2] * view0[-1] != q:
        raise ValueError(f"view shape parameters do not describe the captured flattening: {view0}, {view1}")
    if tuple(getitem_3.shape) != (b, h, q):
        raise ValueError(f"unexpected getitem_3 shape: {tuple(getitem_3.shape)}")
    if scatter_shape != (b, h, q) or gather_shape != (b, h, q, c):
        raise ValueError("scatter/gather shape parameters do not match inputs")
    if view3 != (b, q, h * c) or out_shape != (b * q, h * c):
        raise ValueError(f"unexpected output view shape parameters: {view3}, {out_shape}")
    return b, h, q, c, out_shape


@oracle_impl(hardware="H100", shapes="(T([6144, 64, 64], f32), T([8, 12, 4096], i64, gen=Perm(4096)), S([8, 12, 64, 64, 64]), S([8, 12, 4096, 64]), S([8, 12, 4096]), S([8, 12, 4096, 64]), S([8, 4096, 768]), S([32768, 768]))")
def oracle_forward(inputs):
    """Compute the same full output as Repro.forward(*inputs)."""
    bmm_2, getitem_3 = inputs[0], inputs[1]
    b, h, q, c, out_shape = _validate_inputs(inputs)

    clone_storage = torch.empty((b, q, h, c), device=bmm_2.device, dtype=bmm_2.dtype)
    grid = (b * h, triton.cdiv(q, 32))
    _direct_permutation_layout_kernel[grid](
        bmm_2,
        getitem_3,
        clone_storage,
        Q=q,
        H=h,
        C=c,
        HC=h * c,
        BLOCK_Q=32,
        BLOCK_C=triton.next_power_of_2(c),
    )
    return clone_storage.view(out_shape)


def main():
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
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
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(
                        f"WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
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
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
