"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete NFNet sigmoid-gradient channel vector by reducing each coalesced `(N, channel-block, 7, 7)` tile, applying the sigmoid-derivative epilogue, and atomically accumulating the final `[1536]` vector without a materialized `[128, 1536]` reduction buffer, whereas Inductor currently schedules a spatial `[128, 1536, 1, 1]` reduction and a second sigmoid-derivative batch reduction over that intermediate; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K template that fuses an inner spatial reduction with a dependent per-row epilogue and the outer channel reduction; the fix is COOPERATIVE_SPLIT_K: add a split-K channel-reduction template that accumulates row-local spatial sums directly into the final channel vector."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps syntax checks usable without Triton.
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
    get_shape_key,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

N = 128
C = 1536
H = 7
W = 7
HW = H * W
BLOCK_HW = 64
BLOCK_C = 4
BLOCK_BATCH = 1
ZERO_BLOCK = 1024


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _zero_output_kernel(
        out_ptr,
        C_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        tl.store(out_ptr + offsets, tl.zeros((BLOCK_C_,), tl.float32), mask=offsets < C_)

    @triton.jit
    def _spatial_sigmoid_atomic_kernel(
        getitem_ptr,
        arg401_ptr,
        arg404_ptr,
        out_ptr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
        BLOCK_BATCH_: tl.constexpr,
    ):
        c_block = tl.program_id(0)
        n_tile = tl.program_id(1)

        c_vec = c_block * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        hw = tl.arange(0, BLOCK_HW_)[:, None]
        c = c_vec[None, :]
        channel_acc = tl.zeros((BLOCK_C_,), dtype=tl.float32)
        for i in tl.static_range(0, BLOCK_BATCH_):
            n = n_tile * BLOCK_BATCH_ + i
            active = (hw < HW_) & (c < C_)

            offsets = n * (C_ * HW_) + c * HW_ + hw
            getitem = tl.load(
                getitem_ptr + offsets,
                mask=active,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            arg401 = tl.load(
                arg401_ptr + offsets,
                mask=active,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)

            spatial_terms = (getitem * arg401) * 0.4
            spatial_sum = tl.sum(tl.where(active, spatial_terms, 0.0), axis=0)

            sigmoid = tl.sigmoid(tl.load(arg404_ptr + n * C_ + c_vec, mask=c_vec < C_, other=0.0).to(tl.float32))
            sigmoid_deriv = sigmoid * (1.0 - sigmoid)
            channel_acc += spatial_sum * sigmoid_deriv
        tl.atomic_add(out_ptr + c_vec, channel_acc, sem="relaxed", mask=c_vec < C_)


def _torch_oracle(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    getitem, arg401_1, arg404_1 = inputs
    mul_tensor = torch.ops.aten.mul.Tensor(getitem, 0.2)
    mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 2.0)
    mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, arg401_1)
    sigmoid_default = torch.ops.aten.sigmoid.default(arg404_1)
    sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2, 3], True)
    sub_tensor = torch.ops.aten.sub.Tensor(1, sigmoid_default)
    mul_tensor_3 = torch.ops.aten.mul.Tensor(sigmoid_default, sub_tensor)
    mul_tensor_4 = torch.ops.aten.mul.Tensor(sum_dim_int_list, mul_tensor_3)
    return torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 2, 3])


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 3:
        raise ValueError(f"expected 3 inputs, got {len(inputs)}")
    getitem, arg401_1, arg404_1 = inputs
    expected = (
        ("getitem", getitem, (N, C, H, W)),
        ("arg401_1", arg401_1, (N, C, H, W)),
        ("arg404_1", arg404_1, (N, C, 1, 1)),
    )
    for name, value, shape in expected:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} expected shape {shape}, got {tuple(value.shape)}")
        if value.dtype is not torch.float32:
            raise TypeError(f"{name} expected dtype torch.float32, got {value.dtype}")
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous for this fixed-layout oracle")
    if getitem.device != arg401_1.device or getitem.device != arg404_1.device:
        raise ValueError("all tensor inputs must be on the same device")
    return getitem, arg401_1, arg404_1


@oracle_impl(hardware="H100", shapes="(T([128, 1536, 7, 7], f32), T([128, 1536, 7, 7], f32), T([128, 1536, 1, 1], f32))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full compiled computation scope for Repro.forward."""
    getitem, arg401_1, arg404_1 = _validate_inputs(inputs)
    if getitem.device.type != "cuda":
        return _torch_oracle(inputs)
    if triton is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    out = torch.empty_strided((C,), (1,), device=getitem.device, dtype=torch.float32)
    _zero_output_kernel[(triton.cdiv(C, ZERO_BLOCK),)](
        out,
        C_=C,
        BLOCK_C_=ZERO_BLOCK,
        num_warps=1,
    )
    _spatial_sigmoid_atomic_kernel[(triton.cdiv(C, BLOCK_C), triton.cdiv(N, BLOCK_BATCH))](
        getitem,
        arg401_1,
        arg404_1,
        out,
        C_=C,
        HW_=HW,
        BLOCK_HW_=BLOCK_HW,
        BLOCK_C_=BLOCK_C,
        BLOCK_BATCH_=BLOCK_BATCH,
        num_warps=1,
        num_stages=1,
    )
    return out


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
