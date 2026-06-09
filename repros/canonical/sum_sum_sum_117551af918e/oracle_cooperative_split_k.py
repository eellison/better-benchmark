"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full DistillGPT2 layer-norm-backward/dropout-mask return tuple by row-tiling the `[16384, 768]` producer, reducing each row's hidden-dimension summaries for the masked gradient path, and cooperatively accumulating all three returned `[768]` column sums from the same row-tiled pass, whereas Inductor currently emits separate generic reductions and pointwise kernels for the row sums, sibling `sum([0, 1])` reductions, mask conversion/multiplication, and final masked-gradient column sum over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that combines row-local reductions feeding a dependent column reduction with sibling column reductions in one coordinated producer; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible layer-norm-backward reductions across row tiles, keep row-local summaries in the producer, and finalize multiple column accumulators including dependent masked-gradient sums."""
from __future__ import annotations

import argparse
import sys

import torch
from pathlib import Path

from repro import Repro, make_inputs as repro_make_inputs

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - keeps CPU-only syntax checks usable.
    triton = None
    tl = None


REPRO_ID = "sum_sum_sum_117551af918e"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

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
SHAPE_LABEL = "hf_distillgpt2_train_003_72fdc3f1"

BATCH = 32
SEQ = 512
M = BATCH * SEQ
D = 768

TILE_M = 4
TILE_D = 1024
ZERO_BLOCK_D = 1024
ROW_NUM_WARPS = 2
ZERO_NUM_WARPS = 1


def make_inputs(device: torch.device) -> tuple[object, ...]:
    moved: list[object] = []
    for value in repro_make_inputs():
        if isinstance(value, torch.Tensor):
            moved.append(value.to(device=device))
        else:
            moved.append(value)
    return tuple(moved)


def _as_tuple(out: object) -> tuple[torch.Tensor, ...]:
    if isinstance(out, tuple):
        return out
    return (out,)


def reference_outputs(inputs: tuple[object, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    model = Repro().to(device)
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def oracle_torch(
    mm_1: torch.Tensor,
    arg150_1: torch.Tensor,
    arg38_1: torch.Tensor,
    arg117_1: torch.Tensor,
    arg119_1: torch.Tensor,
    arg116_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    add = arg150_1 + mm_1.view(_shape_param_0)
    weighted = add * arg38_1
    row_sum = weighted.sum(dim=2, keepdim=True)
    row_dot = (weighted * arg117_1).sum(dim=2, keepdim=True)
    grad = arg119_1 * (weighted * D - row_sum - arg117_1 * row_dot)

    out0 = (add * arg117_1).sum(dim=(0, 1))
    out1 = add.sum(dim=(0, 1))
    masked = grad * arg116_1.to(torch.float32)
    out2 = masked.view(_shape_param_1).sum(dim=0, keepdim=True).view(_shape_param_2)
    return out0, out1, out2


if triton is not None:

    @triton.jit
    def _zero_outputs_kernel(
        out_add_xhat_ptr,
        out_add_ptr,
        out_masked_grad_ptr,
        D_: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        d = tl.program_id(0) * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d < D_
        zero = tl.zeros((BLOCK_D,), dtype=tl.float32)
        tl.store(out_add_xhat_ptr + d, zero, mask=d_mask)
        tl.store(out_add_ptr + d, zero, mask=d_mask)
        tl.store(out_masked_grad_ptr + d, zero, mask=d_mask)


    @triton.jit
    def _row_atomic_reduce_kernel(
        mm_ptr,
        residual_ptr,
        gamma_ptr,
        xhat_ptr,
        scale_ptr,
        mask_ptr,
        out_add_xhat_ptr,
        out_add_ptr,
        out_masked_grad_ptr,
        M_: tl.constexpr,
        D_: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        tile_m = tl.program_id(0)
        m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        d = tl.arange(0, BLOCK_D)
        m_mask = m < M_
        d_mask = d < D_
        active = m_mask[:, None] & d_mask[None, :]
        offsets = m[:, None] * D_ + d[None, :]

        mm = tl.load(mm_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        xhat = tl.load(xhat_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + m, mask=m_mask, other=0.0).to(tl.float32)
        keep = tl.load(mask_ptr + offsets, mask=active, other=0).to(tl.float32)

        add = residual + mm
        weighted = add * gamma[None, :]
        row_sum = tl.sum(tl.where(active, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(active, weighted * xhat, 0.0), axis=1)
        grad = scale[:, None] * (
            weighted * D_ - row_sum[:, None] - xhat * row_dot[:, None]
        )

        sum_add_xhat = tl.sum(tl.where(active, add * xhat, 0.0), axis=0)
        sum_add = tl.sum(tl.where(active, add, 0.0), axis=0)
        sum_masked_grad = tl.sum(tl.where(active, grad * keep, 0.0), axis=0)

        tl.atomic_add(out_add_xhat_ptr + d, sum_add_xhat, sem="relaxed", mask=d_mask)
        tl.atomic_add(out_add_ptr + d, sum_add, sem="relaxed", mask=d_mask)
        tl.atomic_add(
            out_masked_grad_ptr + d,
            sum_masked_grad,
            sem="relaxed",
            mask=d_mask,
        )


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        mm_1,
        arg150_1,
        arg38_1,
        arg117_1,
        arg119_1,
        arg116_1,
        *_shape_params,
    ) = inputs

    return (
        mm_1.reshape(M, D).contiguous(),
        arg150_1.reshape(M, D).contiguous(),
        arg38_1.contiguous(),
        arg117_1.reshape(M, D).contiguous(),
        arg119_1.reshape(M).contiguous(),
        arg116_1.reshape(M, D).contiguous(),
    )


def oracle_triton_prepared(
    mm_md: torch.Tensor,
    residual_md: torch.Tensor,
    gamma_d: torch.Tensor,
    xhat_md: torch.Tensor,
    scale_m: torch.Tensor,
    mask_md: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is not available")
    if mm_md.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    assert mm_md.shape == (M, D)
    assert residual_md.shape == (M, D)
    assert gamma_d.shape == (D,)
    assert xhat_md.shape == (M, D)
    assert scale_m.shape == (M,)
    assert mask_md.shape == (M, D)
    assert mm_md.is_contiguous()
    assert residual_md.is_contiguous()
    assert gamma_d.is_contiguous()
    assert xhat_md.is_contiguous()
    assert scale_m.is_contiguous()
    assert mask_md.is_contiguous()

    device = mm_md.device
    out_add_xhat = torch.empty((D,), device=device, dtype=torch.float32)
    out_add = torch.empty((D,), device=device, dtype=torch.float32)
    out_masked_grad = torch.empty((D,), device=device, dtype=torch.float32)
    num_m_tiles = triton.cdiv(M, TILE_M)

    _zero_outputs_kernel[(triton.cdiv(D, ZERO_BLOCK_D),)](
        out_add_xhat,
        out_add,
        out_masked_grad,
        D_=D,
        BLOCK_D=ZERO_BLOCK_D,
        num_warps=ZERO_NUM_WARPS,
    )
    _row_atomic_reduce_kernel[(num_m_tiles,)](
        mm_md,
        residual_md,
        gamma_d,
        xhat_md,
        scale_m,
        mask_md,
        out_add_xhat,
        out_add,
        out_masked_grad,
        M_=M,
        D_=D,
        BLOCK_M=TILE_M,
        BLOCK_D=TILE_D,
        num_warps=ROW_NUM_WARPS,
    )

    return out_add_xhat, out_add, out_masked_grad


def oracle_triton(*inputs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return oracle_triton_prepared(*prepare_oracle_inputs(*inputs))


def oracle_full(
    *inputs: object,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    first_tensor = next(value for value in inputs if isinstance(value, torch.Tensor))
    if impl == "auto":
        impl = "triton" if first_tensor.device.type == "cuda" and triton is not None else "torch"
    if impl == "triton":
        return oracle_triton(*inputs)
    if impl == "torch":
        return oracle_torch(*inputs)
    raise ValueError(f"unknown impl: {impl}")


@oracle_impl(hardware="H100", shapes="(T([16384, 768], f32), T([32, 512, 768], f32), T([768], f32), T([32, 512, 768], f32), T([32, 512, 1], f32), T([32, 512, 768], b8), S([32, 512, 768]), S([16384, 768]), S([768]))")
def oracle_forward(inputs):
    """Thin wrapper for oracle_harness compatibility."""
    return oracle_full(*inputs)


def main() -> None:
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

    inputs = _harness_get_inputs(REPRO_DIR)
    instance = _harness_get_repro_instance(REPRO_DIR)

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
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
