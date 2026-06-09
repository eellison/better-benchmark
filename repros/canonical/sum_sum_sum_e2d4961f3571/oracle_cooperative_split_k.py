"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full `sum_sum_sum_e2d4961f3571` MobileBERT layer-norm-backward return tuple by reducing each `[512]` row once, storing the returned transposed `[512,32768]` masked-gradient view, and cooperatively accumulating the three returned `[512]` column reductions from the same row-tiled producer, whereas Inductor currently schedules the row-local reductions, the masked-gradient/transpose materialization, and the sibling `sum([0, 1])`/`sum([0])` column reductions as separate generic pointwise and reduction kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that keeps row-local reductions, a dependent full-tensor side output, and multiple sibling column accumulators in one coordinated producer; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible layer-norm-backward column reductions across row tiles, finalize their partials, and fuse the dependent masked-gradient side store returned as a transpose view."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - keeps syntax checks usable without Triton.
    triton = None
    tl = None


REPRO_ID = "sum_sum_sum_e2d4961f3571"

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
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "hf_mobilebertformaskedlm_train_014_3ba71d70"

BATCH = 256
SEQ = 128
M = BATCH * SEQ
D = 512
INV_D = 1.0 / D

TILE_M = 8
TILE_D = 512
FINAL_BLOCK_D = 16
FINAL_BLOCK_TILES = 1024


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _as_tuple(out: object) -> tuple[torch.Tensor, ...]:
    if isinstance(out, tuple):
        return out
    return (out,)


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(value.cuda() if isinstance(value, torch.Tensor) else value for value in module.make_inputs())


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def oracle_torch(
    mm_1: torch.Tensor,
    arg581_1: torch.Tensor,
    arg1119_1: torch.Tensor,
    arg1120_1: torch.Tensor,
    arg1121_1: torch.Tensor,
    full_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = mm_1.view(_shape_param_0)
    weighted = x * arg581_1
    relu = torch.relu(arg1119_1.view(_shape_param_1))
    centered = relu - arg1120_1
    scaled_centered = centered * arg1121_1

    row_sum = weighted.sum(dim=2, keepdim=True)
    row_dot = (weighted * scaled_centered).sum(dim=2, keepdim=True)
    masked_grad = (arg1121_1 * INV_D) * (
        weighted * D - row_sum - scaled_centered * row_dot
    )
    masked_grad = torch.where(relu <= 0, full_1, masked_grad)

    out0 = (x * scaled_centered).sum(dim=(0, 1))
    out1 = x.sum(dim=(0, 1))
    out2 = masked_grad.view(_shape_param_2).permute(1, 0)
    out3 = masked_grad.view(_shape_param_2).sum(dim=0, keepdim=True).view(_shape_param_3)
    return out0, out1, out2, out3


if triton is not None:

    @triton.jit
    def _row_tile_kernel(
        x_ptr,
        gamma_ptr,
        relu_input_ptr,
        center_ptr,
        scale_ptr,
        full_ptr,
        partial_x_scaled_ptr,
        partial_x_ptr,
        partial_masked_grad_ptr,
        masked_grad_ptr,
        M_: tl.constexpr,
        D_: tl.constexpr,
        INV_D_: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        tile_m = tl.program_id(0)
        m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        d = tl.arange(0, BLOCK_D)
        m_mask = m < M_
        d_mask = d < D_
        mask = m_mask[:, None] & d_mask[None, :]
        offsets = m[:, None] * D_ + d[None, :]

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
        relu_input = tl.load(relu_input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        center = tl.load(center_ptr + m, mask=m_mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + m, mask=m_mask, other=0.0).to(tl.float32)
        full = tl.load(full_ptr).to(tl.float32)

        relu = tl.maximum(relu_input, 0.0)
        scaled_centered = (relu - center[:, None]) * scale[:, None]
        weighted = x * gamma[None, :]
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(mask, weighted * scaled_centered, 0.0), axis=1)
        grad = scale[:, None] * INV_D_ * (
            weighted * D_ - row_sum[:, None] - scaled_centered * row_dot[:, None]
        )
        masked_grad = tl.where(relu <= 0.0, full, grad)

        tl.store(masked_grad_ptr + offsets, masked_grad, mask=mask)

        partial_offsets = tile_m * D_ + d
        sum_x_scaled = tl.sum(tl.where(mask, x * scaled_centered, 0.0), axis=0)
        sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
        sum_masked_grad = tl.sum(tl.where(mask, masked_grad, 0.0), axis=0)
        tl.store(partial_x_scaled_ptr + partial_offsets, sum_x_scaled, mask=d_mask)
        tl.store(partial_x_ptr + partial_offsets, sum_x, mask=d_mask)
        tl.store(partial_masked_grad_ptr + partial_offsets, sum_masked_grad, mask=d_mask)


    @triton.jit
    def _finalize_column_sums_kernel(
        partial_x_scaled_ptr,
        partial_x_ptr,
        partial_masked_grad_ptr,
        out_x_scaled_ptr,
        out_x_ptr,
        out_masked_grad_ptr,
        NUM_M_TILES: tl.constexpr,
        D_: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        d = tl.program_id(0) * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d < D_
        tile = tl.arange(0, BLOCK_TILES)

        acc_x_scaled = tl.zeros((BLOCK_D,), tl.float32)
        acc_x = tl.zeros((BLOCK_D,), tl.float32)
        acc_masked_grad = tl.zeros((BLOCK_D,), tl.float32)
        for tile_base in range(0, NUM_M_TILES, BLOCK_TILES):
            tile_ids = tile_base + tile
            mask = (tile_ids[:, None] < NUM_M_TILES) & d_mask[None, :]
            offsets = tile_ids[:, None] * D_ + d[None, :]
            acc_x_scaled += tl.sum(
                tl.load(partial_x_scaled_ptr + offsets, mask=mask, other=0.0),
                axis=0,
            )
            acc_x += tl.sum(
                tl.load(partial_x_ptr + offsets, mask=mask, other=0.0),
                axis=0,
            )
            acc_masked_grad += tl.sum(
                tl.load(partial_masked_grad_ptr + offsets, mask=mask, other=0.0),
                axis=0,
            )

        tl.store(out_x_scaled_ptr + d, acc_x_scaled, mask=d_mask)
        tl.store(out_x_ptr + d, acc_x, mask=d_mask)
        tl.store(out_masked_grad_ptr + d, acc_masked_grad, mask=d_mask)


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        mm_1,
        arg581_1,
        arg1119_1,
        arg1120_1,
        arg1121_1,
        full_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
    ) = inputs

    return (
        mm_1.reshape(M, D).contiguous(),
        arg581_1.contiguous(),
        arg1119_1.reshape(M, D).contiguous(),
        arg1120_1.reshape(M).contiguous(),
        arg1121_1.reshape(M).contiguous(),
        full_1.contiguous(),
    )


def oracle_triton_prepared(
    x_md: torch.Tensor,
    gamma_d: torch.Tensor,
    relu_input_md: torch.Tensor,
    center_m: torch.Tensor,
    scale_m: torch.Tensor,
    full_scalar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is not available")
    if x_md.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    assert x_md.shape == (M, D)
    assert gamma_d.shape == (D,)
    assert relu_input_md.shape == (M, D)
    assert center_m.shape == (M,)
    assert scale_m.shape == (M,)
    assert full_scalar.shape == ()
    assert x_md.is_contiguous()
    assert gamma_d.is_contiguous()
    assert relu_input_md.is_contiguous()
    assert center_m.is_contiguous()
    assert scale_m.is_contiguous()
    assert full_scalar.is_contiguous()

    device = x_md.device
    num_m_tiles = triton.cdiv(M, TILE_M)
    partial_x_scaled = torch.empty((num_m_tiles, D), device=device, dtype=torch.float32)
    partial_x = torch.empty((num_m_tiles, D), device=device, dtype=torch.float32)
    partial_masked_grad = torch.empty((num_m_tiles, D), device=device, dtype=torch.float32)
    masked_grad_md = torch.empty((M, D), device=device, dtype=torch.float32)

    _row_tile_kernel[(num_m_tiles,)](
        x_md,
        gamma_d,
        relu_input_md,
        center_m,
        scale_m,
        full_scalar,
        partial_x_scaled,
        partial_x,
        partial_masked_grad,
        masked_grad_md,
        M_=M,
        D_=D,
        INV_D_=INV_D,
        BLOCK_M=TILE_M,
        BLOCK_D=TILE_D,
        num_warps=8,
    )

    out_x_scaled = torch.empty((D,), device=device, dtype=torch.float32)
    out_x = torch.empty((D,), device=device, dtype=torch.float32)
    out_masked_grad = torch.empty((D,), device=device, dtype=torch.float32)
    _finalize_column_sums_kernel[(triton.cdiv(D, FINAL_BLOCK_D),)](
        partial_x_scaled,
        partial_x,
        partial_masked_grad,
        out_x_scaled,
        out_x,
        out_masked_grad,
        NUM_M_TILES=num_m_tiles,
        D_=D,
        BLOCK_TILES=FINAL_BLOCK_TILES,
        BLOCK_D=FINAL_BLOCK_D,
        num_warps=8,
    )

    return out_x_scaled, out_x, masked_grad_md.t(), out_masked_grad


def oracle_triton(*inputs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return oracle_triton_prepared(*prepare_oracle_inputs(*inputs))


def oracle_full(
    *inputs: object,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    first_tensor = next(value for value in inputs if isinstance(value, torch.Tensor))
    if impl == "auto":
        impl = "triton" if first_tensor.device.type == "cuda" and triton is not None else "torch"
    if impl == "triton":
        return oracle_triton(*inputs)
    if impl == "torch":
        return oracle_torch(*inputs)
    raise ValueError(f"unknown impl: {impl}")


@oracle_impl(hardware="H100", shapes="(T([32768, 512], f32), T([512], f32), T([32768, 512], f32), T([256, 128, 1], f32), T([256, 128, 1], f32), T([], f32), S([256, 128, 512]), S([256, 128, 512]), S([32768, 512]), S([512]))")
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
