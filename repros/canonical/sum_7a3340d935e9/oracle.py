"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete f32 `arg0 * arg1` producer and `sum([0, 1])` output with a shape-specialized Triton reduction that either cooperatively splits the row dimension through compact per-column partials or uses a looped column program when that schedule wins for the point, whereas Inductor lowers the same producer-reduction through a generic reduction schedule; Inductor cannot do this today because its scheduler/codegen does not autotune this product-column reduction across split-K and looped schedules for the fixed transformer row/hidden sizes; the fix is COOPERATIVE_SPLIT_K: add a guarded product-column reduction lowering that can choose the row-split partial strategy per shape while preserving f32 multiply and accumulation semantics."""

import torch
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.hints import DeviceProperties, ReductionHint

from oracle_harness import oracle_impl

triton_helpers.set_driver_to_gpu()


def _device_properties():
    return DeviceProperties.create(torch.device("cuda", torch.cuda.current_device()))


@triton_heuristics.reduction(
    size_hints={"x": 65536, "r0_": 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={
        "signature": {
            "x_ptr": "*fp32",
            "y_ptr": "*fp32",
            "partials_ptr": "*fp32",
            "xnumel": "i32",
            "r0_numel": "i32",
            "XBLOCK": "constexpr",
            "R0_BLOCK": "constexpr",
        },
        "device": _device_properties(),
        "constants": {},
        "configs": [
            {
                (0,): [["tt.divisibility", 16]],
                (1,): [["tt.divisibility", 16]],
                (2,): [["tt.divisibility", 16]],
                (3,): [["tt.divisibility", 16]],
                (4,): [["tt.divisibility", 16]],
            }
        ],
    },
    inductor_meta={
        "grid_type": "Grid1D",
        "kernel_name": "_c57_mul_sum_partial",
        "mutated_arg_names": [],
        "optimize_mem": True,
        "no_x_dim": False,
        "num_load": 2,
        "num_reduction": 1,
        "autotune_hints": set(),
        "autotune_local_cache": True,
        "autotune_pointwise": True,
        "force_disable_caches": False,
        "dynamic_scale_rblock": True,
        "coordinate_descent_tuning": True,
        "coordinate_descent_search_radius": 1,
        "coordinate_descent_check_all_directions": False,
        "min_split_scan_rblock": 256,
        "spill_threshold": 16,
    },
)
@triton.jit
def _c57_mul_sum_partial(
    x_ptr,
    y_ptr,
    partials_ptr,
    xnumel,
    r0_numel,
    XBLOCK: tl.constexpr,
    R0_BLOCK: tl.constexpr,
):
    xnumel = 65536
    r0_numel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex % 1024
    x1 = xindex // 1024
    acc = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)

    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        offsets = x0 + 1024 * r0_index + 131072 * x1
        x = tl.load(x_ptr + offsets, r0_mask, eviction_policy="evict_first", other=0.0)
        y = tl.load(y_ptr + offsets, r0_mask, eviction_policy="evict_first", other=0.0)
        acc += tl.where(r0_mask & xmask, x * y, 0.0)

    tl.store(partials_ptr + xindex, tl.sum(acc, 1)[:, None], None)


@triton_heuristics.persistent_reduction(
    size_hints={"x": 1024, "r0_": 64},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={
        "signature": {
            "partials_ptr": "*fp32",
            "out_ptr": "*fp32",
            "xnumel": "i32",
            "r0_numel": "i32",
            "XBLOCK": "constexpr",
        },
        "device": _device_properties(),
        "constants": {},
        "configs": [
            {
                (0,): [["tt.divisibility", 16]],
                (1,): [["tt.divisibility", 16]],
                (2,): [["tt.divisibility", 16]],
                (3,): [["tt.divisibility", 16]],
            }
        ],
    },
    inductor_meta={
        "grid_type": "Grid1D",
        "kernel_name": "_c57_finalize_partials",
        "mutated_arg_names": [],
        "optimize_mem": True,
        "no_x_dim": False,
        "num_load": 1,
        "num_reduction": 1,
        "autotune_hints": set(),
        "autotune_local_cache": True,
        "autotune_pointwise": True,
        "force_disable_caches": False,
        "dynamic_scale_rblock": True,
        "coordinate_descent_tuning": True,
        "coordinate_descent_search_radius": 1,
        "coordinate_descent_check_all_directions": False,
        "min_split_scan_rblock": 256,
        "spill_threshold": 16,
    },
)
@triton.jit
def _c57_finalize_partials(
    partials_ptr,
    out_ptr,
    xnumel,
    r0_numel,
    XBLOCK: tl.constexpr,
):
    xnumel = 1024
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    values = tl.load(partials_ptr + xindex + 1024 * r0_index, xmask, other=0.0)
    values = tl.where(xmask, values, 0.0)
    tl.store(out_ptr + xindex, tl.sum(values, 1)[:, None], xmask)


@triton.jit
def _split_partial_kernel(
    x_ptr,
    y_ptr,
    partials_ptr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    TOTAL_PARTIALS: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    r_offsets = tl.arange(0, RBLOCK)[None, :]
    cols = xindex % COLS
    row_block = xindex // COLS
    rows = row_block * ROW_BLOCK + r_offsets
    offsets = rows * COLS + cols

    x = tl.load(x_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
    y = tl.load(y_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
    partial = tl.sum(x * y, axis=1)[:, None]
    tl.store(partials_ptr + xindex, partial)


@triton.jit
def _finalize_partials_kernel(
    partials_ptr,
    out_ptr,
    NUM_ROW_BLOCKS: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK_ROW_BLOCKS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)[:, None]
    row_blocks = tl.arange(0, BLOCK_ROW_BLOCKS)[None, :]
    values = tl.load(partials_ptr + row_blocks * COLS + cols).to(tl.float32)
    out = tl.sum(values, axis=1)[:, None]
    tl.store(out_ptr + cols, out)


@triton.jit
def _looped_column_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)[None, :]
    r_offsets = tl.arange(0, RBLOCK)[:, None]
    col_mask = cols < COLS
    acc = tl.zeros((RBLOCK, BLOCK_COLS), tl.float32)

    for row_base in tl.range(0, ROWS, RBLOCK):
        rows = row_base + r_offsets
        mask = (rows < ROWS) & col_mask
        offsets = rows * COLS + cols
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += tl.where(mask, x * y, 0.0)

    total = tl.sum(acc, axis=0)
    out_cols = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    tl.store(out_ptr + out_cols, total, mask=out_cols < COLS)


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


# c57eca43: (T([8,1024,1024], f32), T([8,1024,1024], f32))
@oracle_impl(hardware="B200", point="c57eca43", MODE=2, ROW_BLOCK=128, XBLOCK=128, RBLOCK=128, FINAL_BLOCK_COLS=16, BLOCK_COLS=8, LOOP_RBLOCK=256, split_warps=8, final_warps=4, loop_warps=8)
# b79fa38b: (T([8,1024,768], f32), T([8,1024,768], f32))
@oracle_impl(hardware="B200", point="b79fa38b", MODE=0, ROW_BLOCK=128, XBLOCK=128, RBLOCK=128, FINAL_BLOCK_COLS=16, BLOCK_COLS=8, LOOP_RBLOCK=256, split_warps=4, final_warps=4, loop_warps=8)
# fc97d3ab: (T([16,1024,768], f32), T([16,1024,768], f32))
@oracle_impl(hardware="B200", point="fc97d3ab", MODE=0, ROW_BLOCK=128, XBLOCK=128, RBLOCK=128, FINAL_BLOCK_COLS=16, BLOCK_COLS=8, LOOP_RBLOCK=256, split_warps=4, final_warps=4, loop_warps=8)
# f4eac48c: (T([64,256,1024], f32), T([64,256,1024], f32))
@oracle_impl(hardware="B200", point="f4eac48c", MODE=0, ROW_BLOCK=128, XBLOCK=128, RBLOCK=128, FINAL_BLOCK_COLS=16, BLOCK_COLS=8, LOOP_RBLOCK=256, split_warps=4, final_warps=4, loop_warps=8)
def oracle_forward(
    inputs,
    *,
    MODE: int,
    ROW_BLOCK: int,
    XBLOCK: int,
    RBLOCK: int,
    FINAL_BLOCK_COLS: int,
    BLOCK_COLS: int,
    LOOP_RBLOCK: int,
    split_warps: int,
    final_warps: int,
    loop_warps: int,
):
    x, y = inputs
    rows = int(x.shape[0]) * int(x.shape[1])
    cols = int(x.shape[2])
    out = torch.empty_strided((cols,), (1,), device=x.device, dtype=torch.float32)

    if MODE == 2:
        partials = torch.empty_strided(
            (1024, 64),
            (1, 1024),
            device=x.device,
            dtype=torch.float32,
        )
        stream = get_raw_stream(x.get_device())
        _c57_mul_sum_partial.run(
            x,
            y,
            partials,
            65536,
            128,
            stream=stream,
        )
        _c57_finalize_partials.run(
            partials,
            out,
            1024,
            64,
            stream=stream,
        )
        return out

    if MODE == 1:
        _looped_column_kernel[(triton.cdiv(cols, BLOCK_COLS),)](
            x,
            y,
            out,
            ROWS=rows,
            COLS=cols,
            BLOCK_COLS=BLOCK_COLS,
            RBLOCK=LOOP_RBLOCK,
            num_warps=loop_warps,
            num_stages=1,
        )
        return out

    num_row_blocks = triton.cdiv(rows, ROW_BLOCK)
    total_partials = num_row_blocks * cols
    partials = torch.empty_strided(
        (num_row_blocks, cols),
        (cols, 1),
        device=x.device,
        dtype=torch.float32,
    )
    _split_partial_kernel[(triton.cdiv(total_partials, XBLOCK),)](
        x,
        y,
        partials,
        ROWS=rows,
        COLS=cols,
        TOTAL_PARTIALS=total_partials,
        ROW_BLOCK=ROW_BLOCK,
        XBLOCK=XBLOCK,
        RBLOCK=RBLOCK,
        num_warps=split_warps,
        num_stages=1,
    )
    _finalize_partials_kernel[(triton.cdiv(cols, FINAL_BLOCK_COLS),)](
        partials,
        out,
        NUM_ROW_BLOCKS=num_row_blocks,
        COLS=cols,
        BLOCK_ROW_BLOCKS=_next_power_of_2(num_row_blocks),
        BLOCK_COLS=FINAL_BLOCK_COLS,
        num_warps=final_warps,
        num_stages=3,
    )
    return out
