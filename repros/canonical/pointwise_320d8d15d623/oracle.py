"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 attention-head layout materialization by writing the fresh contiguous `[B, H, S, D]` clone directly from the contiguous `[B*S, H*D]` projection input, whereas Inductor lowers the captured view/view/permute/clone graph through generic layout materialization; Inductor cannot do this today because its pointwise/layout scheduler does not recognize this transformer head-split transpose clone as a dedicated layout-copy template with shape-specialized affine indexing; the fix is NEW_PATTERN: add a guarded attention-head layout materialization lowering for `view(B,S,H,D).permute(0,2,1,3).contiguous()` that preserves the exact output layout and dtype."""

import torch
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as _get_raw_stream
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.hints import DeviceProperties

from oracle_harness import oracle_impl

triton_helpers.set_driver_to_gpu()


def _device_properties():
    return DeviceProperties.create(torch.device("cuda", torch.cuda.current_device()))


@triton_heuristics.pointwise(
    size_hints={"x": 16777216},
    filename=__file__,
    triton_meta={
        "signature": {
            "input_ptr": "*bf16",
            "output_ptr": "*bf16",
            "xnumel": "i32",
            "XBLOCK": "constexpr",
        },
        "device": _device_properties(),
        "constants": {},
        "native_matmul": False,
        "enable_fp_fusion": True,
        "launch_pdl": False,
        "disable_ftz": False,
        "configs": [
            {
                (0,): [["tt.divisibility", 16]],
                (1,): [["tt.divisibility", 16]],
                (2,): [["tt.divisibility", 16]],
            }
        ],
    },
    inductor_meta={
        "grid_type": "Grid1D",
        "kernel_name": "_head_layout_output_linear_autotuned",
        "no_x_dim": False,
        "atomic_add_found": False,
        "num_load": 1,
        "num_store": 1,
        "num_reduction": 0,
        "autotune_hints": set(),
        "tiling_scores": {"x": 75497472},
        "assert_indirect_indexing": True,
        "autotune_local_cache": True,
        "autotune_pointwise": True,
        "autotune_remote_cache": None,
        "force_disable_caches": False,
        "dynamic_scale_rblock": True,
        "incremental_autotune": False,
        "max_autotune": False,
        "max_autotune_pointwise": False,
        "min_split_scan_rblock": 256,
        "spill_threshold": 16,
        "store_cubin": False,
        "deterministic": False,
        "batch_invariant": False,
        "force_filter_reduction_configs": False,
        "mix_order_reduction_allow_multi_stages": True,
        "dynamic_disable_pipelining": True,
        "are_deterministic_algorithms_enabled": False,
    },
    min_elem_per_thread=0,
)
@triton.jit
def _head_layout_output_linear_autotuned(
    input_ptr,
    output_ptr,
    xnumel,
    XBLOCK: tl.constexpr,
):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    x0 = xindex % 64
    x1 = (xindex // 64) % 512
    x2 = (xindex // 32768) % 12
    x3 = xindex // 393216
    values = tl.load(input_ptr + (x0 + 64 * x2 + 768 * x1 + 393216 * x3), None).to(tl.float32)
    tl.store(output_ptr + xindex, values, None)


@triton.jit
def _head_layout_packed_pair_kernel(
    input_ptr,
    output_ptr,
    SEQ: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM_PAIRS: tl.constexpr,
    HIDDEN_PAIRS: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    input_u32 = input_ptr.to(tl.pointer_type(tl.uint32))
    output_u32 = output_ptr.to(tl.pointer_type(tl.uint32))
    pair_index = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)

    dim_pair = pair_index % HEAD_DIM_PAIRS
    seq = (pair_index // HEAD_DIM_PAIRS) % SEQ
    head = (pair_index // (HEAD_DIM_PAIRS * SEQ)) % HEADS
    batch = pair_index // (HEADS * SEQ * HEAD_DIM_PAIRS)
    input_offsets = dim_pair + HEAD_DIM_PAIRS * head + HIDDEN_PAIRS * seq + SEQ * HIDDEN_PAIRS * batch
    values = tl.load(input_u32 + input_offsets)
    tl.store(output_u32 + pair_index, values)


@triton.jit
def _head_layout_2d_kernel(
    input_ptr,
    output_ptr,
    SEQ: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_head = tl.program_id(0)
    seq_offsets = tl.program_id(1) * BLOCK_S + tl.arange(0, BLOCK_S)[:, None]
    dim_offsets = tl.arange(0, BLOCK_D)[None, :]

    batch = batch_head // HEADS
    head = batch_head - batch * HEADS
    mask = (seq_offsets < SEQ) & (dim_offsets < HEAD_DIM)

    input_offsets = ((batch * SEQ + seq_offsets) * HEADS + head) * HEAD_DIM + dim_offsets
    output_offsets = (batch_head * SEQ + seq_offsets) * HEAD_DIM + dim_offsets
    values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    tl.store(output_ptr + output_offsets, values, mask=mask)


def _resolve_shape(shape, numel):
    dims = [int(dim) for dim in shape]
    if -1 in dims:
        known = 1
        for dim in dims:
            if dim != -1:
                known *= dim
        dims[dims.index(-1)] = numel // known
    return tuple(dims)


@oracle_impl(
    hardware="B200",
    point="d20f46e2",
    USE_PACKED=False,
    USE_AUTOTUNED=True,
    BLOCK_S=16,
    BLOCK_D=64,
    XBLOCK=1024,
    num_warps=4,
    num_stages=1,
)
@oracle_impl(
    hardware="B200",
    point="ad7b2a2c",
    USE_PACKED=True,
    USE_AUTOTUNED=False,
    BLOCK_S=16,
    BLOCK_D=64,
    XBLOCK=1024,
    num_warps=4,
    num_stages=3,
)
@oracle_impl(
    hardware="B200",
    point="da5fdead",
    USE_PACKED=False,
    USE_AUTOTUNED=False,
    BLOCK_S=16,
    BLOCK_D=64,
    XBLOCK=1024,
    num_warps=4,
    num_stages=3,
)
def oracle_forward(
    inputs,
    *,
    USE_PACKED: bool,
    USE_AUTOTUNED: bool,
    BLOCK_S: int,
    BLOCK_D: int,
    XBLOCK: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, _shape_param_0, _shape_param_1 = inputs

    batch, seq, _hidden = (int(dim) for dim in _shape_param_0)
    _, _, heads, head_dim = _resolve_shape(_shape_param_1, arg0_1.numel())

    output = torch.empty_strided(
        (batch, heads, seq, head_dim),
        (heads * seq * head_dim, seq * head_dim, head_dim, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    if USE_PACKED:
        grid_packed = (triton.cdiv(arg0_1.numel() // 2, XBLOCK),)
        _head_layout_packed_pair_kernel[grid_packed](
            arg0_1,
            output,
            SEQ=seq,
            HEADS=heads,
            HEAD_DIM_PAIRS=head_dim // 2,
            HIDDEN_PAIRS=_hidden // 2,
            XBLOCK=XBLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return output

    if USE_AUTOTUNED:
        _head_layout_output_linear_autotuned.run(
            arg0_1,
            output,
            arg0_1.numel(),
            stream=_get_raw_stream(arg0_1.get_device()),
        )
        return output

    grid_2d = (batch * heads, triton.cdiv(seq, BLOCK_S))
    _head_layout_2d_kernel[grid_2d](
        arg0_1,
        output,
        SEQ=seq,
        HEADS=heads,
        HEAD_DIM=head_dim,
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output
