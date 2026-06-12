"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Longformer bf16 inference sliding-window attention scope, including direct assembly of the skewed local score band from the `[288,512,512]` chunked score tensor, key-mask insertion, first/last-window invalid-key masking, fp32 stable natural-exp softmax, query-mask zeroing, final bf16 rounding, and the returned strided `[384,256,768]` pad/slice/view layout; Inductor lowers the captured graph as many generic pad/view/slice_scatter/permute/clone kernels around a row softmax and then schedules the final layout pipeline separately; Inductor cannot do this today because scheduler/codegen does not recognize the Longformer diagonal-skew band plus mask construction as the producer of the softmax reduction with a destination-layout scatter epilogue; the fix is NEW_PATTERN: add a Longformer sliding-window attention lowering that canonicalizes the structured band assembly and emits one fused softmax/layout kernel for this output surface."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 8
SEQ_LEN = 1024
N_HEADS = 12
RNUMEL = 513
LOCAL_CHUNK = 256
CHUNKS = 4
BMM_CHUNKS = 3
PADDED_RNUMEL = 770
FINAL_INNER = 769
OUT_D = 768
OUT_SHAPE = (BATCH * N_HEADS * CHUNKS, LOCAL_CHUNK, OUT_D)
OUT_STRIDE = (LOCAL_CHUNK * PADDED_RNUMEL, FINAL_INNER, 1)
OUT_STORAGE_SIZE = (
    (OUT_SHAPE[0] - 1) * OUT_STRIDE[0]
    + (OUT_SHAPE[1] - 1) * OUT_STRIDE[1]
    + (OUT_SHAPE[2] - 1) * OUT_STRIDE[2]
    + 1
)
ROWS = BATCH * SEQ_LEN * N_HEADS


@triton.jit
def _zero_kernel(out_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offsets, tl.zeros((BLOCK,), tl.float32), mask=offsets < n_elements)


@triton.jit
def _longformer_softmax_layout_kernel(
    query_mask_ptr,
    bmm_ptr,
    key_mask_ptr,
    out_ptr,
    BLOCK_N: tl.constexpr,
    R: tl.constexpr,
    SEQ: tl.constexpr,
    HEADS: tl.constexpr,
    CHUNK: tl.constexpr,
    CHUNKS_: tl.constexpr,
    BMM_CHUNKS_: tl.constexpr,
    PADDED_R: tl.constexpr,
    FINAL_INNER_: tl.constexpr,
    OUT_D_: tl.constexpr,
    OUT_M_STRIDE: tl.constexpr,
    OUT_T_STRIDE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    valid_cols = cols < R

    head = row % HEADS
    row_div_heads = row // HEADS
    seq = row_div_heads % SEQ
    batch = row_div_heads // SEQ
    batch_head = batch * HEADS + head

    col_i32 = cols.to(tl.int32)
    key = seq + col_i32 - CHUNK
    valid_key = valid_cols & (key >= 0) & (key < SEQ)
    safe_key = tl.minimum(tl.maximum(key, 0), SEQ - 1)

    left_chunk = tl.minimum(safe_key // CHUNK, BMM_CHUNKS_ - 1)
    right_chunk = tl.minimum(seq // CHUNK, BMM_CHUNKS_ - 1)
    source_chunk = tl.where(col_i32 < CHUNK, left_chunk, right_chunk)
    source_row = tl.minimum(tl.maximum(seq - source_chunk * CHUNK, 0), 511)
    source_col = tl.minimum(tl.maximum(safe_key - source_chunk * CHUNK, 0), 511)
    bmm_offsets = (
        (batch_head * BMM_CHUNKS_ + source_chunk) * (512 * 512)
        + source_row * 512
        + source_col
    )

    local_scores = tl.load(bmm_ptr + bmm_offsets, mask=valid_key, other=0.0).to(tl.float32)
    key_mask = tl.load(key_mask_ptr + batch * SEQ + safe_key, mask=valid_key, other=0.0)
    mask_bias = tl.where(key_mask != 0.0, -3.3895313892515355e38, 0.0)
    scores = tl.where(valid_key, local_scores + mask_bias, -float("inf"))

    row_max = tl.max(scores, axis=0)
    numer = libdevice.exp(scores - row_max)
    denom = tl.sum(numer, axis=0)
    values = numer / denom

    keep_row = tl.load(query_mask_ptr + batch * SEQ + seq) == 0
    chunk_id = seq // CHUNK
    pos = seq - chunk_id * CHUNK
    out_m = batch_head * CHUNKS_ + chunk_id

    safe_cols = tl.minimum(cols, R - 1)
    padded_linear = pos * PADDED_R + safe_cols
    out_t = padded_linear // FINAL_INNER_
    out_d = padded_linear - out_t * FINAL_INNER_
    out_offsets = out_m * OUT_M_STRIDE + out_t * OUT_T_STRIDE + out_d
    store_mask = valid_cols & (out_d < OUT_D_)
    values = tl.where(keep_row, values, 0.0)
    tl.store(out_ptr + out_offsets, values, mask=store_mask)


def _launch(inputs, *, BLOCK_N: int, num_warps: int):
    query_mask, bmm, key_mask, *_shape_params = inputs
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=bmm.device,
        dtype=torch.bfloat16,
    )
    _zero_kernel[(triton.cdiv(OUT_STORAGE_SIZE, 1024),)](
        out,
        OUT_STORAGE_SIZE,
        BLOCK=1024,
        num_warps=4,
        num_stages=3,
    )
    _longformer_softmax_layout_kernel[(ROWS,)](
        query_mask,
        bmm,
        key_mask,
        out,
        BLOCK_N=BLOCK_N,
        R=RNUMEL,
        SEQ=SEQ_LEN,
        HEADS=N_HEADS,
        CHUNK=LOCAL_CHUNK,
        CHUNKS_=CHUNKS,
        BMM_CHUNKS_=BMM_CHUNKS,
        PADDED_R=PADDED_RNUMEL,
        FINAL_INNER_=FINAL_INNER,
        OUT_D_=OUT_D,
        OUT_M_STRIDE=OUT_STRIDE[0],
        OUT_T_STRIDE=OUT_STRIDE[1],
        num_warps=num_warps,
        num_stages=3,
    )
    return out


@oracle_impl(hardware="B200", point="79c25467", BLOCK_N=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int):
    return _launch(inputs, BLOCK_N=BLOCK_N, num_warps=num_warps)
