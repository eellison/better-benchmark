"""cuTile port of pointwise_6af37bbd9857 (SCHEDULER_FUSION): MobileViT
patch-padding — one all-zero bf16 [512,4,256,40] tensor and one
slice_scatter output whose first 36 columns come from a permuted view of
[512,256,4,36] and last 4 columns are zero.

The output layout dim (=40) is not a power of 2. We flatten to 1D and use
`ct.scatter` with per-lane offsets to skip the last 4 columns of each row.
BLOCK divides total numel evenly.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 512
PATCHES = 4
TOKENS = 256
IN_PATCH_WIDTH = 36
OUT_PATCH_WIDTH = 40
OUT_NUMEL = BATCH * PATCHES * TOKENS * OUT_PATCH_WIDTH


@ct.kernel
def _zero_and_scatter_kernel(
    src_ptr,          # bf16 [BATCH * TOKENS * PATCHES * IN_PATCH_WIDTH]
    zero_out_ptr,     # bf16 [OUT_NUMEL]
    scatter_out_ptr,  # bf16 [OUT_NUMEL]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int64)

    forty = ct.full(shape=(BLOCK,), fill_value=40, dtype=ct.int64)
    two_five_six = ct.full(shape=(BLOCK,), fill_value=256, dtype=ct.int64)
    four = ct.full(shape=(BLOCK,), fill_value=4, dtype=ct.int64)

    k = offsets - (offsets // forty) * forty
    tp = offsets // forty  # (batch*4*256 + patch*256 + token)
    token = tp - (tp // two_five_six) * two_five_six
    bp = tp // two_five_six  # batch*4 + patch
    patch = bp - (bp // four) * four
    batch = bp // four

    src_offsets = (batch * TOKENS + token) * (PATCHES * IN_PATCH_WIDTH) + patch * IN_PATCH_WIDTH + k
    has_source = k < 36

    zero = ct.zeros(shape=(BLOCK,), dtype=ct.bfloat16)
    ct.store(zero_out_ptr, index=(pid,), tile=zero)

    value = ct.gather(src_ptr, (src_offsets,), mask=has_source,
                      padding_value=ct.bfloat16(0))
    ct.store(scatter_out_ptr, index=(pid,), tile=value)


@oracle_impl(hardware="B200", point="51a39cb3", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    src, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    out_shape = (BATCH, PATCHES, TOKENS, OUT_PATCH_WIDTH)
    out_stride = (PATCHES * TOKENS * OUT_PATCH_WIDTH, TOKENS * OUT_PATCH_WIDTH, OUT_PATCH_WIDTH, 1)
    zero_out = torch.empty_strided(
        out_shape, out_stride, device=src.device, dtype=torch.bfloat16,
    )
    scatter_out = torch.empty_strided(
        out_shape, out_stride, device=src.device, dtype=torch.bfloat16,
    )
    zero_flat = torch.as_strided(zero_out, (OUT_NUMEL,), (1,))
    scatter_flat = torch.as_strided(scatter_out, (OUT_NUMEL,), (1,))
    src_flat = src.view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(OUT_NUMEL, BLOCK), 1, 1),
        _zero_and_scatter_kernel,
        (src_flat, zero_flat, scatter_flat, BLOCK),
    )
    return zero_out, scatter_out
