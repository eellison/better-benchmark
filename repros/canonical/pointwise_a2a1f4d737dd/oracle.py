"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 ALBERT scaled attention-score layout scope by directly materializing the fresh contiguous `[4096,4096]` clone storage from the contiguous `[512,512,64]` input with one Triton scale-and-head-transpose copy and returning both the materialized base tensor and its metadata-only transpose view, whereas Inductor lowers the captured view/mul/permute/clone/view/permute graph through generic pointwise layout materialization; Inductor cannot do this today because it does not recognize this scaled `view(B,H,S,D).permute(0,2,1,3).contiguous().view(B*S,H*D)` attention-score layout as one specialized copy pattern with sibling alias-view output; the fix is NEW_PATTERN: add a guarded scaled attention-score layout materialization lowering that emits the direct clone-storage copy and preserves the base-plus-transpose output contract."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK": 4096}, num_warps=8, num_stages=3),
    ],
    key=["N_ELEMENTS"],
)
@triton.jit
def _scaled_attention_scores_layout_kernel(
    input_ptr,
    base_ptr,
    N_ELEMENTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)

    dim = offsets % 64
    head = (offsets // 64) % 64
    seq = (offsets // 4096) % 512
    batch = offsets // 2097152

    input_offsets = dim + 64 * seq + 32768 * head + 2097152 * batch
    values = tl.load(input_ptr + input_offsets).to(tl.float32)
    scaled = values * 0.3535533905932738
    tl.store(base_ptr + offsets, scaled.to(tl.bfloat16, fp_downcast_rounding="rtne"))


# 226fbbfa: (T([512,512,64], bf16), S([8,64,512,64]), S([8,512,4096]), S([4096,4096]))
@oracle_impl(hardware="B200", point="226fbbfa")
def oracle_forward(inputs):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2

    base = torch.empty_strided((4096, 4096), (4096, 1), device=arg0_1.device, dtype=arg0_1.dtype)
    grid = lambda meta: (triton.cdiv(16777216, meta["BLOCK"]),)
    _scaled_attention_scores_layout_kernel[grid](
        arg0_1,
        base,
        N_ELEMENTS=16777216,
    )
    return base, base.permute(1, 0)
