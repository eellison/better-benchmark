"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle mirrors Inductor's generated persistent row-reduction kernel for the complete TrOCR bf16 softmax-backward row update, including the `[1024,256,256] -> [64,16,256,256]` metadata view, fp32 broadcast add from the `[64,1,256,256]` tensor, fp32 subtract/`aten.exp`/divide by the row denominator, fp32 multiplication by the bf16 gradient input, the last-dimension `sum(..., keepdim=True)`, `prims.fma(-div, row_sum, mul)` epilogue, bf16 conversion, and final contiguous `[1024,256,256]` view. Inductor already lowers this full envelope as one persistent reduction, so the remaining finding is deterministic numerics drift versus eager on large bf16-scaled rows rather than missing scope; the fix is BANDWIDTH_BOUND/NUMERICS_FLAG: preserve the generated Inductor math order, casts, strides, and output scope for floor measurement while reporting the eager-check mismatch."""

import torch
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import DeviceProperties, ReductionHint

from oracle_harness import oracle_impl

triton_helpers.set_driver_to_gpu()


ROWS = 1024 * 256
K = 256


def _device_properties():
    return DeviceProperties.create(torch.device("cuda", torch.cuda.current_device()))


@triton_heuristics.persistent_reduction(
    size_hints={"x": 262144, "r0_": 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={
        "signature": {
            "in_ptr0": "*bf16",
            "in_ptr1": "*bf16",
            "in_ptr2": "*fp32",
            "in_ptr3": "*fp32",
            "in_ptr4": "*fp32",
            "out_ptr2": "*bf16",
            "xnumel": "i32",
            "r0_numel": "i32",
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
                (5,): [["tt.divisibility", 16]],
                (6,): [["tt.divisibility", 16]],
                (7,): [["tt.divisibility", 16]],
            }
        ],
    },
    inductor_meta={
        "grid_type": "Grid1D",
        "kernel_name": "_triton_per_fused_convert_element_type_div_exp_mul_sub_sum",
        "mutated_arg_names": [],
        "optimize_mem": True,
        "no_x_dim": True,
        "num_load": 6,
        "num_reduction": 1,
        "autotune_hints": set(),
        "autotune_local_cache": True,
        "autotune_pointwise": True,
        "force_disable_caches": False,
        "dynamic_scale_rblock": True,
        "max_autotune": False,
        "max_autotune_pointwise": False,
        "min_split_scan_rblock": 256,
        "spill_threshold": 16,
    },
)
@triton.jit
def _softmax_backward_kernel(
    in_ptr0,
    in_ptr1,
    in_ptr2,
    in_ptr3,
    in_ptr4,
    out_ptr2,
    xnumel,
    r0_numel,
):
    xnumel = 262144
    XBLOCK: tl.constexpr = 1
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_2 = r0_index
    x3 = xindex
    x0 = xindex % 256
    x1 = xindex // 256
    x5 = xindex // 4096

    tmp0 = tl.load(in_ptr0 + (r0_2 + 256 * x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_2 + 256 * x3), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r0_2 + 256 * x0 + 65536 * (x1 // 16)), None)
    tmp6 = tl.load(in_ptr3 + x3, None, eviction_policy="evict_last")
    tmp9 = tl.load(in_ptr4 + x3, None, eviction_policy="evict_last")
    tmp15 = tl.load(
        in_ptr2 + (r0_2 + 256 * x0 + 65536 * x5),
        None,
        eviction_policy="evict_last",
    )
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 - tmp6
    tmp8 = tl_math.exp(tmp7)
    tmp10 = tmp8 / tmp9
    tmp11 = tmp1 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [R0_BLOCK])
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp16 = tmp3 + tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tl_math.exp(tmp17)
    tmp19 = tmp18 / tmp9
    tmp20 = -tmp19
    tmp21 = libdevice.fma(tmp20, tmp14, tmp11)
    tmp22 = tmp21.to(tl.float32)
    tl.store(out_ptr2 + (r0_2 + 256 * x3), tmp22, None)


# b0ff1d38: TrOCR bf16 softmax-backward row update, K=256.
@oracle_impl(hardware="B200", point="b0ff1d38", num_warps=8, num_stages=3)
def oracle_forward(inputs, *, num_warps: int, num_stages: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, *_shape_params = inputs
    out = torch.empty_strided(
        (1024, 256, 256),
        (K * K, K, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    _softmax_backward_kernel.run(
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        out,
        ROWS,
        K,
        stream=get_raw_stream(arg0_1.get_device()),
    )
    return out
