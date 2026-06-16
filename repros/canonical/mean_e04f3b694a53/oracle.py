"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 NFNet exact-erf GELU, explicit bf16 GELU rounding, fixed bf16 gamma scale, spatial mean over the rounded scaled activation, and final contiguous bf16 `[128,3072]` view in one channels-last Triton reduction kernel, whereas Inductor lowers the exact activation producer and small spatial mean through its generic reduction schedule with a visible dtype boundary; Inductor cannot do this today because its scheduler/codegen does not sink an exact-erf GELU producer with mandatory bf16 rounding boundaries into a fixed small-HW spatial mean template; the fix is SCHEDULER_FUSION: teach reduction scheduling to fuse exact-GELU-scale producers into small channels-last spatial means while preserving the cast boundaries and direct viewed output store."""

import torch
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.hints import DeviceProperties, ReductionHint
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl

triton_helpers.set_driver_to_gpu()


def _device_properties():
    return DeviceProperties.create(torch.device("cuda", torch.cuda.current_device()))


@triton_heuristics.reduction(
    size_hints={"x": 524288, "r0_": 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={
        "signature": {
            "x_ptr": "*bf16",
            "out_ptr": "*bf16",
            "HEIGHT": "i32",
            "WIDTH": "i32",
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
                (4,): [["tt.divisibility", 16]],
            }
        ],
    },
    inductor_meta={
        "grid_type": "Grid1D",
        "kernel_name": "_nfnet_gelu_spatial_mean_kernel",
        "mutated_arg_names": [],
        "optimize_mem": True,
        "no_x_dim": False,
        "num_load": 1,
        "num_store": 1,
        "num_reduction": 1,
        "autotune_hints": set(),
        "autotune_local_cache": True,
        "autotune_pointwise": True,
        "force_disable_caches": False,
        "dynamic_scale_rblock": True,
        "coordinate_descent_tuning": True,
        "coordinate_descent_search_radius": 1,
        "coordinate_descent_check_all_directions": False,
    },
)
@triton.jit
def _nfnet_gelu_spatial_mean_kernel(
    x_ptr,
    out_ptr,
    HEIGHT,
    WIDTH,
    xnumel,
    r0_numel,
    XBLOCK: tl.constexpr,
    R0_BLOCK: tl.constexpr,
):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    channel = xindex % 3072
    batch = xindex // 3072
    acc = tl.full([XBLOCK, R0_BLOCK], 0.0, tl.float32)

    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        offsets = channel + 3072 * r0_index + 3072 * HEIGHT * WIDTH * batch
        x = tl.load(
            x_ptr + offsets,
            mask=r0_mask,
            eviction_policy="evict_first",
            other=0.0,
        ).to(tl.float32)
        half_x = x * 0.5
        erf_arg = x * 0.7071067811865476
        gelu_bf16 = (half_x * (libdevice.erf(erf_arg) + 1.0)).to(tl.bfloat16)
        scaled_bf16 = (gelu_bf16.to(tl.float32) * 1.7015043497085571).to(tl.bfloat16)
        acc += tl.where(r0_mask, scaled_bf16.to(tl.float32), 0.0)

    total = tl.sum(acc, 1)[:, None]
    denom = (HEIGHT * WIDTH).to(tl.float32)
    mean = total / denom
    tl.store(out_ptr + xindex, mean.to(tl.bfloat16), None)


def _launch(inputs):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    out_shape = tuple(int(dim) for dim in _shape_param_2)
    out = torch.empty_strided(
        out_shape,
        (int(arg0_1.shape[1]), 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    n_size = int(arg0_1.shape[0])
    channels = int(arg0_1.shape[1])
    height = int(arg0_1.shape[2])
    width = int(arg0_1.shape[3])
    hw = height * width
    _nfnet_gelu_spatial_mean_kernel.run(
        arg0_1,
        out,
        height,
        width,
        n_size * channels,
        hw,
        stream=get_raw_stream(arg0_1.get_device()),
    )
    return out


# 3e28b16f: bf16[128,3072,8,8], channels-last
@oracle_impl(hardware="B200", point="3e28b16f")
# da46370d: bf16[128,3072,6,6], channels-last
@oracle_impl(hardware="B200", point="da46370d")
def oracle_forward(inputs):
    return _launch(inputs)
