"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Longformer attention-head layout materialization by directly filling the final contiguous `[384,256,64]` output storage from the contiguous `[8192,768]` projection in one Triton layout-copy kernel, folding the intermediate view/permute/view/permute/clone/view/size-one-permute/view chain into the final store map; Inductor lowers the same layout-only graph through a generic pointwise copy schedule with general reshape/transpose index decoding; the fix is NEW_PATTERN: add a guarded Longformer head-layout copy template for `[B,S,H*D] -> [B*H*blocks,tile,D]` materializations that writes the final view storage directly."""

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
    size_hints={"x": 8388608},
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
        "kernel_name": "_longformer_head_layout_kernel",
        "no_x_dim": False,
        "atomic_add_found": False,
        "num_load": 1,
        "num_store": 1,
        "num_reduction": 0,
        "autotune_hints": set(),
        "tiling_scores": {"x": 37748736},
        "backend_hash": "95BE156F202C2AC9B7A14D0E8D63E9A49EB82A31C51A6C460D0AD67FC0560293",
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
        "coordinate_descent_tuning": True,
        "coordinate_descent_search_radius": 1,
        "coordinate_descent_check_all_directions": False,
    },
    min_elem_per_thread=0,
)
@triton.jit
def _longformer_head_layout_kernel(
    input_ptr,
    output_ptr,
    xnumel,
    XBLOCK: tl.constexpr,
):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    x0 = xindex % 64
    x1 = (xindex // 64) % 1024
    x2 = (xindex // 65536) % 12
    x3 = xindex // 786432
    values = tl.load(input_ptr + (x0 + 64 * x2 + 768 * x1 + 786432 * x3), None).to(tl.float32)
    tl.store(output_ptr + xindex, values, None)


@oracle_impl(hardware="B200", point="07bfd41e")
def oracle_forward(inputs):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4 = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3

    output_shape = tuple(int(dim) for dim in _shape_param_4)
    numel = output_shape[0] * output_shape[1] * output_shape[2]
    out = torch.empty_strided(
        output_shape,
        (output_shape[1] * output_shape[2], output_shape[2], 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    _longformer_head_layout_kernel.run(
        arg0_1,
        out,
        numel,
        stream=_get_raw_stream(arg0_1.get_device()),
    )
    return out
