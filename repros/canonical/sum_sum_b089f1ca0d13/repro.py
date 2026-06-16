"""
Standalone repro captured via capture_hook.
Label: torchbench_squeezenet1_1_train
Pattern hash: b089f1ca0d13
Shape hash: 3bd72ec5
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[32, 256, 13, 13]", arg1_1: "i8[32, 256, 13, 13]", arg2_1: "b8[32, 128, 27, 27]", arg3_1: "bf16[]", arg4_1: "b8[32, 128, 27, 27]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        full: "f32[8192, 729]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        clone: "bf16[32, 256, 13, 13]" = torch.ops.aten.clone.default(arg0_1, memory_format = torch.contiguous_format);  arg0_1 = None
        _unsafe_view: "bf16[8192, 169]" = torch.ops.aten._unsafe_view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        _low_memory_max_pool_offsets_to_indices: "i64[32, 256, 13, 13]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(arg1_1, _shape_param_2, _shape_param_3, _shape_param_4, [0, 0], [1, 1]);  arg1_1 = _shape_param_2 = _shape_param_3 = _shape_param_4 = None
        clone_1: "i64[32, 256, 13, 13]" = torch.ops.aten.clone.default(_low_memory_max_pool_offsets_to_indices, memory_format = torch.contiguous_format);  _low_memory_max_pool_offsets_to_indices = None
        _unsafe_view_1: "i64[8192, 169]" = torch.ops.aten._unsafe_view.default(clone_1, _shape_param_5);  clone_1 = _shape_param_5 = None
        convert_element_type: "f32[8192, 169]" = torch.ops.prims.convert_element_type.default(_unsafe_view, torch.float32);  _unsafe_view = None
        scatter_add: "f32[8192, 729]" = torch.ops.aten.scatter_add.default(full, 1, _unsafe_view_1, convert_element_type);  full = _unsafe_view_1 = convert_element_type = None
        view: "f32[32, 256, 27, 27]" = torch.ops.aten.view.default(scatter_add, _shape_param_6);  scatter_add = _shape_param_6 = None
        convert_element_type_1: "bf16[32, 256, 27, 27]" = torch.ops.prims.convert_element_type.default(view, torch.bfloat16);  view = None
        slice_1: "bf16[32, 128, 27, 27]" = torch.ops.aten.slice.Tensor(convert_element_type_1, 1, 0, 128)
        slice_2: "bf16[32, 128, 27, 27]" = torch.ops.aten.slice.Tensor(convert_element_type_1, 1, 128, 256);  convert_element_type_1 = None
        where: "bf16[32, 128, 27, 27]" = torch.ops.aten.where.self(arg2_1, arg3_1, slice_2);  arg2_1 = slice_2 = None
        sum_1: "bf16[128]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
        convert_element_type_2: "f32[128]" = torch.ops.prims.convert_element_type.default(sum_1, torch.float32);  sum_1 = None
        where_1: "bf16[32, 128, 27, 27]" = torch.ops.aten.where.self(arg4_1, arg3_1, slice_1);  arg4_1 = arg3_1 = slice_1 = None
        sum_2: "bf16[128]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
        convert_element_type_3: "f32[128]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        return (where, convert_element_type_2, where_1, convert_element_type_3)



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
