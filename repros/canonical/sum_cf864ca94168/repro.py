"""
Standalone repro captured via capture_hook.
Label: torchbench_alexnet_train
Pattern hash: cf864ca94168
Shape hash: 94a62ed8
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
    def forward(self, arg0_1: "bf16[128, 64, 27, 27]", arg1_1: "i8[128, 64, 27, 27]", arg2_1: "b8[128, 64, 55, 55]", arg3_1: "bf16[]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        full: "f32[8192, 3025]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        clone: "bf16[128, 64, 27, 27]" = torch.ops.aten.clone.default(arg0_1, memory_format = torch.contiguous_format);  arg0_1 = None
        _unsafe_view: "bf16[8192, 729]" = torch.ops.aten._unsafe_view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        _low_memory_max_pool_offsets_to_indices: "i64[128, 64, 27, 27]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(arg1_1, _shape_param_2, _shape_param_3, _shape_param_4, [0, 0], [1, 1]);  arg1_1 = _shape_param_2 = _shape_param_3 = _shape_param_4 = None
        clone_1: "i64[128, 64, 27, 27]" = torch.ops.aten.clone.default(_low_memory_max_pool_offsets_to_indices, memory_format = torch.contiguous_format);  _low_memory_max_pool_offsets_to_indices = None
        _unsafe_view_1: "i64[8192, 729]" = torch.ops.aten._unsafe_view.default(clone_1, _shape_param_5);  clone_1 = _shape_param_5 = None
        convert_element_type: "f32[8192, 729]" = torch.ops.prims.convert_element_type.default(_unsafe_view, torch.float32);  _unsafe_view = None
        scatter_add: "f32[8192, 3025]" = torch.ops.aten.scatter_add.default(full, 1, _unsafe_view_1, convert_element_type);  full = _unsafe_view_1 = convert_element_type = None
        view: "f32[128, 64, 55, 55]" = torch.ops.aten.view.default(scatter_add, _shape_param_6);  scatter_add = _shape_param_6 = None
        convert_element_type_1: "bf16[128, 64, 55, 55]" = torch.ops.prims.convert_element_type.default(view, torch.bfloat16);  view = None
        where: "bf16[128, 64, 55, 55]" = torch.ops.aten.where.self(arg2_1, arg3_1, convert_element_type_1);  arg2_1 = arg3_1 = convert_element_type_1 = None
        sum_1: "bf16[64]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
        convert_element_type_2: "f32[64]" = torch.ops.prims.convert_element_type.default(sum_1, torch.float32);  sum_1 = None
        return (where, convert_element_type_2)



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
