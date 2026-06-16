"""
Standalone repro captured via capture_hook.
Label: torchbench_alexnet_train
Pattern hash: ccfc54d6b3cf
Shape hash: e1c3cc78
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
    def forward(self, arg0_1: "bf16[128, 256, 6, 6]", arg1_1: "i8[128, 256, 6, 6]", arg2_1: "b8[128, 256, 13, 13]", arg3_1: "bf16[]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        full: "f32[32768, 169]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        view: "bf16[32768, 36]" = torch.ops.aten.view.default(arg0_1, _shape_param_1);  arg0_1 = _shape_param_1 = None
        _low_memory_max_pool_offsets_to_indices: "i64[128, 256, 6, 6]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(arg1_1, _shape_param_2, _shape_param_3, _shape_param_4, [0, 0], [1, 1]);  arg1_1 = _shape_param_2 = _shape_param_3 = _shape_param_4 = None
        view_1: "i64[32768, 36]" = torch.ops.aten.view.default(_low_memory_max_pool_offsets_to_indices, _shape_param_5);  _low_memory_max_pool_offsets_to_indices = _shape_param_5 = None
        convert_element_type: "f32[32768, 36]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        scatter_add: "f32[32768, 169]" = torch.ops.aten.scatter_add.default(full, 1, view_1, convert_element_type);  full = view_1 = convert_element_type = None
        view_2: "f32[128, 256, 13, 13]" = torch.ops.aten.view.default(scatter_add, _shape_param_6);  scatter_add = _shape_param_6 = None
        convert_element_type_1: "bf16[128, 256, 13, 13]" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        where: "bf16[128, 256, 13, 13]" = torch.ops.aten.where.self(arg2_1, arg3_1, convert_element_type_1);  arg2_1 = arg3_1 = convert_element_type_1 = None
        sum_1: "bf16[256]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
        convert_element_type_2: "f32[256]" = torch.ops.prims.convert_element_type.default(sum_1, torch.float32);  sum_1 = None
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
