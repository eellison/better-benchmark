"""
Standalone repro captured via capture_hook.
Label: torchbench_vgg16_train_001
Pattern hash: 9fb0f8852874
Shape hash: 297a9ccc
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 4096], b8), T([128, 4096], f32), T([128, 4096], b8), S([4096]))"

class Repro(torch.nn.Module):
    def forward(self, arg38_1: "b8[128, 4096]", mm: "f32[128, 4096]", arg40_1: "b8[128, 4096]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[128, 4096]" = torch.ops.prims.convert_element_type.default(arg38_1, torch.float32);  arg38_1 = None
        mul_tensor: "f32[128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 2.0);  convert_element_type_default = None
        mul_tensor_1: "f32[128, 4096]" = torch.ops.aten.mul.Tensor(mm, mul_tensor);  mm = mul_tensor = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[128, 4096]" = torch.ops.aten.where.self(arg40_1, full_default, mul_tensor_1);  arg40_1 = full_default = mul_tensor_1 = None
        permute_default: "f32[4096, 128]" = torch.ops.aten.permute.default(where_self, [1, 0])
        sum_dim_int_list: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(where_self, [0], True);  where_self = None
        view_default: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None
        return (permute_default, view_default)

def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)

def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
