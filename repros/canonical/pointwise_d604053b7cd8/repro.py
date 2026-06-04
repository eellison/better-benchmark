"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_T5_train_001
Pattern hash: d604053b7cd8
Shape hash: 3d9a73b7
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([8192, 2048], f32), T([8, 1024, 2048], b8), T([8, 1024, 2048], b8), T([], f32), S([8, 1024, 2048]), S([8192, 2048]))"

class Repro(torch.nn.Module):
    def forward(self, mm_183: "f32[8192, 2048]", arg147_1: "b8[8, 1024, 2048]", arg429_1: "b8[8, 1024, 2048]", full_1: "f32[]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[8, 1024, 2048]" = torch.ops.aten.view.default(mm_183, _shape_param_0);  mm_183 = _shape_param_0 = None
        convert_element_type_default: "f32[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(arg147_1, torch.float32);  arg147_1 = None
        mul_tensor: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_default, mul_tensor);  view_default = mul_tensor = None
        where_self: "f32[8, 1024, 2048]" = torch.ops.aten.where.self(arg429_1, full_1, mul_tensor_1);  arg429_1 = full_1 = mul_tensor_1 = None
        view_default_1: "f32[8192, 2048]" = torch.ops.aten.view.default(where_self, _shape_param_1);  where_self = _shape_param_1 = None
        permute_default: "f32[2048, 8192]" = torch.ops.aten.permute.default(view_default_1, [1, 0]);  view_default_1 = None
        return permute_default

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
