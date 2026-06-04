"""
Standalone repro captured via capture_hook.
Label: hf_BlenderbotForCausalLM_train_007
Pattern hash: b57af4bf1f43
Shape hash: fd11e6d8
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([32, 128, 2560], b8), T([32, 128, 2560], f32), S([4096, 2560]), S([2560]))"

class Repro(torch.nn.Module):
    def forward(self, arg27_1: "b8[32, 128, 2560]", arg28_1: "f32[32, 128, 2560]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[32, 128, 2560]" = torch.ops.prims.convert_element_type.default(arg27_1, torch.float32);  arg27_1 = None
        mul_tensor: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(arg28_1, mul_tensor);  arg28_1 = mul_tensor = None
        view_default: "f32[4096, 2560]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_0);  mul_tensor_1 = _shape_param_0 = None
        permute_default: "f32[2560, 4096]" = torch.ops.aten.permute.default(view_default, [1, 0])
        sum_dim_int_list: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_default, [0], True);  view_default = None
        view_default_1: "f32[2560]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_1);  sum_dim_int_list = _shape_param_1 = None
        return (permute_default, view_default_1)

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
