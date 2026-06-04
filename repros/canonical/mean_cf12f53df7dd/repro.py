"""
Standalone repro captured via capture_hook.
Label: torchbench_mobilenet_v3_large_infer_000
Pattern hash: cf12f53df7dd
Shape hash: d38f7ef8
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([960], f16), T([256, 960, 7, 7], f16), T([960], f16), T([960], f16), T([960], f16), S([256, 960]))"

class Repro(torch.nn.Module):
    def forward(self, arg259_1: "f16[960]", convolution_61: "f16[256, 960, 7, 7]", arg260_1: "f16[960]", arg261_1: "f16[960]", arg262_1: "f16[960]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[960]" = torch.ops.prims.convert_element_type.default(arg259_1, torch.float32);  arg259_1 = None
        unsqueeze_default: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default, -1);  convert_element_type_default = None
        unsqueeze_default_1: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[256, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_default_1);  convolution_61 = unsqueeze_default_1 = None
        convert_element_type_default_1: "f32[960]" = torch.ops.prims.convert_element_type.default(arg260_1, torch.float32);  arg260_1 = None
        add_tensor: "f32[960]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 0.001);  convert_element_type_default_1 = None
        sqrt_default: "f32[960]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[256, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f16[960, 1]" = torch.ops.aten.unsqueeze.default(arg261_1, -1);  arg261_1 = None
        unsqueeze_default_5: "f16[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[256, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f16[960, 1]" = torch.ops.aten.unsqueeze.default(arg262_1, -1);  arg262_1 = None
        unsqueeze_default_7: "f16[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[256, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        convert_element_type_default_2: "f32[256, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32);  add_tensor_1 = None
        add_tensor_2: "f32[256, 960, 7, 7]" = torch.ops.aten.add.Tensor(convert_element_type_default_2, 3)
        clamp_min_default: "f32[256, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_tensor_2, 0);  add_tensor_2 = None
        clamp_max_default: "f32[256, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        mul_tensor_3: "f32[256, 960, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, clamp_max_default);  convert_element_type_default_2 = clamp_max_default = None
        div_tensor: "f32[256, 960, 7, 7]" = torch.ops.aten.div.Tensor(mul_tensor_3, 6);  mul_tensor_3 = None
        convert_element_type_default_3: "f16[256, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.float16);  div_tensor = None
        mean_dim: "f16[256, 960, 1, 1]" = torch.ops.aten.mean.dim(convert_element_type_default_3, [-1, -2], True);  convert_element_type_default_3 = None
        view_default: "f16[256, 960]" = torch.ops.aten.view.default(mean_dim, _shape_param_0);  mean_dim = _shape_param_0 = None
        return view_default

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
