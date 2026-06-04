"""
Standalone repro captured via capture_hook.
Label: torchbench_mobilenet_v2_infer_000
Pattern hash: 604e62335f7c
Shape hash: 77470fa6
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([160], f16), T([128, 160, 7, 7], f16), T([160], f16), T([160], f16), T([160], f16), T([128, 160, 7, 7], f16))"

class Repro(torch.nn.Module):
    def forward(self, arg237_1: "f16[160]", convolution_47: "f16[128, 160, 7, 7]", arg238_1: "f16[160]", arg239_1: "f16[160]", arg240_1: "f16[160]", add_98: "f16[128, 160, 7, 7]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[160]" = torch.ops.prims.convert_element_type.default(arg237_1, torch.float32);  arg237_1 = None
        unsqueeze_default: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default, -1);  convert_element_type_default = None
        unsqueeze_default_1: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[128, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_default_1);  convolution_47 = unsqueeze_default_1 = None
        convert_element_type_default_1: "f32[160]" = torch.ops.prims.convert_element_type.default(arg238_1, torch.float32);  arg238_1 = None
        add_tensor: "f32[160]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1e-05);  convert_element_type_default_1 = None
        sqrt_default: "f32[160]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[128, 160, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f16[160, 1]" = torch.ops.aten.unsqueeze.default(arg239_1, -1);  arg239_1 = None
        unsqueeze_default_5: "f16[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[128, 160, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f16[160, 1]" = torch.ops.aten.unsqueeze.default(arg240_1, -1);  arg240_1 = None
        unsqueeze_default_7: "f16[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 160, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        convert_element_type_default_2: "f16[128, 160, 7, 7]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None
        add_tensor_2: "f16[128, 160, 7, 7]" = torch.ops.aten.add.Tensor(add_98, convert_element_type_default_2);  add_98 = convert_element_type_default_2 = None
        return add_tensor_2

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
