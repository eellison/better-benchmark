"""
Standalone repro captured via capture_hook.
Label: torchbench_timm_efficientnet_infer_000
Pattern hash: c324cfc06a94
Shape hash: 3ed92b6f
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([1152], f16), T([64, 1152, 7, 7], f16), T([1152], f16), T([1152], f16), T([1152], f16))"

class Repro(torch.nn.Module):
    def forward(self, arg292_1: "f16[1152]", convolution_76: "f16[64, 1152, 7, 7]", arg293_1: "f16[1152]", arg294_1: "f16[1152]", arg295_1: "f16[1152]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[1152]" = torch.ops.prims.convert_element_type.default(arg292_1, torch.float32);  arg292_1 = None
        unsqueeze_default: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default, -1);  convert_element_type_default = None
        unsqueeze_default_1: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[64, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_default_1);  convolution_76 = unsqueeze_default_1 = None
        convert_element_type_default_1: "f32[1152]" = torch.ops.prims.convert_element_type.default(arg293_1, torch.float32);  arg293_1 = None
        add_tensor: "f32[1152]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1e-05);  convert_element_type_default_1 = None
        sqrt_default: "f32[1152]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[64, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f16[1152, 1]" = torch.ops.aten.unsqueeze.default(arg294_1, -1);  arg294_1 = None
        unsqueeze_default_5: "f16[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[64, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f16[1152, 1]" = torch.ops.aten.unsqueeze.default(arg295_1, -1);  arg295_1 = None
        unsqueeze_default_7: "f16[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[64, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        convert_element_type_default_2: "f32[64, 1152, 7, 7]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32);  add_tensor_1 = None
        neg_default: "f32[64, 1152, 7, 7]" = torch.ops.aten.neg.default(convert_element_type_default_2)
        exp_default: "f32[64, 1152, 7, 7]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_2: "f32[64, 1152, 7, 7]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[64, 1152, 7, 7]" = torch.ops.aten.div.Tensor(convert_element_type_default_2, add_tensor_2);  convert_element_type_default_2 = add_tensor_2 = None
        convert_element_type_default_3: "f16[64, 1152, 7, 7]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.float16);  div_tensor = None
        mean_dim: "f16[64, 1152, 1, 1]" = torch.ops.aten.mean.dim(convert_element_type_default_3, [2, 3], True);  convert_element_type_default_3 = None
        return mean_dim

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
