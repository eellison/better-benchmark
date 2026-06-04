"""
Standalone repro captured via capture_hook.
Label: torchbench_timm_resnest_infer_000
Pattern hash: 6a5649feb1f2
Shape hash: 54ad4f87
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([1024], f16), T([32, 1024, 14, 14], f16), T([1024], f16), T([1024], f16), T([1024], f16), T([1024], f16), T([32, 1024, 14, 14], f16), T([1024], f16), T([1024], f16), T([1024], f16))"

class Repro(torch.nn.Module):
    def forward(self, arg91_1: "f16[1024]", convolution_19: "f16[32, 1024, 14, 14]", arg92_1: "f16[1024]", arg93_1: "f16[1024]", arg94_1: "f16[1024]", arg96_1: "f16[1024]", convolution_20: "f16[32, 1024, 14, 14]", arg97_1: "f16[1024]", arg98_1: "f16[1024]", arg99_1: "f16[1024]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg91_1, torch.float32);  arg91_1 = None
        unsqueeze_default: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default, -1);  convert_element_type_default = None
        unsqueeze_default_1: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_default_1);  convolution_19 = unsqueeze_default_1 = None
        convert_element_type_default_1: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg92_1, torch.float32);  arg92_1 = None
        add_tensor: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1e-05);  convert_element_type_default_1 = None
        sqrt_default: "f32[1024]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg93_1, -1);  arg93_1 = None
        unsqueeze_default_5: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_default_7: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        convert_element_type_default_2: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None
        convert_element_type_default_3: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg96_1, torch.float32);  arg96_1 = None
        unsqueeze_default_8: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_3, -1);  convert_element_type_default_3 = None
        unsqueeze_default_9: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, -1);  unsqueeze_default_8 = None
        sub_tensor_1: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_default_9);  convolution_20 = unsqueeze_default_9 = None
        convert_element_type_default_4: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg97_1, torch.float32);  arg97_1 = None
        add_tensor_2: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_default_4, 1e-05);  convert_element_type_default_4 = None
        sqrt_default_1: "f32[1024]" = torch.ops.aten.sqrt.default(add_tensor_2);  add_tensor_2 = None
        reciprocal_default_1: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_default_1);  sqrt_default_1 = None
        mul_tensor_3: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_default_1, 1);  reciprocal_default_1 = None
        unsqueeze_default_10: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, -1);  mul_tensor_3 = None
        unsqueeze_default_11: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, -1);  unsqueeze_default_10 = None
        mul_tensor_4: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_11);  sub_tensor_1 = unsqueeze_default_11 = None
        unsqueeze_default_12: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg98_1, -1);  arg98_1 = None
        unsqueeze_default_13: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, -1);  unsqueeze_default_12 = None
        mul_tensor_5: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_4, unsqueeze_default_13);  mul_tensor_4 = unsqueeze_default_13 = None
        unsqueeze_default_14: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_default_15: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, -1);  unsqueeze_default_14 = None
        add_tensor_3: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_5, unsqueeze_default_15);  mul_tensor_5 = unsqueeze_default_15 = None
        convert_element_type_default_5: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float16);  add_tensor_3 = None
        add_tensor_4: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_default_2, convert_element_type_default_5);  convert_element_type_default_2 = convert_element_type_default_5 = None
        relu_default: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_tensor_4);  add_tensor_4 = None
        avg_pool2d_default: "f16[32, 1024, 7, 7]" = torch.ops.aten.avg_pool2d.default(relu_default, [2, 2], [2, 2], [0, 0], True, False);  relu_default = None
        return avg_pool2d_default

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
