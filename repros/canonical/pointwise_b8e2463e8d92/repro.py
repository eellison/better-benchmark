"""
Standalone repro captured via capture_hook.
Label: torchbench_densenet121_infer_000
Pattern hash: b8e2463e8d92
Shape hash: 4432a5a8
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([64, 512, 7, 7], f16), T([64, 32, 7, 7], f16), T([64, 32, 7, 7], f16), T([64, 32, 7, 7], f16), T([608], f16), T([608], f16), T([608], f16), T([608], f16))"

class Repro(torch.nn.Module):
    def forward(self, avg_pool2d_2: "f16[64, 512, 7, 7]", convolution_89: "f16[64, 32, 7, 7]", convolution_91: "f16[64, 32, 7, 7]", convolution_93: "f16[64, 32, 7, 7]", arg471_1: "f16[608]", arg472_1: "f16[608]", arg473_1: "f16[608]", arg474_1: "f16[608]"):
        # No stacktrace found for following nodes
        cat_default: "f16[64, 608, 7, 7]" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93], 1);  avg_pool2d_2 = convolution_89 = convolution_91 = convolution_93 = None
        convert_element_type_default: "f32[608]" = torch.ops.prims.convert_element_type.default(arg471_1, torch.float32);  arg471_1 = None
        unsqueeze_default: "f32[608, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default, -1);  convert_element_type_default = None
        unsqueeze_default_1: "f32[608, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[64, 608, 7, 7]" = torch.ops.aten.sub.Tensor(cat_default, unsqueeze_default_1);  cat_default = unsqueeze_default_1 = None
        convert_element_type_default_1: "f32[608]" = torch.ops.prims.convert_element_type.default(arg472_1, torch.float32);  arg472_1 = None
        add_tensor: "f32[608]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1e-05);  convert_element_type_default_1 = None
        sqrt_default: "f32[608]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[608]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[608]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[608, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[608, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[64, 608, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f16[608, 1]" = torch.ops.aten.unsqueeze.default(arg473_1, -1);  arg473_1 = None
        unsqueeze_default_5: "f16[608, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[64, 608, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f16[608, 1]" = torch.ops.aten.unsqueeze.default(arg474_1, -1);  arg474_1 = None
        unsqueeze_default_7: "f16[608, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[64, 608, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        convert_element_type_default_2: "f16[64, 608, 7, 7]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None
        relu_default: "f16[64, 608, 7, 7]" = torch.ops.aten.relu.default(convert_element_type_default_2);  convert_element_type_default_2 = None
        return relu_default



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
