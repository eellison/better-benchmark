"""
Standalone repro captured via capture_hook.
Label: inductor_timm_perf_cuda_h100-4-7-linux.aws.h100_graph53
Pattern hash: 402da566e943
Shape hash: 0f0f46d2
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([8, 80, 7, 7], f16), T([80], f32), T([80], f32), T([8, 80, 7, 7], f16), T([8, 160, 7, 7], f16), T([160], f32), T([160], f32), T([480, 160, 1, 1], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_70: "f16[8, 80, 7, 7]", arg385_1: "f32[80]", arg386_1: "f32[80]", convert_element_type_210: "f16[8, 80, 7, 7]", convolution_72: "f16[8, 160, 7, 7]", arg397_1: "f32[160]", arg398_1: "f32[160]", arg399_1: "f32[480, 160, 1, 1]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[8, 80, 7, 7]" = torch.ops.prims.convert_element_type.default(convolution_70, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_default = None
        getitem: "f32[1, 80, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 80, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 80, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 80, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_70, getitem_1);  convolution_70 = getitem_1 = None
        mul_tensor: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg385_1, -1);  arg385_1 = None
        unsqueeze_default_1: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg386_1, -1);  arg386_1 = None
        unsqueeze_default_3: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        convert_element_type_default_1: "f16[8, 80, 7, 7]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None
        cat_default: "f16[8, 160, 7, 7]" = torch.ops.aten.cat.default([convert_element_type_210, convert_element_type_default_1], 1);  convert_element_type_210 = convert_element_type_default_1 = None
        convert_element_type_default_2: "f32[8, 160, 7, 7]" = torch.ops.prims.convert_element_type.default(convolution_72, torch.float32)
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convert_element_type_default_2, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_default_2 = None
        getitem_2: "f32[1, 160, 1, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[1, 160, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_2: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_default_1: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor_1: "f32[8, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_72, getitem_3);  convolution_72 = getitem_3 = None
        mul_tensor_2: "f32[8, 160, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        unsqueeze_default_4: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg397_1, -1);  arg397_1 = None
        unsqueeze_default_5: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_3: "f32[8, 160, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_2, unsqueeze_default_5);  mul_tensor_2 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg398_1, -1);  arg398_1 = None
        unsqueeze_default_7: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_3: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_3, unsqueeze_default_7);  mul_tensor_3 = unsqueeze_default_7 = None
        convert_element_type_default_3: "f16[8, 160, 7, 7]" = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float16);  add_tensor_3 = None
        add_tensor_4: "f16[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(cat_default, convert_element_type_default_3);  cat_default = convert_element_type_default_3 = None
        convert_element_type_default_4: "f16[480, 160, 1, 1]" = torch.ops.prims.convert_element_type.default(arg399_1, torch.float16);  arg399_1 = None
        return (add_tensor_4, convert_element_type_default_4)


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
