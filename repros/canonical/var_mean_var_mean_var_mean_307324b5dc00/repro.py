"""
Standalone repro captured via capture_hook.
Label: inductor_timm_perf-5-6-linux.aws.a100_graph18
Pattern hash: 307324b5dc00
Shape hash: 4dd4b6a7
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, relu_19: "f16[8, 384, 14, 14]", arg324_1: "f32[384]", arg325_1: "f32[384]", convolution_40: "f16[8, 384, 14, 14]", arg330_1: "f32[384]", arg331_1: "f32[384]", convolution_41: "f16[8, 384, 14, 14]", arg336_1: "f32[384]", arg337_1: "f32[384]", arg338_1: "f32[1408, 384, 1, 1]", arg344_1: "f32[1408, 384, 3, 3]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[8, 384, 14, 14]" = torch.ops.prims.convert_element_type.default(relu_19, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_default = None
        getitem: "f32[1, 384, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 384, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(relu_19, getitem_1);  relu_19 = getitem_1 = None
        mul_tensor: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg324_1, -1);  arg324_1 = None
        unsqueeze_default_1: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg325_1, -1);  arg325_1 = None
        unsqueeze_default_3: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        convert_element_type_default_1: "f16[8, 384, 14, 14]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None
        convert_element_type_default_2: "f32[8, 384, 14, 14]" = torch.ops.prims.convert_element_type.default(convolution_40, torch.float32)
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convert_element_type_default_2, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_default_2 = None
        getitem_2: "f32[1, 384, 1, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[1, 384, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_2: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_default_1: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor_1: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, getitem_3);  convolution_40 = getitem_3 = None
        mul_tensor_2: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        unsqueeze_default_4: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg330_1, -1);  arg330_1 = None
        unsqueeze_default_5: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_3: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_2, unsqueeze_default_5);  mul_tensor_2 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg331_1, -1);  arg331_1 = None
        unsqueeze_default_7: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_3: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_3, unsqueeze_default_7);  mul_tensor_3 = unsqueeze_default_7 = None
        convert_element_type_default_3: "f16[8, 384, 14, 14]" = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float16);  add_tensor_3 = None
        convert_element_type_default_4: "f32[8, 384, 14, 14]" = torch.ops.prims.convert_element_type.default(convolution_41, torch.float32)
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(convert_element_type_default_4, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_default_4 = None
        getitem_4: "f32[1, 384, 1, 1]" = var_mean_correction_2[0]
        getitem_5: "f32[1, 384, 1, 1]" = var_mean_correction_2[1];  var_mean_correction_2 = None
        add_tensor_4: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_default_2: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_4);  add_tensor_4 = None
        sub_tensor_2: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, getitem_5);  convolution_41 = getitem_5 = None
        mul_tensor_4: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default_2);  sub_tensor_2 = rsqrt_default_2 = None
        unsqueeze_default_8: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg336_1, -1);  arg336_1 = None
        unsqueeze_default_9: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, -1);  unsqueeze_default_8 = None
        mul_tensor_5: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_4, unsqueeze_default_9);  mul_tensor_4 = unsqueeze_default_9 = None
        unsqueeze_default_10: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg337_1, -1);  arg337_1 = None
        unsqueeze_default_11: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, -1);  unsqueeze_default_10 = None
        add_tensor_5: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_5, unsqueeze_default_11);  mul_tensor_5 = unsqueeze_default_11 = None
        convert_element_type_default_5: "f16[8, 384, 14, 14]" = torch.ops.prims.convert_element_type.default(add_tensor_5, torch.float16);  add_tensor_5 = None
        add_tensor_6: "f16[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_default_3, convert_element_type_default_5);  convert_element_type_default_3 = convert_element_type_default_5 = None
        add_tensor_7: "f16[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_6, convert_element_type_default_1);  add_tensor_6 = convert_element_type_default_1 = None
        relu_default: "f16[8, 384, 14, 14]" = torch.ops.aten.relu.default(add_tensor_7);  add_tensor_7 = None
        convert_element_type_default_6: "f16[1408, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(arg338_1, torch.float16);  arg338_1 = None
        convert_element_type_default_7: "f16[1408, 384, 3, 3]" = torch.ops.prims.convert_element_type.default(arg344_1, torch.float16);  arg344_1 = None
        return (relu_default, convert_element_type_default_6, convert_element_type_default_7)


def _default_make_inputs():
    return [
    torch.randn([8, 384, 14, 14], dtype=torch.float16, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([8, 384, 14, 14], dtype=torch.float16, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([8, 384, 14, 14], dtype=torch.float16, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([1408, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1408, 384, 3, 3], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
