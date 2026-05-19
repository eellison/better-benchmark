"""
Standalone repro captured via capture_hook.
Label: inductor_timm_perf_cuda_h100-4-7-linux.aws.h100_graph53
Pattern hash: ccc91f85f833
Shape hash: ece7b951
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
    def forward(self, convolution_92: "f16[8, 80, 7, 7]", arg501_1: "f32[80]", arg502_1: "f32[80]", convert_element_type_276: "f16[8, 80, 7, 7]", add_395: "f16[8, 160, 7, 7]", arg503_1: "f32[960, 160, 1, 1]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[8, 80, 7, 7]" = torch.ops.prims.convert_element_type.default(convolution_92, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_default = None
        getitem: "f32[1, 80, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 80, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 80, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 80, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_92, getitem_1);  convolution_92 = getitem_1 = None
        mul_tensor: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg501_1, -1);  arg501_1 = None
        unsqueeze_default_1: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg502_1, -1);  arg502_1 = None
        unsqueeze_default_3: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        convert_element_type_default_1: "f16[8, 80, 7, 7]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None
        cat_default: "f16[8, 160, 7, 7]" = torch.ops.aten.cat.default([convert_element_type_276, convert_element_type_default_1], 1);  convert_element_type_276 = convert_element_type_default_1 = None
        add_tensor_2: "f16[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(cat_default, add_395);  cat_default = add_395 = None
        convert_element_type_default_2: "f16[960, 160, 1, 1]" = torch.ops.prims.convert_element_type.default(arg503_1, torch.float16);  arg503_1 = None
        return (add_tensor_2, convert_element_type_default_2)


def _default_make_inputs():
    return [
    torch.randn([8, 80, 7, 7], dtype=torch.float16, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([8, 80, 7, 7], dtype=torch.float16, device='cuda'),
    torch.randn([8, 160, 7, 7], dtype=torch.float16, device='cuda'),
    torch.randn([960, 160, 1, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
