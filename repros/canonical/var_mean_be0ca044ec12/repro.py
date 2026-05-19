"""
Standalone repro captured via capture_hook.
Label: inductor_torchbench_perf-4-6-linux.aws.a100_graph24
Pattern hash: be0ca044ec12
Shape hash: ea620de9
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
    def forward(self, convolution_47: "f16[32, 192, 7, 7]", arg287_1: "f32[192]", arg288_1: "f32[192]", add_233: "f16[32, 192, 7, 7]", arg289_1: "f32[1152, 192, 1, 1]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[32, 192, 7, 7]" = torch.ops.prims.convert_element_type.default(convolution_47, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_default = None
        getitem: "f32[1, 192, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 192, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[32, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_47, getitem_1);  convolution_47 = getitem_1 = None
        mul_tensor: "f32[32, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg287_1, -1);  arg287_1 = None
        unsqueeze_default_1: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[32, 192, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg288_1, -1);  arg288_1 = None
        unsqueeze_default_3: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[32, 192, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        convert_element_type_default_1: "f16[32, 192, 7, 7]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None
        add_tensor_2: "f16[32, 192, 7, 7]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, add_233);  convert_element_type_default_1 = add_233 = None
        convert_element_type_default_2: "f16[1152, 192, 1, 1]" = torch.ops.prims.convert_element_type.default(arg289_1, torch.float16);  arg289_1 = None
        return (add_tensor_2, convert_element_type_default_2)


def _default_make_inputs():
    return [
    torch.randn([32, 192, 7, 7], dtype=torch.float16, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([32, 192, 7, 7], dtype=torch.float16, device='cuda'),
    torch.randn([1152, 192, 1, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
