"""
Standalone repro captured via capture_hook.
Label: inductor_timm_perf-2-6-linux.aws.a100_graph15
Pattern hash: 8d36dae866fa
Shape hash: 5ec24099
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
    def forward(self, convolution_76: "bf16[128, 768, 8, 8]", arg221_1: "bf16[1536, 768, 1, 1]", arg222_1: "bf16[1536, 1, 1, 1]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[128, 768, 8, 8]" = torch.ops.prims.convert_element_type.default(convolution_76, torch.float32);  convolution_76 = None
        mul_tensor: "f32[128, 768, 8, 8]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.5)
        mul_tensor_1: "f32[128, 768, 8, 8]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.7071067811865476);  convert_element_type_default = None
        erf_default: "f32[128, 768, 8, 8]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[128, 768, 8, 8]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[128, 768, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None
        convert_element_type_default_1: "bf16[128, 768, 8, 8]" = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.bfloat16);  mul_tensor_2 = None
        mul_tensor_3: "bf16[128, 768, 8, 8]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.7015043497085571);  convert_element_type_default_1 = None
        view_default: "bf16[1, 1536, 768]" = torch.ops.aten.view.default(arg221_1, _shape_param_0);  arg221_1 = _shape_param_0 = None
        mul_tensor_4: "bf16[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg222_1, 0.03608439182435161);  arg222_1 = None
        view_default_1: "bf16[1536]" = torch.ops.aten.view.default(mul_tensor_4, [-1]);  mul_tensor_4 = None
        convert_element_type_default_2: "f32[1, 1536, 768]" = torch.ops.prims.convert_element_type.default(view_default, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default_2, [0, 2], correction = 0, keepdim = True);  convert_element_type_default_2 = None
        getitem: "f32[1, 1536, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 1536, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[1, 1536, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 1536, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(view_default, getitem_1);  view_default = getitem_1 = None
        mul_tensor_5: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "bf16[1536, 1]" = torch.ops.aten.unsqueeze.default(view_default_1, -1);  view_default_1 = None
        mul_tensor_6: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_5, unsqueeze_default);  mul_tensor_5 = unsqueeze_default = None
        convert_element_type_default_3: "bf16[1, 1536, 768]" = torch.ops.prims.convert_element_type.default(mul_tensor_6, torch.bfloat16);  mul_tensor_6 = None
        view_default_2: "bf16[1536, 768, 1, 1]" = torch.ops.aten.view.default(convert_element_type_default_3, _shape_param_1);  convert_element_type_default_3 = _shape_param_1 = None
        return (mul_tensor_3, view_default_2)


def _default_make_inputs():
    return [
    torch.randn([128, 768, 8, 8], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1536, 768, 1, 1], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1536, 1, 1, 1], dtype=torch.bfloat16, device='cuda'),
    [1, 1536, -1],  # _shape_param_0
    [1536, 768, 1, 1],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
