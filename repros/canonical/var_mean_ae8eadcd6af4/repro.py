"""
Standalone repro captured via capture_hook.
Label: inductor_timm_perf-6-6-linux.aws.a100_graph21
Pattern hash: ae8eadcd6af4
Shape hash: 6912c1d1
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
    def forward(self, convolution: "bf16[128, 768, 37, 37]", arg4_1: "bf16[1, 1, 768]", arg3_1: "bf16[1, 1370, 768]", arg5_1: "bf16[768]", arg6_1: "bf16[768]", arg7_1: "bf16[2304, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "bf16[128, 768, 1369]" = torch.ops.aten.view.default(convolution, _shape_param_0);  convolution = _shape_param_0 = None
        permute_default: "bf16[128, 1369, 768]" = torch.ops.aten.permute.default(view_default, [0, 2, 1]);  view_default = None
        expand_default: "bf16[128, 1, 768]" = torch.ops.aten.expand.default(arg4_1, _shape_param_1);  arg4_1 = _shape_param_1 = None
        cat_default: "bf16[128, 1370, 768]" = torch.ops.aten.cat.default([expand_default, permute_default], 1);  expand_default = permute_default = None
        add_tensor: "bf16[128, 1370, 768]" = torch.ops.aten.add.Tensor(cat_default, arg3_1);  cat_default = arg3_1 = None
        convert_element_type_default: "f32[128, 1370, 768]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 1370, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 1370, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[128, 1370, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt_default: "f32[128, 1370, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[128, 1370, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1);  convert_element_type_default = getitem_1 = None
        mul_tensor: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg5_1);  mul_tensor = arg5_1 = None
        add_tensor_2: "f32[128, 1370, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg6_1);  mul_tensor_1 = arg6_1 = None
        convert_element_type_default_1: "bf16[128, 1370, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None
        view_default_1: "bf16[175360, 768]" = torch.ops.aten.view.default(convert_element_type_default_1, _shape_param_2);  convert_element_type_default_1 = _shape_param_2 = None
        permute_default_1: "bf16[768, 2304]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        return (view_default_1, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([128, 768, 37, 37], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 1, 768], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 1370, 768], dtype=torch.bfloat16, device='cuda'),
    torch.randn([768], dtype=torch.bfloat16, device='cuda'),
    torch.randn([768], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2304, 768], dtype=torch.bfloat16, device='cuda'),
    [128, 768, 1369],  # _shape_param_0
    [128, -1, -1],  # _shape_param_1
    [175360, 768],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
