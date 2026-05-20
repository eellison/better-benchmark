"""
Standalone repro captured via capture_hook.
Label: inductor_timm_perf-6-6-linux.aws.a100_graph21
Pattern hash: 5abc04a98527
Shape hash: ab506e77
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([175360, 768], bf16), T([768], bf16), T([128, 1370, 768], bf16), T([768], bf16), T([768], bf16), T([3072, 768], bf16), S([128, 1370, 768]), S([175360, 768]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_45: "bf16[175360, 768]", arg165_1: "bf16[768]", add_77: "bf16[128, 1370, 768]", arg166_1: "bf16[768]", arg167_1: "bf16[768]", arg168_1: "bf16[3072, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "bf16[128, 1370, 768]" = torch.ops.aten.view.default(addmm_45, _shape_param_0);  addmm_45 = _shape_param_0 = None
        mul_tensor: "bf16[128, 1370, 768]" = torch.ops.aten.mul.Tensor(view_default, arg165_1);  view_default = arg165_1 = None
        add_tensor: "bf16[128, 1370, 768]" = torch.ops.aten.add.Tensor(add_77, mul_tensor);  add_77 = mul_tensor = None
        convert_element_type_default: "f32[128, 1370, 768]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 1370, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 1370, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[128, 1370, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt_default: "f32[128, 1370, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[128, 1370, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1);  convert_element_type_default = getitem_1 = None
        mul_tensor_1: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_2: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg166_1);  mul_tensor_1 = arg166_1 = None
        add_tensor_2: "f32[128, 1370, 768]" = torch.ops.aten.add.Tensor(mul_tensor_2, arg167_1);  mul_tensor_2 = arg167_1 = None
        convert_element_type_default_1: "bf16[128, 1370, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None
        view_default_1: "bf16[175360, 768]" = torch.ops.aten.view.default(convert_element_type_default_1, _shape_param_1);  convert_element_type_default_1 = _shape_param_1 = None
        permute_default: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
        return (view_default_1, permute_default)


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
