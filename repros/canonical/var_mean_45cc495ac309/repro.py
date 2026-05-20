"""
Standalone repro captured via capture_hook.
Label: inductor_timm_perf-4-6-linux.aws.a100_graph17
Pattern hash: 45cc495ac309
Shape hash: 841a5797
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([128, 1536, 1, 1], bf16), T([128, 1536, 9, 9], bf16), T([128, 1536, 9, 9], bf16), T([384, 1536, 1, 1], bf16), T([384, 1, 1, 1], bf16), S([1, 384, -1]), S([384, 1536, 1, 1]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_73: "bf16[128, 1536, 1, 1]", convolution_71: "bf16[128, 1536, 9, 9]", add_100: "bf16[128, 1536, 9, 9]", arg201_1: "bf16[384, 1536, 1, 1]", arg202_1: "bf16[384, 1, 1, 1]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        sigmoid_default: "bf16[128, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_73);  convolution_73 = None
        mul_tensor: "bf16[128, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(convolution_71, sigmoid_default);  convolution_71 = sigmoid_default = None
        mul_tensor_1: "bf16[128, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(mul_tensor, 2.0);  mul_tensor = None
        mul_tensor_2: "bf16[128, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(mul_tensor_1, 0.2);  mul_tensor_1 = None
        add_tensor: "bf16[128, 1536, 9, 9]" = torch.ops.aten.add.Tensor(mul_tensor_2, add_100);  mul_tensor_2 = add_100 = None
        convert_element_type_default: "f32[128, 1536, 9, 9]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        neg_default: "f32[128, 1536, 9, 9]" = torch.ops.aten.neg.default(convert_element_type_default)
        exp_default: "f32[128, 1536, 9, 9]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_1: "f32[128, 1536, 9, 9]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[128, 1536, 9, 9]" = torch.ops.aten.div.Tensor(convert_element_type_default, add_tensor_1);  convert_element_type_default = add_tensor_1 = None
        convert_element_type_default_1: "bf16[128, 1536, 9, 9]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        mul_tensor_3: "bf16[128, 1536, 9, 9]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 0.9622504486493761);  convert_element_type_default_1 = None
        view_default: "bf16[1, 384, 1536]" = torch.ops.aten.view.default(arg201_1, _shape_param_0);  arg201_1 = _shape_param_0 = None
        mul_tensor_4: "bf16[384, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg202_1, 0.04562504637317021);  arg202_1 = None
        view_default_1: "bf16[384]" = torch.ops.aten.view.default(mul_tensor_4, [-1]);  mul_tensor_4 = None
        convert_element_type_default_2: "f32[1, 384, 1536]" = torch.ops.prims.convert_element_type.default(view_default, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default_2, [0, 2], correction = 0, keepdim = True);  convert_element_type_default_2 = None
        getitem: "f32[1, 384, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 384, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_2: "f32[1, 384, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 384, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor: "f32[1, 384, 1536]" = torch.ops.aten.sub.Tensor(view_default, getitem_1);  view_default = getitem_1 = None
        mul_tensor_5: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "bf16[384, 1]" = torch.ops.aten.unsqueeze.default(view_default_1, -1);  view_default_1 = None
        mul_tensor_6: "f32[1, 384, 1536]" = torch.ops.aten.mul.Tensor(mul_tensor_5, unsqueeze_default);  mul_tensor_5 = unsqueeze_default = None
        convert_element_type_default_3: "bf16[1, 384, 1536]" = torch.ops.prims.convert_element_type.default(mul_tensor_6, torch.bfloat16);  mul_tensor_6 = None
        view_default_2: "bf16[384, 1536, 1, 1]" = torch.ops.aten.view.default(convert_element_type_default_3, _shape_param_1);  convert_element_type_default_3 = _shape_param_1 = None
        return (mul_tensor_3, view_default_2)


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
