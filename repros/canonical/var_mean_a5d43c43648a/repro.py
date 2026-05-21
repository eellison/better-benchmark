"""
Standalone repro captured via capture_hook.
Label: torchbench_timm_vision_transformer_infer_000
Pattern hash: a5d43c43648a
Shape hash: 6d6ef27f
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
_shapes_config = "(T([1, 1, 384], f16), T([32, 384, 14, 14], f16), T([1, 197, 384], f16), T([384], f16), T([384], f16), S([32, -1, -1]), S([32, 384, 196]), S([6304, 384]))"

class Repro(torch.nn.Module):
    def forward(self, arg3_1: "f16[1, 1, 384]", convolution: "f16[32, 384, 14, 14]", arg4_1: "f16[1, 197, 384]", arg5_1: "f16[384]", arg6_1: "f16[384]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        expand_default: "f16[32, 1, 384]" = torch.ops.aten.expand.default(arg3_1, _shape_param_0);  arg3_1 = _shape_param_0 = None
        view_default: "f16[32, 384, 196]" = torch.ops.aten.view.default(convolution, _shape_param_1);  convolution = _shape_param_1 = None
        permute_default: "f16[32, 196, 384]" = torch.ops.aten.permute.default(view_default, [0, 2, 1]);  view_default = None
        cat_default: "f16[32, 197, 384]" = torch.ops.aten.cat.default([expand_default, permute_default], 1);  expand_default = permute_default = None
        add_tensor: "f16[32, 197, 384]" = torch.ops.aten.add.Tensor(cat_default, arg4_1);  cat_default = arg4_1 = None
        convert_element_type_default: "f32[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 197, 1]" = var_mean_correction[0]
        getitem_1: "f32[32, 197, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[32, 197, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1);  convert_element_type_default = getitem_1 = None
        add_tensor_1: "f32[32, 197, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt_default: "f32[32, 197, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[32, 197, 384]" = torch.ops.aten.mul.Tensor(mul_tensor, arg5_1);  mul_tensor = arg5_1 = None
        add_tensor_2: "f32[32, 197, 384]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg6_1);  mul_tensor_1 = arg6_1 = None
        convert_element_type_default_1: "f16[32, 197, 384]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float16);  add_tensor_2 = None
        view_default_1: "f16[6304, 384]" = torch.ops.aten.view.default(convert_element_type_default_1, _shape_param_2);  convert_element_type_default_1 = _shape_param_2 = None
        return view_default_1



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
