"""
Standalone repro captured via capture_hook.
Label: inductor_timm_perf_cuda_h100-2-7-linux.aws.h100_graph49
Pattern hash: a44841f49721
Shape hash: b0201862
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([1584, 768], f16), T([8, 198, 768], f32), T([768], f32), T([768], f32), T([3072], f32), T([3072, 768], f32), S([8, 198, 768]), S([1584, 768]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_45: "f16[1584, 768]", add_77: "f32[8, 198, 768]", arg144_1: "f32[768]", arg145_1: "f32[768]", arg147_1: "f32[3072]", arg146_1: "f32[3072, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f16[8, 198, 768]" = torch.ops.aten.view.default(addmm_45, _shape_param_0);  addmm_45 = _shape_param_0 = None
        add_tensor: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_77, view_default);  add_77 = view_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 198, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 198, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt_default: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg144_1);  mul_tensor = arg144_1 = None
        add_tensor_2: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg145_1);  mul_tensor_1 = arg145_1 = None
        convert_element_type_default: "f16[3072]" = torch.ops.prims.convert_element_type.default(arg147_1, torch.float16);  arg147_1 = None
        convert_element_type_default_1: "f16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg146_1, torch.float16);  arg146_1 = None
        convert_element_type_default_2: "f16[8, 198, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float16);  add_tensor_2 = None
        view_default_1: "f16[1584, 768]" = torch.ops.aten.view.default(convert_element_type_default_2, _shape_param_1);  convert_element_type_default_2 = _shape_param_1 = None
        permute_default: "f16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        return (convert_element_type_default, view_default_1, permute_default)


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
