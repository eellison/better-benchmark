"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-3-5-linux.aws.a100_graph7
Pattern hash: 72a919b4543c
Shape hash: cc7232f4
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([512, 768], f16), T([1, 512, 768], f32), T([768], f32), T([768], f32), T([768], f32), T([768, 768], f32), S([1, 512, 768]), S([512, 768]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_71: "f16[512, 768]", add_89: "f32[1, 512, 768]", arg201_1: "f32[768]", arg202_1: "f32[768]", arg206_1: "f32[768]", arg205_1: "f32[768, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f16[1, 512, 768]" = torch.ops.aten.view.default(addmm_71, _shape_param_0);  addmm_71 = _shape_param_0 = None
        native_dropout_default = torch.ops.aten.native_dropout.default(view_default, 0.1, True);  view_default = None
        getitem: "f16[1, 512, 768]" = native_dropout_default[0]
        getitem_1: "b8[1, 512, 768]" = native_dropout_default[1];  native_dropout_default = None
        add_tensor: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem, add_89);  getitem = add_89 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem_2: "f32[1, 512, 1]" = var_mean_correction[0]
        getitem_3: "f32[1, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_default: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_3);  add_tensor = getitem_3 = None
        mul_tensor: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg201_1);  mul_tensor = arg201_1 = None
        add_tensor_2: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg202_1);  mul_tensor_1 = arg202_1 = None
        convert_element_type_default: "f16[768]" = torch.ops.prims.convert_element_type.default(arg206_1, torch.float16);  arg206_1 = None
        convert_element_type_default_1: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(arg205_1, torch.float16);  arg205_1 = None
        convert_element_type_default_2: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float16);  add_tensor_2 = None
        view_default_1: "f16[512, 768]" = torch.ops.aten.view.default(convert_element_type_default_2, _shape_param_1);  convert_element_type_default_2 = _shape_param_1 = None
        permute_default: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        return (convert_element_type_default, view_default_1, permute_default, getitem_1)


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
