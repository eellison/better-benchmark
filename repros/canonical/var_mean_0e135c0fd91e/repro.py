"""
Standalone repro captured via capture_hook.
Label: inductor_timm_perf-2-6-linux.aws.a100_graph15
Pattern hash: 0e135c0fd91e
Shape hash: 518fd513
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([128, 3, 256, 256], bf16), T([16, 3, 3, 3], bf16), T([16, 1, 1, 1], bf16), S([1, 16, -1]), S([16, 3, 3, 3]))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[128, 3, 256, 256]", arg1_1: "bf16[16, 3, 3, 3]", arg2_1: "bf16[16, 1, 1, 1]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        constant_pad_nd_default: "bf16[128, 3, 257, 257]" = torch.ops.aten.constant_pad_nd.default(arg0_1, [0, 1, 0, 1], 0.0);  arg0_1 = None
        view_default: "bf16[1, 16, 27]" = torch.ops.aten.view.default(arg1_1, _shape_param_0);  arg1_1 = _shape_param_0 = None
        mul_tensor: "bf16[16, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg2_1, 0.19245008972987526);  arg2_1 = None
        view_default_1: "bf16[16]" = torch.ops.aten.view.default(mul_tensor, [-1]);  mul_tensor = None
        convert_element_type_default: "f32[1, 16, 27]" = torch.ops.prims.convert_element_type.default(view_default, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [0, 2], correction = 0, keepdim = True);  convert_element_type_default = None
        getitem: "f32[1, 16, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 16, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[1, 16, 27]" = torch.ops.aten.sub.Tensor(view_default, getitem_1);  view_default = getitem_1 = None
        mul_tensor_1: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "bf16[16, 1]" = torch.ops.aten.unsqueeze.default(view_default_1, -1);  view_default_1 = None
        mul_tensor_2: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default);  mul_tensor_1 = unsqueeze_default = None
        convert_element_type_default_1: "bf16[1, 16, 27]" = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.bfloat16);  mul_tensor_2 = None
        view_default_2: "bf16[16, 3, 3, 3]" = torch.ops.aten.view.default(convert_element_type_default_1, _shape_param_1);  convert_element_type_default_1 = _shape_param_1 = None
        return (constant_pad_nd_default, view_default_2)


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
