"""
Standalone repro captured via capture_hook.
Label: convnextv2_nano.fcmae_ft_in22k_in1k
Pattern hash: 2f604c8da9a8
Shape hash: 6e84df56
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[128, 640, 7, 7]", arg1_1: "bf16[128, 640, 7, 7]", arg2_1: "f32[640]", arg3_1: "f32[640]", arg4_1: "f32[1000]", arg5_1: "f32[1000, 640]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        add: "bf16[128, 640, 7, 7]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        mean: "bf16[128, 640, 1, 1]" = torch.ops.aten.mean.dim(add, [-1, -2], True);  add = None
        as_strided: "bf16[128, 640, 1, 1]" = torch.ops.aten.as_strided.default(mean, _shape_param_0, _shape_param_1);  mean = _shape_param_0 = _shape_param_1 = None
        permute: "bf16[128, 1, 1, 640]" = torch.ops.aten.permute.default(as_strided, [0, 2, 3, 1]);  as_strided = None
        convert_element_type: "f32[128, 1, 1, 640]" = torch.ops.prims.convert_element_type.default(permute, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [3], correction = 0, keepdim = True)
        getitem: "f32[128, 1, 1, 1]" = var_mean[0]
        getitem_1: "f32[128, 1, 1, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[128, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt: "f32[128, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[128, 1, 1, 640]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = None
        mul: "f32[128, 1, 1, 640]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[128, 1, 1, 640]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = arg2_1 = None
        add_2: "f32[128, 1, 1, 640]" = torch.ops.aten.add.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        permute_1: "f32[128, 640, 1, 1]" = torch.ops.aten.permute.default(add_2, [0, 3, 1, 2]);  add_2 = None
        view: "f32[128, 640]" = torch.ops.aten.view.default(permute_1, _shape_param_2);  permute_1 = _shape_param_2 = None
        convert_element_type_1: "bf16[1000]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.bfloat16);  arg4_1 = None
        convert_element_type_2: "bf16[1000, 640]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        convert_element_type_3: "bf16[128, 640]" = torch.ops.prims.convert_element_type.default(view, torch.bfloat16);  view = None
        permute_2: "bf16[640, 1000]" = torch.ops.aten.permute.default(convert_element_type_2, [1, 0]);  convert_element_type_2 = None
        addmm: "bf16[128, 1000]" = torch.ops.aten.addmm.default(convert_element_type_1, convert_element_type_3, permute_2);  convert_element_type_1 = None
        permute_3: "bf16[1000, 640]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        return (permute, getitem_1, rsqrt, convert_element_type_3, addmm, permute_3)



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
