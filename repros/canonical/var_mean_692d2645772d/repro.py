"""
Standalone repro captured via capture_hook.
Label: timm_deit_base_distilled_patch16_224_train
Pattern hash: 692d2645772d
Shape hash: 1d6386ee
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
    def forward(self, arg0_1: "bf16[128, 768, 14, 14]", arg1_1: "f32[1, 1, 768]", arg2_1: "f32[1, 1, 768]", arg3_1: "f32[1, 198, 768]", arg4_1: "f32[768]", arg5_1: "f32[768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view: "bf16[128, 768, 196]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        permute: "bf16[128, 196, 768]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
        expand: "f32[128, 1, 768]" = torch.ops.aten.expand.default(arg1_1, _shape_param_1);  arg1_1 = _shape_param_1 = None
        expand_1: "f32[128, 1, 768]" = torch.ops.aten.expand.default(arg2_1, _shape_param_2);  arg2_1 = _shape_param_2 = None
        cat: "f32[128, 198, 768]" = torch.ops.aten.cat.default([expand, expand_1, permute], 1);  expand = expand_1 = permute = None
        add: "f32[128, 198, 768]" = torch.ops.aten.add.Tensor(cat, arg3_1);  arg3_1 = None
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 198, 1]" = var_mean[0]
        getitem_1: "f32[128, 198, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[128, 198, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt: "f32[128, 198, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[128, 198, 768]" = torch.ops.aten.sub.Tensor(add, getitem_1)
        mul: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[128, 198, 768]" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
        add_2: "f32[128, 198, 768]" = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
        convert_element_type: "bf16[128, 198, 768]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        view_1: "bf16[25344, 768]" = torch.ops.aten.view.default(convert_element_type, _shape_param_3);  convert_element_type = _shape_param_3 = None
        return (cat, add, getitem_1, rsqrt, view_1)



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
