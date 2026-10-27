"""
Standalone repro captured via capture_hook.
Label: beit_base_patch16_224
Pattern hash: fef83128d46f
Shape hash: 538246c2
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
    def forward(self, arg0_1: "bf16[128, 768, 14, 14]", arg1_1: "f32[1, 1, 768]", arg2_1: "f32[768]", arg3_1: "f32[768]", arg4_1: "f32[768]", arg5_1: "f32[768]", arg6_1: "f32[768]", arg7_1: "f32[2304, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view: "bf16[128, 768, 196]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        permute: "bf16[128, 196, 768]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
        expand: "f32[128, 1, 768]" = torch.ops.aten.expand.default(arg1_1, _shape_param_1);  arg1_1 = _shape_param_1 = None
        cat: "f32[128, 197, 768]" = torch.ops.aten.cat.default([expand, permute], 1);  expand = permute = None
        var_mean = torch.ops.aten.var_mean.correction(cat, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 197, 1]" = var_mean[0]
        getitem_1: "f32[128, 197, 1]" = var_mean[1];  var_mean = None
        add: "f32[128, 197, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt: "f32[128, 197, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[128, 197, 768]" = torch.ops.aten.sub.Tensor(cat, getitem_1)
        mul: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = arg2_1 = None
        add_1: "f32[128, 197, 768]" = torch.ops.aten.add.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        cat_1: "f32[2304]" = torch.ops.aten.cat.default([arg4_1, arg5_1, arg6_1]);  arg4_1 = arg5_1 = arg6_1 = None
        convert_element_type: "bf16[2304]" = torch.ops.prims.convert_element_type.default(cat_1, torch.bfloat16);  cat_1 = None
        convert_element_type_1: "bf16[2304, 768]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.bfloat16);  arg7_1 = None
        convert_element_type_2: "bf16[128, 197, 768]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        view_1: "bf16[25216, 768]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_2);  convert_element_type_2 = _shape_param_2 = None
        permute_1: "bf16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "bf16[25216, 2304]" = torch.ops.aten.addmm.default(convert_element_type, view_1, permute_1);  convert_element_type = None
        view_2: "bf16[128, 197, 2304]" = torch.ops.aten.view.default(addmm, _shape_param_3);  addmm = _shape_param_3 = None
        view_3: "bf16[128, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_2, _shape_param_4);  view_2 = _shape_param_4 = None
        permute_2: "bf16[3, 128, 12, 197, 64]" = torch.ops.aten.permute.default(view_3, [2, 0, 3, 1, 4]);  view_3 = None
        unbind = torch.ops.aten.unbind.int(permute_2);  permute_2 = None
        getitem_2: "bf16[128, 12, 197, 64]" = unbind[0]
        getitem_3: "bf16[128, 12, 197, 64]" = unbind[1]
        getitem_4: "bf16[128, 12, 197, 64]" = unbind[2];  unbind = None
        permute_3: "bf16[2304, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        return (cat, getitem_1, rsqrt, view_1, getitem_2, getitem_3, getitem_4, permute_3)



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
