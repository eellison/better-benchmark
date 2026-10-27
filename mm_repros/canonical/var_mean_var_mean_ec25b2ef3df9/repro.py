"""
Standalone repro captured via capture_hook.
Label: mobilevit_s
Pattern hash: ec25b2ef3df9
Shape hash: 460b5352
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
    def forward(self, arg0_1: "bf16[512, 4, 64, 48]", arg1_1: "f32[192]", arg2_1: "f32[192, 192]", arg3_1: "bf16[512, 64, 192]", arg4_1: "f32[192]", arg5_1: "f32[192]", arg6_1: "f32[384]", arg7_1: "f32[384, 192]", arg8_1: "f32[192]", arg9_1: "f32[192, 384]", arg10_1: "f32[192]", arg11_1: "f32[192]", arg12_1: "f32[576]", arg13_1: "f32[576, 192]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9):
        # No stacktrace found for following nodes
        permute: "bf16[512, 64, 4, 48]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
        view: "bf16[512, 64, 192]" = torch.ops.aten.view.default(permute, _shape_param_0);  permute = _shape_param_0 = None
        convert_element_type: "bf16[192]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        convert_element_type_1: "bf16[192, 192]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        view_1: "bf16[32768, 192]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute_1: "bf16[192, 192]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "bf16[32768, 192]" = torch.ops.aten.addmm.default(convert_element_type, view_1, permute_1);  convert_element_type = view_1 = None
        view_2: "bf16[512, 64, 192]" = torch.ops.aten.view.default(addmm, _shape_param_2);  addmm = _shape_param_2 = None
        add: "bf16[512, 64, 192]" = torch.ops.aten.add.Tensor(arg3_1, view_2);  arg3_1 = view_2 = None
        convert_element_type_2: "f32[512, 64, 192]" = torch.ops.prims.convert_element_type.default(add, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_2, [2], correction = 0, keepdim = True)
        getitem: "f32[512, 64, 1]" = var_mean[0]
        getitem_1: "f32[512, 64, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[512, 64, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[512, 64, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[512, 64, 192]" = torch.ops.aten.sub.Tensor(convert_element_type_2, getitem_1);  convert_element_type_2 = None
        mul: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
        add_2: "f32[512, 64, 192]" = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
        convert_element_type_3: "bf16[384]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.bfloat16);  arg6_1 = None
        convert_element_type_4: "bf16[384, 192]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.bfloat16);  arg7_1 = None
        convert_element_type_5: "bf16[512, 64, 192]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        view_3: "bf16[32768, 192]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_3);  convert_element_type_5 = _shape_param_3 = None
        permute_2: "bf16[192, 384]" = torch.ops.aten.permute.default(convert_element_type_4, [1, 0]);  convert_element_type_4 = None
        addmm_1: "bf16[32768, 384]" = torch.ops.aten.addmm.default(convert_element_type_3, view_3, permute_2);  convert_element_type_3 = None
        view_4: "bf16[512, 64, 384]" = torch.ops.aten.view.default(addmm_1, _shape_param_4);  _shape_param_4 = None
        convert_element_type_6: "f32[512, 64, 384]" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        neg: "f32[512, 64, 384]" = torch.ops.aten.neg.default(convert_element_type_6)
        exp: "f32[512, 64, 384]" = torch.ops.aten.exp.default(neg);  neg = None
        add_3: "f32[512, 64, 384]" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[512, 64, 384]" = torch.ops.aten.div.Tensor(convert_element_type_6, add_3);  convert_element_type_6 = add_3 = None
        convert_element_type_7: "bf16[512, 64, 384]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        convert_element_type_8: "bf16[192]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.bfloat16);  arg8_1 = None
        convert_element_type_9: "bf16[192, 384]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.bfloat16);  arg9_1 = None
        view_5: "bf16[32768, 384]" = torch.ops.aten.view.default(convert_element_type_7, _shape_param_5);  convert_element_type_7 = _shape_param_5 = None
        permute_3: "bf16[384, 192]" = torch.ops.aten.permute.default(convert_element_type_9, [1, 0]);  convert_element_type_9 = None
        addmm_2: "bf16[32768, 192]" = torch.ops.aten.addmm.default(convert_element_type_8, view_5, permute_3);  convert_element_type_8 = None
        view_6: "bf16[512, 64, 192]" = torch.ops.aten.view.default(addmm_2, _shape_param_6);  addmm_2 = _shape_param_6 = None
        add_4: "bf16[512, 64, 192]" = torch.ops.aten.add.Tensor(add, view_6);  view_6 = None
        convert_element_type_10: "f32[512, 64, 192]" = torch.ops.prims.convert_element_type.default(add_4, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_10, [2], correction = 0, keepdim = True)
        getitem_2: "f32[512, 64, 1]" = var_mean_1[0]
        getitem_3: "f32[512, 64, 1]" = var_mean_1[1];  var_mean_1 = None
        add_5: "f32[512, 64, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[512, 64, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_1: "f32[512, 64, 192]" = torch.ops.aten.sub.Tensor(convert_element_type_10, getitem_3);  convert_element_type_10 = None
        mul_2: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_3: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul_2, arg10_1);  mul_2 = arg10_1 = None
        add_6: "f32[512, 64, 192]" = torch.ops.aten.add.Tensor(mul_3, arg11_1);  mul_3 = arg11_1 = None
        convert_element_type_11: "bf16[576]" = torch.ops.prims.convert_element_type.default(arg12_1, torch.bfloat16);  arg12_1 = None
        convert_element_type_12: "bf16[576, 192]" = torch.ops.prims.convert_element_type.default(arg13_1, torch.bfloat16);  arg13_1 = None
        convert_element_type_13: "bf16[512, 64, 192]" = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16);  add_6 = None
        view_7: "bf16[32768, 192]" = torch.ops.aten.view.default(convert_element_type_13, _shape_param_7);  convert_element_type_13 = _shape_param_7 = None
        permute_4: "bf16[192, 576]" = torch.ops.aten.permute.default(convert_element_type_12, [1, 0]);  convert_element_type_12 = None
        addmm_3: "bf16[32768, 576]" = torch.ops.aten.addmm.default(convert_element_type_11, view_7, permute_4);  convert_element_type_11 = None
        view_8: "bf16[512, 64, 576]" = torch.ops.aten.view.default(addmm_3, _shape_param_8);  addmm_3 = _shape_param_8 = None
        view_9: "bf16[512, 64, 3, 4, 48]" = torch.ops.aten.view.default(view_8, _shape_param_9);  view_8 = _shape_param_9 = None
        permute_5: "bf16[3, 512, 4, 64, 48]" = torch.ops.aten.permute.default(view_9, [2, 0, 3, 1, 4]);  view_9 = None
        unbind = torch.ops.aten.unbind.int(permute_5);  permute_5 = None
        getitem_4: "bf16[512, 4, 64, 48]" = unbind[0]
        getitem_5: "bf16[512, 4, 64, 48]" = unbind[1]
        getitem_6: "bf16[512, 4, 64, 48]" = unbind[2];  unbind = None
        permute_6: "bf16[576, 192]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        permute_7: "bf16[192, 384]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        permute_8: "bf16[384, 192]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_9: "bf16[192, 192]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        return (add, getitem_1, rsqrt, view_3, addmm_1, view_5, add_4, getitem_3, rsqrt_1, view_7, getitem_4, getitem_5, getitem_6, permute_6, permute_7, permute_8, permute_9)



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
