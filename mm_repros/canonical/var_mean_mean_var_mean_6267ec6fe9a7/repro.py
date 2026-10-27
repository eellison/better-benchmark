"""
Standalone repro captured via capture_hook.
Label: beit_base_patch16_224
Pattern hash: 6267ec6fe9a7
Shape hash: 4f4093d6
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
    def forward(self, arg0_1: "bf16[128, 12, 197, 64]", arg1_1: "f32[768]", arg2_1: "f32[768, 768]", arg3_1: "f32[768]", arg4_1: "f32[128, 197, 768]", arg5_1: "f32[768]", arg6_1: "f32[768]", arg7_1: "f32[3072]", arg8_1: "f32[3072, 768]", arg9_1: "f32[768]", arg10_1: "f32[768, 3072]", arg11_1: "f32[768]", arg12_1: "f32[768]", arg13_1: "f32[768]", arg14_1: "f32[1000]", arg15_1: "f32[1000, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        permute: "bf16[128, 197, 12, 64]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
        view: "bf16[128, 197, 768]" = torch.ops.aten.view.default(permute, _shape_param_0);  permute = _shape_param_0 = None
        convert_element_type: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        convert_element_type_1: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        view_1: "bf16[25216, 768]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute_1: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "bf16[25216, 768]" = torch.ops.aten.addmm.default(convert_element_type, view_1, permute_1);  convert_element_type = view_1 = None
        view_2: "bf16[128, 197, 768]" = torch.ops.aten.view.default(addmm, _shape_param_2);  _shape_param_2 = None
        mul: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(arg3_1, view_2);  arg3_1 = view_2 = None
        add: "f32[128, 197, 768]" = torch.ops.aten.add.Tensor(arg4_1, mul);  arg4_1 = mul = None
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 197, 1]" = var_mean[0]
        getitem_1: "f32[128, 197, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[128, 197, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt: "f32[128, 197, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[128, 197, 768]" = torch.ops.aten.sub.Tensor(add, getitem_1);  getitem_1 = None
        mul_1: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_2: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(mul_1, arg5_1);  arg5_1 = None
        add_2: "f32[128, 197, 768]" = torch.ops.aten.add.Tensor(mul_2, arg6_1);  mul_2 = arg6_1 = None
        convert_element_type_2: "bf16[3072]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.bfloat16);  arg7_1 = None
        convert_element_type_3: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.bfloat16);  arg8_1 = None
        convert_element_type_4: "bf16[128, 197, 768]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        view_3: "bf16[25216, 768]" = torch.ops.aten.view.default(convert_element_type_4, _shape_param_3);  convert_element_type_4 = _shape_param_3 = None
        permute_2: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_3, [1, 0]);  convert_element_type_3 = None
        addmm_1: "bf16[25216, 3072]" = torch.ops.aten.addmm.default(convert_element_type_2, view_3, permute_2);  convert_element_type_2 = None
        view_4: "bf16[128, 197, 3072]" = torch.ops.aten.view.default(addmm_1, _shape_param_4);  _shape_param_4 = None
        convert_element_type_5: "f32[128, 197, 3072]" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        mul_3: "f32[128, 197, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 0.5)
        mul_4: "f32[128, 197, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 0.7071067811865476);  convert_element_type_5 = None
        erf: "f32[128, 197, 3072]" = torch.ops.aten.erf.default(mul_4);  mul_4 = None
        add_3: "f32[128, 197, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_5: "f32[128, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_3, add_3);  mul_3 = add_3 = None
        convert_element_type_6: "bf16[128, 197, 3072]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        convert_element_type_7: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.bfloat16);  arg9_1 = None
        convert_element_type_8: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg10_1, torch.bfloat16);  arg10_1 = None
        view_5: "bf16[25216, 3072]" = torch.ops.aten.view.default(convert_element_type_6, _shape_param_5);  convert_element_type_6 = _shape_param_5 = None
        permute_3: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_8, [1, 0]);  convert_element_type_8 = None
        addmm_2: "bf16[25216, 768]" = torch.ops.aten.addmm.default(convert_element_type_7, view_5, permute_3);  convert_element_type_7 = None
        view_6: "bf16[128, 197, 768]" = torch.ops.aten.view.default(addmm_2, _shape_param_6);  _shape_param_6 = None
        mul_6: "f32[128, 197, 768]" = torch.ops.aten.mul.Tensor(arg11_1, view_6);  arg11_1 = view_6 = None
        add_4: "f32[128, 197, 768]" = torch.ops.aten.add.Tensor(add, mul_6);  add = mul_6 = None
        slice_1: "f32[128, 196, 768]" = torch.ops.aten.slice.Tensor(add_4, 1, 1, 9223372036854775807);  add_4 = None
        mean: "f32[128, 768]" = torch.ops.aten.mean.dim(slice_1, [1]);  slice_1 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(mean, [1], correction = 0, keepdim = True)
        getitem_2: "f32[128, 1]" = var_mean_1[0]
        getitem_3: "f32[128, 1]" = var_mean_1[1];  var_mean_1 = None
        add_5: "f32[128, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-06);  getitem_2 = None
        rsqrt_1: "f32[128, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_1: "f32[128, 768]" = torch.ops.aten.sub.Tensor(mean, getitem_3);  mean = getitem_3 = None
        mul_7: "f32[128, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_8: "f32[128, 768]" = torch.ops.aten.mul.Tensor(mul_7, arg12_1);  arg12_1 = None
        add_6: "f32[128, 768]" = torch.ops.aten.add.Tensor(mul_8, arg13_1);  mul_8 = arg13_1 = None
        convert_element_type_9: "bf16[1000]" = torch.ops.prims.convert_element_type.default(arg14_1, torch.bfloat16);  arg14_1 = None
        convert_element_type_10: "bf16[1000, 768]" = torch.ops.prims.convert_element_type.default(arg15_1, torch.bfloat16);  arg15_1 = None
        convert_element_type_11: "bf16[128, 768]" = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16);  add_6 = None
        permute_4: "bf16[768, 1000]" = torch.ops.aten.permute.default(convert_element_type_10, [1, 0]);  convert_element_type_10 = None
        addmm_3: "bf16[128, 1000]" = torch.ops.aten.addmm.default(convert_element_type_9, convert_element_type_11, permute_4);  convert_element_type_9 = None
        permute_5: "bf16[1000, 768]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        div: "f32[128, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
        permute_6: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        permute_7: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        div_1: "f32[128, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        permute_8: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        return (addmm, mul_1, view_3, addmm_1, view_5, addmm_2, mul_7, convert_element_type_11, addmm_3, permute_5, div, permute_6, permute_7, div_1, permute_8)



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
