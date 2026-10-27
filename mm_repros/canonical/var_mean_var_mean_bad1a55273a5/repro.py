"""
Standalone repro captured via capture_hook.
Label: whisper-tiny
Pattern hash: bad1a55273a5
Shape hash: a7c62b6c
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
    def forward(self, arg0_1: "bf16[1, 6, 1500, 64]", arg1_1: "bf16[384, 384]", arg2_1: "bf16[384]", arg3_1: "bf16[1, 1500, 384]", arg4_1: "bf16[384]", arg5_1: "bf16[384]", arg6_1: "bf16[1536, 384]", arg7_1: "bf16[1536]", arg8_1: "bf16[384, 1536]", arg9_1: "bf16[384]", arg10_1: "bf16[384]", arg11_1: "bf16[384]", arg12_1: "bf16[384, 384]", arg13_1: "bf16[384, 384]", arg14_1: "bf16[384]", arg15_1: "bf16[384, 384]", arg16_1: "bf16[384, 384]", arg17_1: "bf16[384]", arg18_1: "bf16[384, 384]", arg19_1: "bf16[384, 384]", arg20_1: "bf16[384]", arg21_1: "bf16[384, 384]", arg22_1: "bf16[384, 384]", arg23_1: "bf16[384]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30):
        # No stacktrace found for following nodes
        permute: "bf16[1, 1500, 6, 64]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
        clone: "bf16[1, 1500, 6, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view: "bf16[1, 1500, 384]" = torch.ops.aten.view.default(clone, _shape_param_0);  clone = _shape_param_0 = None
        view_1: "bf16[1500, 384]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute_1: "bf16[384, 384]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        addmm: "bf16[1500, 384]" = torch.ops.aten.addmm.default(arg2_1, view_1, permute_1);  arg2_1 = view_1 = permute_1 = None
        view_2: "bf16[1, 1500, 384]" = torch.ops.aten.view.default(addmm, _shape_param_2);  addmm = _shape_param_2 = None
        add: "bf16[1, 1500, 384]" = torch.ops.aten.add.Tensor(arg3_1, view_2);  arg3_1 = view_2 = None
        clone_1: "bf16[1, 1500, 384]" = torch.ops.aten.clone.default(add, memory_format = torch.contiguous_format)
        convert_element_type: "f32[1, 1500, 384]" = torch.ops.prims.convert_element_type.default(clone_1, torch.float32);  clone_1 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 1500, 1]" = var_mean[0]
        getitem_1: "f32[1, 1500, 1]" = var_mean[1];  var_mean = None
        sub: "f32[1, 1500, 384]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_1: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[1, 1500, 384]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[1, 1500, 384]" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
        add_2: "f32[1, 1500, 384]" = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
        convert_element_type_1: "bf16[1, 1500, 384]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        view_3: "bf16[1500, 384]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_3);  convert_element_type_1 = _shape_param_3 = None
        permute_2: "bf16[384, 1536]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        addmm_1: "bf16[1500, 1536]" = torch.ops.aten.addmm.default(arg7_1, view_3, permute_2);  arg7_1 = view_3 = permute_2 = None
        view_4: "bf16[1, 1500, 1536]" = torch.ops.aten.view.default(addmm_1, _shape_param_4);  addmm_1 = _shape_param_4 = None
        convert_element_type_2: "f32[1, 1500, 1536]" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        mul_2: "f32[1, 1500, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 0.5)
        mul_3: "f32[1, 1500, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 0.7071067811865476);  convert_element_type_2 = None
        erf: "f32[1, 1500, 1536]" = torch.ops.aten.erf.default(mul_3);  mul_3 = None
        add_3: "f32[1, 1500, 1536]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_4: "f32[1, 1500, 1536]" = torch.ops.aten.mul.Tensor(mul_2, add_3);  mul_2 = add_3 = None
        convert_element_type_3: "bf16[1, 1500, 1536]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        view_5: "bf16[1500, 1536]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_5);  convert_element_type_3 = _shape_param_5 = None
        permute_3: "bf16[1536, 384]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm_2: "bf16[1500, 384]" = torch.ops.aten.addmm.default(arg9_1, view_5, permute_3);  arg9_1 = view_5 = permute_3 = None
        view_6: "bf16[1, 1500, 384]" = torch.ops.aten.view.default(addmm_2, _shape_param_6);  addmm_2 = _shape_param_6 = None
        add_4: "bf16[1, 1500, 384]" = torch.ops.aten.add.Tensor(add, view_6);  add = view_6 = None
        clone_2: "bf16[1, 1500, 384]" = torch.ops.aten.clone.default(add_4, memory_format = torch.contiguous_format);  add_4 = None
        convert_element_type_4: "f32[1, 1500, 384]" = torch.ops.prims.convert_element_type.default(clone_2, torch.float32);  clone_2 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_4, [2], correction = 0, keepdim = True)
        getitem_2: "f32[1, 1500, 1]" = var_mean_1[0]
        getitem_3: "f32[1, 1500, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_1: "f32[1, 1500, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_4, getitem_3);  convert_element_type_4 = getitem_3 = None
        add_5: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        mul_5: "f32[1, 1500, 384]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_6: "f32[1, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_5, arg10_1);  mul_5 = arg10_1 = None
        add_6: "f32[1, 1500, 384]" = torch.ops.aten.add.Tensor(mul_6, arg11_1);  mul_6 = arg11_1 = None
        convert_element_type_5: "bf16[1, 1500, 384]" = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16);  add_6 = None
        view_7: "bf16[1500, 384]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_7);  _shape_param_7 = None
        permute_4: "bf16[384, 384]" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        mm: "bf16[1500, 384]" = torch.ops.aten.mm.default(view_7, permute_4);  view_7 = permute_4 = None
        view_8: "bf16[1, 1500, 384]" = torch.ops.aten.view.default(mm, _shape_param_8);  mm = _shape_param_8 = None
        view_9: "bf16[1, 1500, 6, 64]" = torch.ops.aten.view.default(view_8, _shape_param_9);  view_8 = _shape_param_9 = None
        permute_5: "bf16[1, 6, 1500, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
        clone_3: "bf16[1, 6, 1500, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_10: "bf16[1500, 384]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_10);  _shape_param_10 = None
        permute_6: "bf16[384, 384]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_3: "bf16[1500, 384]" = torch.ops.aten.addmm.default(arg14_1, view_10, permute_6);  arg14_1 = view_10 = permute_6 = None
        view_11: "bf16[1, 1500, 384]" = torch.ops.aten.view.default(addmm_3, _shape_param_11);  addmm_3 = _shape_param_11 = None
        view_12: "bf16[1, 1500, 6, 64]" = torch.ops.aten.view.default(view_11, _shape_param_12);  view_11 = _shape_param_12 = None
        permute_7: "bf16[1, 6, 1500, 64]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
        clone_4: "bf16[1, 6, 1500, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_13: "bf16[1500, 384]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_13);  _shape_param_13 = None
        permute_8: "bf16[384, 384]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        mm_1: "bf16[1500, 384]" = torch.ops.aten.mm.default(view_13, permute_8);  view_13 = permute_8 = None
        view_14: "bf16[1, 1500, 384]" = torch.ops.aten.view.default(mm_1, _shape_param_14);  mm_1 = _shape_param_14 = None
        view_15: "bf16[1, 1500, 6, 64]" = torch.ops.aten.view.default(view_14, _shape_param_15);  view_14 = _shape_param_15 = None
        permute_9: "bf16[1, 6, 1500, 64]" = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None
        clone_5: "bf16[1, 6, 1500, 64]" = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None
        view_16: "bf16[1500, 384]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_16);  _shape_param_16 = None
        permute_10: "bf16[384, 384]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        addmm_4: "bf16[1500, 384]" = torch.ops.aten.addmm.default(arg17_1, view_16, permute_10);  arg17_1 = view_16 = permute_10 = None
        view_17: "bf16[1, 1500, 384]" = torch.ops.aten.view.default(addmm_4, _shape_param_17);  addmm_4 = _shape_param_17 = None
        view_18: "bf16[1, 1500, 6, 64]" = torch.ops.aten.view.default(view_17, _shape_param_18);  view_17 = _shape_param_18 = None
        permute_11: "bf16[1, 6, 1500, 64]" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        clone_6: "bf16[1, 6, 1500, 64]" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
        view_19: "bf16[1500, 384]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_19);  _shape_param_19 = None
        permute_12: "bf16[384, 384]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        mm_2: "bf16[1500, 384]" = torch.ops.aten.mm.default(view_19, permute_12);  view_19 = permute_12 = None
        view_20: "bf16[1, 1500, 384]" = torch.ops.aten.view.default(mm_2, _shape_param_20);  mm_2 = _shape_param_20 = None
        view_21: "bf16[1, 1500, 6, 64]" = torch.ops.aten.view.default(view_20, _shape_param_21);  view_20 = _shape_param_21 = None
        permute_13: "bf16[1, 6, 1500, 64]" = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        clone_7: "bf16[1, 6, 1500, 64]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
        view_22: "bf16[1500, 384]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_22);  _shape_param_22 = None
        permute_14: "bf16[384, 384]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_5: "bf16[1500, 384]" = torch.ops.aten.addmm.default(arg20_1, view_22, permute_14);  arg20_1 = view_22 = permute_14 = None
        view_23: "bf16[1, 1500, 384]" = torch.ops.aten.view.default(addmm_5, _shape_param_23);  addmm_5 = _shape_param_23 = None
        view_24: "bf16[1, 1500, 6, 64]" = torch.ops.aten.view.default(view_23, _shape_param_24);  view_23 = _shape_param_24 = None
        permute_15: "bf16[1, 6, 1500, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        clone_8: "bf16[1, 6, 1500, 64]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
        view_25: "bf16[1500, 384]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_25);  _shape_param_25 = None
        permute_16: "bf16[384, 384]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        mm_3: "bf16[1500, 384]" = torch.ops.aten.mm.default(view_25, permute_16);  view_25 = permute_16 = None
        view_26: "bf16[1, 1500, 384]" = torch.ops.aten.view.default(mm_3, _shape_param_26);  mm_3 = _shape_param_26 = None
        view_27: "bf16[1, 1500, 6, 64]" = torch.ops.aten.view.default(view_26, _shape_param_27);  view_26 = _shape_param_27 = None
        permute_17: "bf16[1, 6, 1500, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None
        clone_9: "bf16[1, 6, 1500, 64]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
        view_28: "bf16[1500, 384]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_28);  _shape_param_28 = None
        permute_18: "bf16[384, 384]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        addmm_6: "bf16[1500, 384]" = torch.ops.aten.addmm.default(arg23_1, view_28, permute_18);  arg23_1 = view_28 = permute_18 = None
        view_29: "bf16[1, 1500, 384]" = torch.ops.aten.view.default(addmm_6, _shape_param_29);  addmm_6 = _shape_param_29 = None
        view_30: "bf16[1, 1500, 6, 64]" = torch.ops.aten.view.default(view_29, _shape_param_30);  view_29 = _shape_param_30 = None
        permute_19: "bf16[1, 6, 1500, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
        clone_10: "bf16[1, 6, 1500, 64]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
        return (convert_element_type_5, clone_3, clone_4, clone_5, clone_6, clone_7, clone_8, clone_9, clone_10)



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
