"""
Standalone repro captured via capture_hook.
Label: YituTechConvBert
Pattern hash: e4090ceb0568
Shape hash: f0c2fbae
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
    def forward(self, arg0_1: "bf16[32, 6, 512, 64]", arg1_1: "bf16[32, 512, 768]", arg2_1: "bf16[384, 768]", arg3_1: "bf16[384]", arg4_1: "bf16[32, 384, 512]", arg5_1: "bf16[384, 1]", arg6_1: "bf16[32, 512, 384]", arg7_1: "bf16[54, 384]", arg8_1: "bf16[54]", arg9_1: "bf16[768, 768]", arg10_1: "bf16[768]", arg11_1: "bf16[768]", arg12_1: "bf16[768]", arg13_1: "bf16[3072, 768]", arg14_1: "bf16[3072]", arg15_1: "bf16[768, 3072]", arg16_1: "bf16[768]", arg17_1: "bf16[768]", arg18_1: "bf16[768]", arg19_1: "bf16[384, 768]", arg20_1: "bf16[384]", arg21_1: "bf16[384, 768]", arg22_1: "bf16[384]", arg23_1: "bf16[384, 768]", arg24_1: "bf16[384]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28):
        # No stacktrace found for following nodes
        permute: "bf16[32, 512, 6, 64]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
        clone: "bf16[32, 512, 6, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view: "bf16[16384, 768]" = torch.ops.aten.view.default(arg1_1, _shape_param_0);  _shape_param_0 = None
        permute_1: "bf16[768, 384]" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        addmm: "bf16[16384, 384]" = torch.ops.aten.addmm.default(arg3_1, view, permute_1);  arg3_1 = view = permute_1 = None
        view_1: "bf16[32, 512, 384]" = torch.ops.aten.view.default(addmm, _shape_param_1);  addmm = _shape_param_1 = None
        permute_2: "bf16[32, 384, 512]" = torch.ops.aten.permute.default(view_1, [0, 2, 1]);  view_1 = None
        clone_1: "bf16[32, 384, 512]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        unsqueeze: "bf16[32, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_1, -1);  clone_1 = None
        constant_pad_nd: "bf16[32, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze, _shape_param_2, 0.0);  unsqueeze = _shape_param_2 = None
        iota: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_1: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
        iota_1: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_2: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
        add: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_1, unsqueeze_2);  unsqueeze_1 = unsqueeze_2 = None
        unsqueeze_3: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add, -1);  add = None
        unsqueeze_4: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, -1);  unsqueeze_3 = None
        full: "i64[1, 1]" = torch.ops.aten.full.default([1, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index: "bf16[32, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(constant_pad_nd, [None, None, unsqueeze_4, full]);  constant_pad_nd = unsqueeze_4 = full = None
        permute_3: "bf16[32, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index, [0, 1, 2, 4, 3, 5]);  index = None
        view_2: "bf16[32, 3456, 512]" = torch.ops.aten.view.default(permute_3, _shape_param_3);  permute_3 = _shape_param_3 = None
        permute_4: "bf16[32, 512, 3456]" = torch.ops.aten.permute.default(view_2, [0, 2, 1]);  view_2 = None
        view_3: "bf16[32, 512, 384, 9]" = torch.ops.aten.view.default(permute_4, _shape_param_4);  permute_4 = _shape_param_4 = None
        clone_2: "bf16[32, 512, 384, 9]" = torch.ops.aten.clone.default(view_3, memory_format = torch.contiguous_format);  view_3 = None
        view_4: "bf16[98304, 64, 9]" = torch.ops.aten.view.default(clone_2, _shape_param_5);  clone_2 = _shape_param_5 = None
        expand: "bf16[98304, 64, 9]" = torch.ops.aten.expand.default(view_4, _shape_param_6);  view_4 = _shape_param_6 = None
        add_1: "bf16[32, 384, 512]" = torch.ops.aten.add.Tensor(arg4_1, arg5_1);  arg4_1 = arg5_1 = None
        permute_5: "bf16[32, 512, 384]" = torch.ops.aten.permute.default(add_1, [0, 2, 1]);  add_1 = None
        mul: "bf16[32, 512, 384]" = torch.ops.aten.mul.Tensor(permute_5, arg6_1);  permute_5 = arg6_1 = None
        clone_3: "bf16[32, 512, 384]" = torch.ops.aten.clone.default(mul, memory_format = torch.contiguous_format);  mul = None
        view_5: "bf16[16384, 384]" = torch.ops.aten.view.default(clone_3, _shape_param_7);  clone_3 = _shape_param_7 = None
        permute_6: "bf16[384, 54]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        mm: "bf16[16384, 54]" = torch.ops.aten.mm.default(view_5, permute_6);  view_5 = permute_6 = None
        view_6: "bf16[32, 512, 54]" = torch.ops.aten.view.default(mm, _shape_param_8);  mm = _shape_param_8 = None
        add_2: "bf16[32, 512, 54]" = torch.ops.aten.add.Tensor(view_6, arg8_1);  view_6 = arg8_1 = None
        view_7: "bf16[98304, 9, 1]" = torch.ops.aten.view.default(add_2, _shape_param_9);  add_2 = _shape_param_9 = None
        convert_element_type: "f32[98304, 9, 1]" = torch.ops.prims.convert_element_type.default(view_7, torch.float32);  view_7 = None
        amax: "f32[98304, 1, 1]" = torch.ops.aten.amax.default(convert_element_type, [1], True)
        sub: "f32[98304, 9, 1]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[98304, 9, 1]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[98304, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True)
        div: "f32[98304, 9, 1]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_1: "bf16[98304, 9, 1]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        expand_1: "bf16[98304, 9, 1]" = torch.ops.aten.expand.default(convert_element_type_1, _shape_param_10);  convert_element_type_1 = _shape_param_10 = None
        bmm: "bf16[98304, 64, 1]" = torch.ops.aten.bmm.default(expand, expand_1);  expand = expand_1 = None
        view_8: "bf16[16384, 384]" = torch.ops.aten.view.default(bmm, _shape_param_11);  bmm = _shape_param_11 = None
        view_9: "bf16[32, 512, 6, 64]" = torch.ops.aten.view.default(view_8, _shape_param_12);  view_8 = _shape_param_12 = None
        cat: "bf16[32, 512, 12, 64]" = torch.ops.aten.cat.default([clone, view_9], 2);  clone = view_9 = None
        view_10: "bf16[32, 512, 768]" = torch.ops.aten.view.default(cat, _shape_param_13);  cat = _shape_param_13 = None
        view_11: "bf16[16384, 768]" = torch.ops.aten.view.default(view_10, _shape_param_14);  view_10 = _shape_param_14 = None
        permute_7: "bf16[768, 768]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_1: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg10_1, view_11, permute_7);  arg10_1 = view_11 = permute_7 = None
        view_12: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_1, _shape_param_15);  addmm_1 = _shape_param_15 = None
        add_3: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(view_12, arg1_1);  view_12 = arg1_1 = None
        convert_element_type_2: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_3, torch.float32);  add_3 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_2, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean[0]
        getitem_1: "f32[32, 512, 1]" = var_mean[1];  var_mean = None
        sub_1: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_2, getitem_1);  convert_element_type_2 = getitem_1 = None
        add_4: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = rsqrt = None
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_1, arg11_1);  mul_1 = arg11_1 = None
        add_5: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_2, arg12_1);  mul_2 = arg12_1 = None
        convert_element_type_3: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16);  add_5 = None
        view_13: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_16);  _shape_param_16 = None
        permute_8: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_2: "bf16[16384, 3072]" = torch.ops.aten.addmm.default(arg14_1, view_13, permute_8);  arg14_1 = view_13 = permute_8 = None
        view_14: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(addmm_2, _shape_param_17);  addmm_2 = _shape_param_17 = None
        convert_element_type_4: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_14, torch.float32);  view_14 = None
        mul_3: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 0.5)
        mul_4: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 0.7071067811865476);  convert_element_type_4 = None
        erf: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_4);  mul_4 = None
        add_6: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_5: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_3, add_6);  mul_3 = add_6 = None
        convert_element_type_5: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        view_15: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_18);  convert_element_type_5 = _shape_param_18 = None
        permute_9: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_3: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg16_1, view_15, permute_9);  arg16_1 = view_15 = permute_9 = None
        view_16: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_3, _shape_param_19);  addmm_3 = _shape_param_19 = None
        add_7: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(view_16, convert_element_type_3);  view_16 = convert_element_type_3 = None
        convert_element_type_6: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_7, torch.float32);  add_7 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_6, [2], correction = 0, keepdim = True)
        getitem_2: "f32[32, 512, 1]" = var_mean_1[0]
        getitem_3: "f32[32, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_2: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_6, getitem_3);  convert_element_type_6 = getitem_3 = None
        add_8: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        mul_6: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_7: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_6, arg17_1);  mul_6 = arg17_1 = None
        add_9: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_7, arg18_1);  mul_7 = arg18_1 = None
        convert_element_type_7: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_9, torch.bfloat16);  add_9 = None
        view_17: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_7, _shape_param_20);  _shape_param_20 = None
        permute_10: "bf16[768, 384]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_4: "bf16[16384, 384]" = torch.ops.aten.addmm.default(arg20_1, view_17, permute_10);  arg20_1 = view_17 = permute_10 = None
        view_18: "bf16[32, 512, 384]" = torch.ops.aten.view.default(addmm_4, _shape_param_21);  addmm_4 = _shape_param_21 = None
        view_19: "bf16[32, 512, 6, 64]" = torch.ops.aten.view.default(view_18, _shape_param_22);  _shape_param_22 = None
        permute_11: "bf16[32, 6, 512, 64]" = torch.ops.aten.permute.default(view_19, [0, 2, 1, 3]);  view_19 = None
        view_20: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_7, _shape_param_23);  _shape_param_23 = None
        permute_12: "bf16[768, 384]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_5: "bf16[16384, 384]" = torch.ops.aten.addmm.default(arg22_1, view_20, permute_12);  arg22_1 = view_20 = permute_12 = None
        view_21: "bf16[32, 512, 384]" = torch.ops.aten.view.default(addmm_5, _shape_param_24);  addmm_5 = _shape_param_24 = None
        view_22: "bf16[32, 512, 6, 64]" = torch.ops.aten.view.default(view_21, _shape_param_25);  view_21 = _shape_param_25 = None
        permute_13: "bf16[32, 6, 512, 64]" = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
        view_23: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_7, _shape_param_26);  _shape_param_26 = None
        permute_14: "bf16[768, 384]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_6: "bf16[16384, 384]" = torch.ops.aten.addmm.default(arg24_1, view_23, permute_14);  arg24_1 = view_23 = permute_14 = None
        view_24: "bf16[32, 512, 384]" = torch.ops.aten.view.default(addmm_6, _shape_param_27);  addmm_6 = _shape_param_27 = None
        view_25: "bf16[32, 512, 6, 64]" = torch.ops.aten.view.default(view_24, _shape_param_28);  view_24 = _shape_param_28 = None
        permute_15: "bf16[32, 6, 512, 64]" = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
        permute_16: "bf16[32, 768, 512]" = torch.ops.aten.permute.default(convert_element_type_7, [0, 2, 1])
        return (convert_element_type_7, view_18, permute_11, permute_13, permute_15, permute_16)



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
