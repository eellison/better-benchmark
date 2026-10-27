"""
Standalone repro captured via capture_hook.
Label: LayoutLMForMaskedLM
Pattern hash: 46667ecf6942
Shape hash: 4ed11eee
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
    def forward(self, arg0_1: "bf16[32, 12, 512, 64]", arg1_1: "bf16[768, 768]", arg2_1: "bf16[768]", arg3_1: "bf16[32, 512, 768]", arg4_1: "bf16[768]", arg5_1: "bf16[768]", arg6_1: "bf16[3072, 768]", arg7_1: "bf16[3072]", arg8_1: "bf16[768, 3072]", arg9_1: "bf16[768]", arg10_1: "bf16[768]", arg11_1: "bf16[768]", arg12_1: "bf16[768, 768]", arg13_1: "bf16[768]", arg14_1: "i64[32, 512]", arg15_1: "bf16[30522]", arg16_1: "bf16[768]", arg17_1: "bf16[768]", arg18_1: "bf16[30522, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13):
        # No stacktrace found for following nodes
        permute: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
        clone: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone, _shape_param_0);  clone = _shape_param_0 = None
        view_1: "bf16[16384, 768]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute_1: "bf16[768, 768]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        addmm: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg2_1, view_1, permute_1);  arg2_1 = view_1 = permute_1 = None
        view_2: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm, _shape_param_2);  addmm = _shape_param_2 = None
        add: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(view_2, arg3_1);  view_2 = arg3_1 = None
        convert_element_type: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean[0]
        getitem_1: "f32[32, 512, 1]" = var_mean[1];  var_mean = None
        sub: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_1: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
        add_2: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
        convert_element_type_1: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        view_3: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_3);  _shape_param_3 = None
        permute_2: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        addmm_1: "bf16[16384, 3072]" = torch.ops.aten.addmm.default(arg7_1, view_3, permute_2);  arg7_1 = view_3 = permute_2 = None
        view_4: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(addmm_1, _shape_param_4);  addmm_1 = _shape_param_4 = None
        convert_element_type_2: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        mul_2: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 0.5)
        mul_3: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 0.7071067811865476);  convert_element_type_2 = None
        erf: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_3);  mul_3 = None
        add_3: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_4: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_2, add_3);  mul_2 = add_3 = None
        convert_element_type_3: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        view_5: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_5);  convert_element_type_3 = _shape_param_5 = None
        permute_3: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm_2: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg9_1, view_5, permute_3);  arg9_1 = view_5 = permute_3 = None
        view_6: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_2, _shape_param_6);  addmm_2 = _shape_param_6 = None
        add_4: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(view_6, convert_element_type_1);  view_6 = convert_element_type_1 = None
        convert_element_type_4: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_4, torch.float32);  add_4 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_4, [2], correction = 0, keepdim = True)
        getitem_2: "f32[32, 512, 1]" = var_mean_1[0]
        getitem_3: "f32[32, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_1: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_4, getitem_3);  convert_element_type_4 = getitem_3 = None
        add_5: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        mul_5: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_6: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_5, arg10_1);  mul_5 = arg10_1 = None
        add_6: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_6, arg11_1);  mul_6 = arg11_1 = None
        convert_element_type_5: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16);  add_6 = None
        view_7: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_7);  convert_element_type_5 = _shape_param_7 = None
        permute_4: "bf16[768, 768]" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        addmm_3: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg13_1, view_7, permute_4);  arg13_1 = view_7 = permute_4 = None
        view_8: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_3, _shape_param_8);  addmm_3 = _shape_param_8 = None
        convert_element_type_6: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_8, torch.float32);  view_8 = None
        mul_7: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.5)
        mul_8: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.7071067811865476);  convert_element_type_6 = None
        erf_1: "f32[32, 512, 768]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
        add_7: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_9: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_7, add_7);  mul_7 = add_7 = None
        convert_element_type_7: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
        convert_element_type_8: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_7, torch.float32);  convert_element_type_7 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_8, [2], correction = 0, keepdim = True)
        getitem_4: "f32[32, 512, 1]" = var_mean_2[0]
        getitem_5: "f32[32, 512, 1]" = var_mean_2[1];  var_mean_2 = None
        view_9: "i64[16384]" = torch.ops.aten.view.default(arg14_1, [-1]);  arg14_1 = None
        ne: "b8[16384]" = torch.ops.aten.ne.Scalar(view_9, -100)
        full: "bf16[6]" = torch.ops.aten.full.default(_shape_param_9, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_9 = None
        cat: "bf16[30528]" = torch.ops.aten.cat.default([arg15_1, full]);  arg15_1 = full = None
        sub_2: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_8, getitem_5);  convert_element_type_8 = getitem_5 = None
        add_8: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        mul_10: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        mul_11: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_10, arg16_1);  mul_10 = arg16_1 = None
        add_9: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_11, arg17_1);  mul_11 = arg17_1 = None
        convert_element_type_9: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_9, torch.bfloat16);  add_9 = None
        view_10: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_9, _shape_param_10);  convert_element_type_9 = _shape_param_10 = None
        permute_5: "bf16[768, 30522]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        constant_pad_nd: "bf16[768, 30528]" = torch.ops.aten.constant_pad_nd.default(permute_5, _shape_param_11);  permute_5 = _shape_param_11 = None
        addmm_4: "bf16[16384, 30528]" = torch.ops.aten.addmm.default(cat, view_10, constant_pad_nd);  cat = view_10 = constant_pad_nd = None
        slice_1: "bf16[16384, 30522]" = torch.ops.aten.slice.Tensor(addmm_4, 1, 0, -6);  addmm_4 = None
        view_11: "bf16[32, 512, 30522]" = torch.ops.aten.view.default(slice_1, _shape_param_12);  slice_1 = _shape_param_12 = None
        view_12: "bf16[16384, 30522]" = torch.ops.aten.view.default(view_11, _shape_param_13);  _shape_param_13 = None
        convert_element_type_10: "f32[16384, 30522]" = torch.ops.prims.convert_element_type.default(view_12, torch.float32);  view_12 = None
        amax: "f32[16384, 1]" = torch.ops.aten.amax.default(convert_element_type_10, [1], True)
        sub_3: "f32[16384, 30522]" = torch.ops.aten.sub.Tensor(convert_element_type_10, amax);  convert_element_type_10 = amax = None
        exp: "f32[16384, 30522]" = torch.ops.aten.exp.default(sub_3)
        sum_1: "f32[16384, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[16384, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_4: "f32[16384, 30522]" = torch.ops.aten.sub.Tensor(sub_3, log);  sub_3 = log = None
        convert_element_type_11: "bf16[16384, 30522]" = torch.ops.prims.convert_element_type.default(sub_4, torch.bfloat16);  sub_4 = None
        ne_1: "b8[16384]" = torch.ops.aten.ne.Scalar(view_9, -100)
        full_1: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[16384]" = torch.ops.aten.where.self(ne_1, view_9, full_1);  ne_1 = full_1 = None
        unsqueeze: "i64[16384, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "bf16[16384, 1]" = torch.ops.aten.gather.default(convert_element_type_11, 1, unsqueeze);  convert_element_type_11 = unsqueeze = None
        squeeze: "bf16[16384]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "bf16[16384]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_2: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[16384]" = torch.ops.aten.where.self(ne, neg, full_2);  ne = neg = full_2 = None
        sum_2: "bf16[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
        ne_2: "b8[16384]" = torch.ops.aten.ne.Scalar(view_9, -100);  view_9 = None
        sum_3: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_12: "bf16[]" = torch.ops.prims.convert_element_type.default(sum_3, torch.bfloat16);  sum_3 = None
        div: "bf16[]" = torch.ops.aten.div.Tensor(sum_2, convert_element_type_12);  sum_2 = convert_element_type_12 = None
        return (view_11, div)



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
