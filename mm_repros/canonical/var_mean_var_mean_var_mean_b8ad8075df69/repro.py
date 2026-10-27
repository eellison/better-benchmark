"""
Standalone repro captured via capture_hook.
Label: MegatronBertForCausalLM
Pattern hash: b8ad8075df69
Shape hash: ec280408
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
    def forward(self, arg0_1: "bf16[16, 16, 512, 64]", arg1_1: "f32[1024]", arg2_1: "f32[1024, 1024]", arg3_1: "i64[49]", arg4_1: "f32[16, 512, 1024]", arg5_1: "f32[1024]", arg6_1: "f32[1024]", arg7_1: "f32[4096]", arg8_1: "f32[4096, 1024]", arg9_1: "f32[1024]", arg10_1: "f32[1024, 4096]", arg11_1: "f32[1024]", arg12_1: "f32[1024]", arg13_1: "f32[1024]", arg14_1: "f32[1024, 1024]", arg15_1: "f32[1024]", arg16_1: "f32[1024]", arg17_1: "f32[29056]", arg18_1: "f32[29056, 1024]", arg19_1: "i64[16, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13):
        # No stacktrace found for following nodes
        permute: "bf16[16, 512, 16, 64]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
        clone: "bf16[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(clone, _shape_param_0);  clone = _shape_param_0 = None
        convert_element_type: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        convert_element_type_1: "bf16[1024, 1024]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        view_1: "bf16[8192, 1024]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute_1: "bf16[1024, 1024]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(convert_element_type, view_1, permute_1);  convert_element_type = None
        view_2: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(addmm, _shape_param_2);  addmm = _shape_param_2 = None
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg3_1, 47)
        inductor_random: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_3, inductor_lookup_seed, 'rand');  _shape_param_3 = inductor_lookup_seed = None
        convert_element_type_2: "bf16[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_2, 0.1);  convert_element_type_2 = None
        mul: "bf16[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt, view_2);  view_2 = None
        mul_1: "bf16[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        add: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(arg4_1, mul_1);  arg4_1 = mul_1 = None
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[16, 512, 1]" = var_mean[0]
        getitem_1: "f32[16, 512, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add, getitem_1);  getitem_1 = None
        mul_2: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_3: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_2, arg5_1);  arg5_1 = None
        add_2: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_3, arg6_1);  mul_3 = arg6_1 = None
        convert_element_type_3: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.bfloat16);  arg7_1 = None
        convert_element_type_4: "bf16[4096, 1024]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.bfloat16);  arg8_1 = None
        convert_element_type_5: "bf16[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        view_3: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_4);  convert_element_type_5 = _shape_param_4 = None
        permute_2: "bf16[1024, 4096]" = torch.ops.aten.permute.default(convert_element_type_4, [1, 0]);  convert_element_type_4 = None
        addmm_1: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(convert_element_type_3, view_3, permute_2);  convert_element_type_3 = None
        view_4: "bf16[16, 512, 4096]" = torch.ops.aten.view.default(addmm_1, _shape_param_5);  _shape_param_5 = None
        convert_element_type_6: "f32[16, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        mul_4: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.5)
        mul_5: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.7071067811865476);  convert_element_type_6 = None
        erf: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_3: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_4, add_3);  mul_4 = add_3 = None
        convert_element_type_7: "bf16[16, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        convert_element_type_8: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.bfloat16);  arg9_1 = None
        convert_element_type_9: "bf16[1024, 4096]" = torch.ops.prims.convert_element_type.default(arg10_1, torch.bfloat16);  arg10_1 = None
        view_5: "bf16[8192, 4096]" = torch.ops.aten.view.default(convert_element_type_7, _shape_param_6);  convert_element_type_7 = _shape_param_6 = None
        permute_3: "bf16[4096, 1024]" = torch.ops.aten.permute.default(convert_element_type_9, [1, 0]);  convert_element_type_9 = None
        addmm_2: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(convert_element_type_8, view_5, permute_3);  convert_element_type_8 = None
        view_6: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(addmm_2, _shape_param_7);  addmm_2 = _shape_param_7 = None
        inductor_lookup_seed_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg3_1, 48);  arg3_1 = None
        inductor_random_1: "f32[16, 512, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_8, inductor_lookup_seed_1, 'rand');  _shape_param_8 = inductor_lookup_seed_1 = None
        convert_element_type_10: "bf16[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_1, torch.bfloat16);  inductor_random_1 = None
        gt_1: "b8[16, 512, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_10, 0.1);  convert_element_type_10 = None
        mul_7: "bf16[16, 512, 1024]" = torch.ops.aten.mul.Tensor(gt_1, view_6);  view_6 = None
        mul_8: "bf16[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_7, 1.1111111111111112);  mul_7 = None
        add_4: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add, mul_8);  add = mul_8 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
        getitem_2: "f32[16, 512, 1]" = var_mean_1[0]
        getitem_3: "f32[16, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        add_5: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_1: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_4, getitem_3);  add_4 = getitem_3 = None
        mul_9: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_10: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_9, arg11_1);  arg11_1 = None
        add_6: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_10, arg12_1);  mul_10 = arg12_1 = None
        convert_element_type_11: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg13_1, torch.bfloat16);  arg13_1 = None
        convert_element_type_12: "bf16[1024, 1024]" = torch.ops.prims.convert_element_type.default(arg14_1, torch.bfloat16);  arg14_1 = None
        convert_element_type_13: "bf16[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16);  add_6 = None
        view_7: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_13, _shape_param_9);  convert_element_type_13 = _shape_param_9 = None
        permute_4: "bf16[1024, 1024]" = torch.ops.aten.permute.default(convert_element_type_12, [1, 0]);  convert_element_type_12 = None
        addmm_3: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(convert_element_type_11, view_7, permute_4);  convert_element_type_11 = None
        view_8: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(addmm_3, _shape_param_10);  _shape_param_10 = None
        convert_element_type_14: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(view_8, torch.float32);  view_8 = None
        mul_11: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 0.5)
        mul_12: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 0.7071067811865476);  convert_element_type_14 = None
        erf_1: "f32[16, 512, 1024]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_7: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_13: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_11, add_7);  mul_11 = add_7 = None
        convert_element_type_15: "bf16[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_13, torch.bfloat16);  mul_13 = None
        convert_element_type_16: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_15, torch.float32);  convert_element_type_15 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_16, [2], correction = 0, keepdim = True)
        getitem_4: "f32[16, 512, 1]" = var_mean_2[0]
        getitem_5: "f32[16, 512, 1]" = var_mean_2[1];  var_mean_2 = None
        add_8: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_2: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_16, getitem_5);  convert_element_type_16 = None
        mul_14: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        mul_15: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_14, arg15_1);  mul_14 = arg15_1 = None
        add_9: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_15, arg16_1);  mul_15 = arg16_1 = None
        convert_element_type_17: "bf16[29056]" = torch.ops.prims.convert_element_type.default(arg17_1, torch.bfloat16);  arg17_1 = None
        convert_element_type_18: "bf16[29056, 1024]" = torch.ops.prims.convert_element_type.default(arg18_1, torch.bfloat16);  arg18_1 = None
        convert_element_type_19: "bf16[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_9, torch.bfloat16);  add_9 = None
        view_9: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_19, _shape_param_11);  convert_element_type_19 = _shape_param_11 = None
        permute_5: "bf16[1024, 29056]" = torch.ops.aten.permute.default(convert_element_type_18, [1, 0]);  convert_element_type_18 = None
        addmm_4: "bf16[8192, 29056]" = torch.ops.aten.addmm.default(convert_element_type_17, view_9, permute_5);  convert_element_type_17 = None
        view_10: "bf16[16, 512, 29056]" = torch.ops.aten.view.default(addmm_4, _shape_param_12);  addmm_4 = _shape_param_12 = None
        convert_element_type_20: "f32[16, 512, 29056]" = torch.ops.prims.convert_element_type.default(view_10, torch.float32)
        constant_pad_nd: "i64[16, 513]" = torch.ops.aten.constant_pad_nd.default(arg19_1, [0, 1], -100.0);  arg19_1 = None
        slice_1: "i64[16, 512]" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807)
        clone_1: "i64[16, 512]" = torch.ops.aten.clone.default(slice_1, memory_format = torch.contiguous_format);  slice_1 = None
        view_11: "f32[8192, 29056]" = torch.ops.aten.view.default(convert_element_type_20, _shape_param_13);  convert_element_type_20 = _shape_param_13 = None
        view_12: "i64[8192]" = torch.ops.aten.view.default(clone_1, [-1]);  clone_1 = None
        amax: "f32[8192, 1]" = torch.ops.aten.amax.default(view_11, [1], True)
        sub_3: "f32[8192, 29056]" = torch.ops.aten.sub.Tensor(view_11, amax);  view_11 = None
        exp: "f32[8192, 29056]" = torch.ops.aten.exp.default(sub_3)
        sum_1: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[8192, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_4: "f32[8192, 29056]" = torch.ops.aten.sub.Tensor(sub_3, log);  sub_3 = None
        ne: "b8[8192]" = torch.ops.aten.ne.Scalar(view_12, -100)
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[8192]" = torch.ops.aten.where.self(ne, view_12, full);  view_12 = full = None
        unsqueeze: "i64[8192, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[8192, 1]" = torch.ops.aten.gather.default(sub_4, 1, unsqueeze);  sub_4 = unsqueeze = None
        squeeze: "f32[8192]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[8192]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[8192]" = torch.ops.aten.where.self(ne, neg, full_1);  neg = full_1 = None
        sum_2: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type_21: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
        div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_21);  sum_3 = None
        permute_6: "bf16[29056, 1024]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        permute_7: "bf16[1024, 1024]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        div_1: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 1024);  rsqrt_1 = None
        permute_8: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        permute_9: "bf16[4096, 1024]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        div_2: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None
        permute_10: "bf16[1024, 1024]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        return (view_1, gt, mul_2, view_3, addmm_1, view_5, gt_1, mul_9, view_7, addmm_3, getitem_5, rsqrt_2, view_9, view_10, constant_pad_nd, amax, log, convert_element_type_21, div, permute_6, permute_7, div_1, permute_8, permute_9, div_2, permute_10)



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
