"""
Standalone repro captured via capture_hook.
Label: GoogleFnet
Pattern hash: c41d310b7b9c
Shape hash: 6f25edd4
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
    def forward(self, arg0_1: "f32[32, 512, 768, 2]", arg1_1: "f32[32, 512, 768]", arg2_1: "f32[768]", arg3_1: "f32[768]", arg4_1: "f32[3072, 768]", arg5_1: "f32[3072]", arg6_1: "f32[768, 3072]", arg7_1: "f32[768]", arg8_1: "i64[13]", arg9_1: "f32[768]", arg10_1: "f32[768]", arg11_1: "f32[768, 768]", arg12_1: "f32[768]", arg13_1: "f32[768]", arg14_1: "f32[768]", arg15_1: "f32[32000, 768]", arg16_1: "f32[32000]", arg17_1: "i64[32, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9):
        # No stacktrace found for following nodes
        select: "f32[32, 512, 768]" = torch.ops.aten.select.int(arg0_1, 3, 0);  arg0_1 = None
        add: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(arg1_1, select);  arg1_1 = select = None
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean[0]
        getitem_1: "f32[32, 512, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  arg2_1 = None
        add_2: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        view: "f32[16384, 768]" = torch.ops.aten.view.default(add_2, _shape_param_0);  _shape_param_0 = None
        permute: "f32[768, 3072]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        addmm: "f32[16384, 3072]" = torch.ops.aten.addmm.default(arg5_1, view, permute);  arg5_1 = permute = None
        view_1: "f32[32, 512, 3072]" = torch.ops.aten.view.default(addmm, _shape_param_1);  _shape_param_1 = None
        mul_2: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_1, 0.5)
        pow_1: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_1, 3.0)
        mul_3: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_3: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_1, mul_3);  view_1 = mul_3 = None
        mul_4: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_3, 0.7978845608028654);  add_3 = None
        tanh: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_4);  mul_4 = None
        add_4: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_5: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_2, add_4);  mul_2 = add_4 = None
        view_2: "f32[16384, 3072]" = torch.ops.aten.view.default(mul_5, _shape_param_2);  mul_5 = _shape_param_2 = None
        permute_1: "f32[3072, 768]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        addmm_1: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg7_1, view_2, permute_1);  arg7_1 = permute_1 = None
        view_3: "f32[32, 512, 768]" = torch.ops.aten.view.default(addmm_1, _shape_param_3);  addmm_1 = _shape_param_3 = None
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg8_1, 12);  arg8_1 = None
        inductor_random: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_4, inductor_lookup_seed, 'rand');  _shape_param_4 = inductor_lookup_seed = None
        gt: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random, 0.1);  inductor_random = None
        mul_6: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt, view_3);  view_3 = None
        mul_7: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_6, 1.1111111111111112);  mul_6 = None
        add_5: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_7, add_2);  mul_7 = add_2 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
        getitem_2: "f32[32, 512, 1]" = var_mean_1[0]
        getitem_3: "f32[32, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        add_6: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_1: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_5, getitem_3);  add_5 = getitem_3 = None
        mul_8: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_9: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_8, arg9_1);  arg9_1 = None
        add_7: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_9, arg10_1);  mul_9 = arg10_1 = None
        view_4: "f32[16384, 768]" = torch.ops.aten.view.default(add_7, _shape_param_5);  add_7 = _shape_param_5 = None
        permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_2: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg12_1, view_4, permute_2);  arg12_1 = permute_2 = None
        view_5: "f32[32, 512, 768]" = torch.ops.aten.view.default(addmm_2, _shape_param_6);  _shape_param_6 = None
        mul_10: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_5, 0.5)
        pow_2: "f32[32, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(view_5, 3.0)
        mul_11: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_8: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_5, mul_11);  view_5 = mul_11 = None
        mul_12: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_8, 0.7978845608028654);  add_8 = None
        tanh_1: "f32[32, 512, 768]" = torch.ops.aten.tanh.default(mul_12);  mul_12 = None
        add_9: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_13: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_10, add_9);  mul_10 = add_9 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(mul_13, [2], correction = 0, keepdim = True)
        getitem_4: "f32[32, 512, 1]" = var_mean_2[0]
        getitem_5: "f32[32, 512, 1]" = var_mean_2[1];  var_mean_2 = None
        add_10: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_2: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_13, getitem_5);  mul_13 = None
        mul_14: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        mul_15: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_14, arg13_1);  mul_14 = arg13_1 = None
        add_11: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_15, arg14_1);  mul_15 = arg14_1 = None
        view_6: "f32[16384, 768]" = torch.ops.aten.view.default(add_11, _shape_param_7);  add_11 = _shape_param_7 = None
        permute_3: "f32[768, 32000]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_3: "f32[16384, 32000]" = torch.ops.aten.addmm.default(arg16_1, view_6, permute_3);  arg16_1 = permute_3 = None
        view_7: "f32[32, 512, 32000]" = torch.ops.aten.view.default(addmm_3, _shape_param_8);  addmm_3 = _shape_param_8 = None
        view_8: "f32[16384, 32000]" = torch.ops.aten.view.default(view_7, _shape_param_9);  _shape_param_9 = None
        view_9: "i64[16384]" = torch.ops.aten.view.default(arg17_1, [-1]);  arg17_1 = None
        amax: "f32[16384, 1]" = torch.ops.aten.amax.default(view_8, [1], True)
        sub_3: "f32[16384, 32000]" = torch.ops.aten.sub.Tensor(view_8, amax);  view_8 = None
        exp: "f32[16384, 32000]" = torch.ops.aten.exp.default(sub_3)
        sum_1: "f32[16384, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[16384, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_4: "f32[16384, 32000]" = torch.ops.aten.sub.Tensor(sub_3, log);  sub_3 = None
        ne: "b8[16384]" = torch.ops.aten.ne.Scalar(view_9, -100)
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[16384]" = torch.ops.aten.where.self(ne, view_9, full);  view_9 = full = None
        unsqueeze: "i64[16384, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[16384, 1]" = torch.ops.aten.gather.default(sub_4, 1, unsqueeze);  sub_4 = unsqueeze = None
        squeeze: "f32[16384]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[16384]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[16384]" = torch.ops.aten.where.self(ne, neg, full_1);  neg = full_1 = None
        sum_2: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
        div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type);  sum_3 = None
        div_1: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
        div_2: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        return (mul, view, addmm, view_2, gt, mul_8, view_4, addmm_2, getitem_5, rsqrt_2, view_6, view_7, amax, log, convert_element_type, div, div_1, div_2)



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
