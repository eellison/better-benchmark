"""
Standalone repro captured via capture_hook.
Label: MBartForCausalLM
Pattern hash: ce5cede51f0b
Shape hash: 949c0685
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
    def forward(self, arg0_1: "bf16[8, 16, 1024, 64]", arg1_1: "bf16[1024, 1024]", arg2_1: "bf16[1024]", arg3_1: "bf16[8, 1024, 1024]", arg4_1: "bf16[1024]", arg5_1: "bf16[1024]", arg6_1: "bf16[4096, 1024]", arg7_1: "bf16[4096]", arg8_1: "bf16[1024, 4096]", arg9_1: "bf16[1024]", arg10_1: "i64[8, 1024]", arg11_1: "bf16[1024]", arg12_1: "bf16[1024]", arg13_1: "bf16[50265, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10):
        # No stacktrace found for following nodes
        permute: "bf16[8, 1024, 16, 64]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
        view: "bf16[8, 1024, 1024]" = torch.ops.aten.view.default(permute, _shape_param_0);  permute = _shape_param_0 = None
        view_1: "bf16[8192, 1024]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute_1: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        addmm: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg2_1, view_1, permute_1);  arg2_1 = view_1 = permute_1 = None
        view_2: "bf16[8, 1024, 1024]" = torch.ops.aten.view.default(addmm, _shape_param_2);  addmm = _shape_param_2 = None
        add: "bf16[8, 1024, 1024]" = torch.ops.aten.add.Tensor(arg3_1, view_2);  arg3_1 = view_2 = None
        convert_element_type: "f32[8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 1024, 1]" = var_mean[0]
        getitem_1: "f32[8, 1024, 1]" = var_mean[1];  var_mean = None
        sub: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_1: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
        add_2: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
        convert_element_type_1: "bf16[8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        view_3: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_3);  convert_element_type_1 = _shape_param_3 = None
        permute_2: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        addmm_1: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(arg7_1, view_3, permute_2);  arg7_1 = view_3 = permute_2 = None
        view_4: "bf16[8, 1024, 4096]" = torch.ops.aten.view.default(addmm_1, _shape_param_4);  addmm_1 = _shape_param_4 = None
        convert_element_type_2: "f32[8, 1024, 4096]" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        mul_2: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 0.5)
        mul_3: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 0.7071067811865476);  convert_element_type_2 = None
        erf: "f32[8, 1024, 4096]" = torch.ops.aten.erf.default(mul_3);  mul_3 = None
        add_3: "f32[8, 1024, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_4: "f32[8, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_2, add_3);  mul_2 = add_3 = None
        convert_element_type_3: "bf16[8, 1024, 4096]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        view_5: "bf16[8192, 4096]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_5);  convert_element_type_3 = _shape_param_5 = None
        permute_3: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm_2: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg9_1, view_5, permute_3);  arg9_1 = view_5 = permute_3 = None
        view_6: "bf16[8, 1024, 1024]" = torch.ops.aten.view.default(addmm_2, _shape_param_6);  addmm_2 = _shape_param_6 = None
        add_4: "bf16[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add, view_6);  add = view_6 = None
        convert_element_type_4: "f32[8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_4, torch.float32);  add_4 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_4, [2], correction = 0, keepdim = True)
        getitem_2: "f32[8, 1024, 1]" = var_mean_1[0]
        getitem_3: "f32[8, 1024, 1]" = var_mean_1[1];  var_mean_1 = None
        view_7: "i64[8192]" = torch.ops.aten.view.default(arg10_1, [-1]);  arg10_1 = None
        ne: "b8[8192]" = torch.ops.aten.ne.Scalar(view_7, -100)
        sub_1: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_4, getitem_3);  convert_element_type_4 = getitem_3 = None
        add_5: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        mul_5: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_6: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_5, arg11_1);  mul_5 = arg11_1 = None
        add_6: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_6, arg12_1);  mul_6 = arg12_1 = None
        convert_element_type_5: "bf16[8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16);  add_6 = None
        view_8: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_7);  convert_element_type_5 = _shape_param_7 = None
        permute_4: "bf16[1024, 50265]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        constant_pad_nd: "bf16[1024, 50272]" = torch.ops.aten.constant_pad_nd.default(permute_4, _shape_param_8);  permute_4 = _shape_param_8 = None
        mm: "bf16[8192, 50272]" = torch.ops.aten.mm.default(view_8, constant_pad_nd);  view_8 = constant_pad_nd = None
        slice_1: "bf16[8192, 50265]" = torch.ops.aten.slice.Tensor(mm, 1, 0, -7);  mm = None
        view_9: "bf16[8, 1024, 50265]" = torch.ops.aten.view.default(slice_1, _shape_param_9);  slice_1 = _shape_param_9 = None
        view_10: "bf16[8192, 50265]" = torch.ops.aten.view.default(view_9, _shape_param_10);  _shape_param_10 = None
        convert_element_type_6: "f32[8192, 50265]" = torch.ops.prims.convert_element_type.default(view_10, torch.float32);  view_10 = None
        amax: "f32[8192, 1]" = torch.ops.aten.amax.default(convert_element_type_6, [1], True)
        sub_2: "f32[8192, 50265]" = torch.ops.aten.sub.Tensor(convert_element_type_6, amax);  convert_element_type_6 = amax = None
        exp: "f32[8192, 50265]" = torch.ops.aten.exp.default(sub_2)
        sum_1: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[8192, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_3: "f32[8192, 50265]" = torch.ops.aten.sub.Tensor(sub_2, log);  sub_2 = log = None
        convert_element_type_7: "bf16[8192, 50265]" = torch.ops.prims.convert_element_type.default(sub_3, torch.bfloat16);  sub_3 = None
        ne_1: "b8[8192]" = torch.ops.aten.ne.Scalar(view_7, -100)
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[8192]" = torch.ops.aten.where.self(ne_1, view_7, full);  ne_1 = full = None
        unsqueeze: "i64[8192, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "bf16[8192, 1]" = torch.ops.aten.gather.default(convert_element_type_7, 1, unsqueeze);  convert_element_type_7 = unsqueeze = None
        squeeze: "bf16[8192]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "bf16[8192]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[8192]" = torch.ops.aten.where.self(ne, neg, full_1);  ne = neg = full_1 = None
        sum_2: "bf16[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
        ne_2: "b8[8192]" = torch.ops.aten.ne.Scalar(view_7, -100);  view_7 = None
        sum_3: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_8: "bf16[]" = torch.ops.prims.convert_element_type.default(sum_3, torch.bfloat16);  sum_3 = None
        div: "bf16[]" = torch.ops.aten.div.Tensor(sum_2, convert_element_type_8);  sum_2 = convert_element_type_8 = None
        return (view_9, div)



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
