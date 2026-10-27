"""
Standalone repro captured via capture_hook.
Label: AllenaiLongformerBase
Pattern hash: 356c99074d3d
Shape hash: 2ebca403
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
    def forward(self, arg0_1: "f32[768]", arg1_1: "f32[768, 768]", arg2_1: "f32[8, 1024, 768]", arg3_1: "f32[768]", arg4_1: "f32[768]", arg5_1: "f32[50265]", arg6_1: "f32[50265, 768]", arg7_1: "i64[8, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        convert_element_type_2: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        view: "bf16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_0);  convert_element_type_2 = _shape_param_0 = None
        permute: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "bf16[8192, 768]" = torch.ops.aten.addmm.default(convert_element_type, view, permute);  convert_element_type = None
        view_1: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(addmm, _shape_param_1);  _shape_param_1 = None
        convert_element_type_3: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        mul: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 0.5)
        mul_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 0.7071067811865476);  convert_element_type_3 = None
        erf: "f32[8, 1024, 768]" = torch.ops.aten.erf.default(mul_1);  mul_1 = None
        add: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, add);  mul = add = None
        convert_element_type_4: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        convert_element_type_5: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_4, torch.float32);  convert_element_type_4 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_5, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 1024, 1]" = var_mean[0]
        getitem_1: "f32[8, 1024, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_5, getitem_1);  convert_element_type_5 = None
        mul_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_4: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_3, arg3_1);  mul_3 = arg3_1 = None
        add_2: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_4, arg4_1);  mul_4 = arg4_1 = None
        convert_element_type_6: "bf16[50265]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        convert_element_type_7: "bf16[50265, 768]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.bfloat16);  arg6_1 = None
        convert_element_type_8: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        view_2: "bf16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_8, _shape_param_2);  convert_element_type_8 = _shape_param_2 = None
        permute_1: "bf16[768, 50265]" = torch.ops.aten.permute.default(convert_element_type_7, [1, 0]);  convert_element_type_7 = None
        constant_pad_nd: "bf16[768, 50272]" = torch.ops.aten.constant_pad_nd.default(permute_1, _shape_param_3);  _shape_param_3 = None
        full: "bf16[7]" = torch.ops.aten.full.default(_shape_param_4, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_4 = None
        cat: "bf16[50272]" = torch.ops.aten.cat.default([convert_element_type_6, full]);  convert_element_type_6 = full = None
        addmm_1: "bf16[8192, 50272]" = torch.ops.aten.addmm.default(cat, view_2, constant_pad_nd);  cat = constant_pad_nd = None
        slice_1: "bf16[8192, 50265]" = torch.ops.aten.slice.Tensor(addmm_1, 1, 0, -7);  addmm_1 = None
        view_3: "bf16[8, 1024, 50265]" = torch.ops.aten.view.default(slice_1, _shape_param_5);  slice_1 = _shape_param_5 = None
        view_4: "bf16[8192, 50265]" = torch.ops.aten.view.default(view_3, _shape_param_6);  _shape_param_6 = None
        view_5: "i64[8192]" = torch.ops.aten.view.default(arg7_1, [-1]);  arg7_1 = None
        convert_element_type_9: "f32[8192, 50265]" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        amax: "f32[8192, 1]" = torch.ops.aten.amax.default(convert_element_type_9, [1], True)
        sub_1: "f32[8192, 50265]" = torch.ops.aten.sub.Tensor(convert_element_type_9, amax);  convert_element_type_9 = None
        exp: "f32[8192, 50265]" = torch.ops.aten.exp.default(sub_1)
        sum_1: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[8192, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_2: "f32[8192, 50265]" = torch.ops.aten.sub.Tensor(sub_1, log);  sub_1 = None
        convert_element_type_10: "bf16[8192, 50265]" = torch.ops.prims.convert_element_type.default(sub_2, torch.bfloat16);  sub_2 = None
        convert_element_type_11: "f32[8192, 50265]" = torch.ops.prims.convert_element_type.default(convert_element_type_10, torch.float32);  convert_element_type_10 = None
        ne: "b8[8192]" = torch.ops.aten.ne.Scalar(view_5, -100)
        full_1: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[8192]" = torch.ops.aten.where.self(ne, view_5, full_1);  view_5 = full_1 = None
        unsqueeze: "i64[8192, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[8192, 1]" = torch.ops.aten.gather.default(convert_element_type_11, 1, unsqueeze);  convert_element_type_11 = unsqueeze = None
        squeeze: "f32[8192]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[8192]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[8192]" = torch.ops.aten.where.self(ne, neg, full_2);  neg = full_2 = None
        sum_2: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type_12: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
        div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_12);  sum_3 = None
        permute_2: "bf16[50265, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        permute_3: "bf16[768, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (view, addmm, getitem_1, rsqrt, view_2, view_3, amax, log, convert_element_type_12, div, permute_2, permute_3)



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
