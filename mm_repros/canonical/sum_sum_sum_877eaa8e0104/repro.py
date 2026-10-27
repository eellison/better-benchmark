"""
Standalone repro captured via capture_hook.
Label: AllenaiLongformerBase
Pattern hash: 877eaa8e0104
Shape hash: b78a12e3
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
    def forward(self, arg0_1: "f32[]", arg1_1: "f32[]", arg2_1: "i64[8, 1024]", arg3_1: "bf16[8, 1024, 50265]", arg4_1: "f32[8192, 1]", arg5_1: "f32[8192, 1]", arg6_1: "bf16[8, 1024, 50265]", arg7_1: "bf16[50265, 768]", arg8_1: "bf16[8192, 768]", arg9_1: "f32[768]", arg10_1: "bf16[8192, 768]", arg11_1: "f32[8, 1024, 1]", arg12_1: "f32[8, 1024, 1]", arg13_1: "bf16[768, 768]", arg14_1: "bf16[8192, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13):
        # No stacktrace found for following nodes
        div: "f32[]" = torch.ops.aten.div.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "i64[8192]" = torch.ops.aten.view.default(arg2_1, [-1]);  arg2_1 = None
        unsqueeze: "i64[8192, 1]" = torch.ops.aten.unsqueeze.default(view, 1);  view = None
        ne: "b8[8192, 1]" = torch.ops.aten.ne.Scalar(unsqueeze, -100)
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[8192, 1]" = torch.ops.aten.where.self(ne, unsqueeze, full);  unsqueeze = full = None
        iota: "i64[50265]" = torch.ops.prims.iota.default(50265, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_1: "i64[1, 50265]" = torch.ops.aten.view.default(iota, _shape_param_0);  iota = _shape_param_0 = None
        expand: "i64[8192, 50265]" = torch.ops.aten.expand.default(where, _shape_param_1);  where = _shape_param_1 = None
        eq: "b8[8192, 50265]" = torch.ops.aten.eq.Tensor(expand, view_1);  expand = view_1 = None
        scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where_1: "f32[8192, 50265]" = torch.ops.aten.where.self(eq, scalar_tensor_1, scalar_tensor);  eq = scalar_tensor_1 = scalar_tensor = None
        full_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[8192, 1]" = torch.ops.aten.where.self(ne, div, full_1);  ne = div = full_1 = None
        mul: "f32[8192, 50265]" = torch.ops.aten.mul.Tensor(where_1, where_2);  where_1 = where_2 = None
        convert_element_type: "bf16[8192, 50265]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        convert_element_type_1: "f32[8192, 50265]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        view_2: "bf16[8192, 50265]" = torch.ops.aten.view.default(arg3_1, _shape_param_2);  arg3_1 = _shape_param_2 = None
        convert_element_type_2: "f32[8192, 50265]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        sub: "f32[8192, 50265]" = torch.ops.aten.sub.Tensor(convert_element_type_2, arg4_1);  convert_element_type_2 = arg4_1 = None
        sub_1: "f32[8192, 50265]" = torch.ops.aten.sub.Tensor(sub, arg5_1);  sub = arg5_1 = None
        convert_element_type_3: "bf16[8192, 50265]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        convert_element_type_4: "f32[8192, 50265]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        exp: "f32[8192, 50265]" = torch.ops.aten.exp.default(convert_element_type_4);  convert_element_type_4 = None
        sum_1: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1, [1], True)
        mul_1: "f32[8192, 50265]" = torch.ops.aten.mul.Tensor(exp, sum_1);  exp = sum_1 = None
        sub_2: "f32[8192, 50265]" = torch.ops.aten.sub.Tensor(convert_element_type_1, mul_1);  convert_element_type_1 = mul_1 = None
        convert_element_type_5: "bf16[8192, 50265]" = torch.ops.prims.convert_element_type.default(sub_2, torch.bfloat16);  sub_2 = None
        view_3: "bf16[8, 1024, 50265]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_3);  convert_element_type_5 = _shape_param_3 = None
        add: "bf16[8, 1024, 50265]" = torch.ops.aten.add.Tensor(arg6_1, view_3);  arg6_1 = view_3 = None
        view_4: "bf16[8192, 50265]" = torch.ops.aten.view.default(add, _shape_param_4);  add = _shape_param_4 = None
        constant_pad_nd: "bf16[8192, 50272]" = torch.ops.aten.constant_pad_nd.default(view_4, _shape_param_5);  _shape_param_5 = None
        constant_pad_nd_1: "bf16[50272, 768]" = torch.ops.aten.constant_pad_nd.default(arg7_1, _shape_param_6);  arg7_1 = _shape_param_6 = None
        mm: "bf16[8192, 768]" = torch.ops.aten.mm.default(constant_pad_nd, constant_pad_nd_1);  constant_pad_nd = constant_pad_nd_1 = None
        permute: "bf16[50265, 8192]" = torch.ops.aten.permute.default(view_4, [1, 0])
        constant_pad_nd_2: "bf16[50272, 8192]" = torch.ops.aten.constant_pad_nd.default(permute, _shape_param_7);  permute = _shape_param_7 = None
        mm_1: "bf16[50272, 768]" = torch.ops.aten.mm.default(constant_pad_nd_2, arg8_1);  constant_pad_nd_2 = arg8_1 = None
        slice_1: "bf16[50265, 768]" = torch.ops.aten.slice.Tensor(mm_1, 0, 0, -7);  mm_1 = None
        sum_2: "f32[1, 50265]" = torch.ops.aten.sum.dim_IntList(view_4, [0], True, dtype = torch.float32);  view_4 = None
        view_5: "f32[50265]" = torch.ops.aten.view.default(sum_2, _shape_param_8);  sum_2 = _shape_param_8 = None
        convert_element_type_6: "bf16[50265]" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None
        view_6: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm, _shape_param_9);  mm = _shape_param_9 = None
        convert_element_type_7: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None
        convert_element_type_8: "f32[50265, 768]" = torch.ops.prims.convert_element_type.default(slice_1, torch.float32);  slice_1 = None
        convert_element_type_9: "f32[50265]" = torch.ops.prims.convert_element_type.default(convert_element_type_6, torch.float32);  convert_element_type_6 = None
        mul_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_7, arg9_1);  arg9_1 = None
        mul_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2, 768)
        sum_3: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True)
        view_7: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(arg10_1, _shape_param_10);  arg10_1 = _shape_param_10 = None
        convert_element_type_10: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(view_7, torch.float32);  view_7 = None
        mul_4: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 0.5)
        mul_5: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 0.7071067811865476)
        erf: "f32[8, 1024, 768]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_1: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_4, add_1);  mul_4 = None
        convert_element_type_11: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        convert_element_type_12: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_11, torch.float32);  convert_element_type_11 = None
        sub_3: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_12, arg11_1);  convert_element_type_12 = arg11_1 = None
        mul_7: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_3, arg12_1);  sub_3 = None
        mul_8: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2, mul_7);  mul_2 = None
        sum_4: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_8, [2], True);  mul_8 = None
        mul_9: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_7, sum_4);  sum_4 = None
        sub_4: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_3, sum_3);  mul_3 = sum_3 = None
        sub_5: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_4, mul_9);  sub_4 = mul_9 = None
        div_1: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(arg12_1, 768);  arg12_1 = None
        mul_10: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_1, sub_5);  div_1 = sub_5 = None
        mul_11: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_7, mul_7);  mul_7 = None
        sum_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_11, [0, 1]);  mul_11 = None
        sum_6: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_7, [0, 1]);  convert_element_type_7 = None
        convert_element_type_13: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(mul_10, torch.bfloat16);  mul_10 = None
        convert_element_type_14: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_13, torch.float32);  convert_element_type_13 = None
        mul_12: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_1, 0.5);  add_1 = None
        mul_13: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_10, convert_element_type_10)
        mul_14: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_13, -0.5);  mul_13 = None
        exp_1: "f32[8, 1024, 768]" = torch.ops.aten.exp.default(mul_14);  mul_14 = None
        mul_15: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
        mul_16: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_10, mul_15);  convert_element_type_10 = mul_15 = None
        add_2: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_12, mul_16);  mul_12 = mul_16 = None
        mul_17: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_14, add_2);  convert_element_type_14 = add_2 = None
        convert_element_type_15: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16);  mul_17 = None
        view_8: "bf16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_15, _shape_param_11);  convert_element_type_15 = _shape_param_11 = None
        mm_2: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_8, arg13_1);  arg13_1 = None
        permute_1: "bf16[768, 8192]" = torch.ops.aten.permute.default(view_8, [1, 0])
        mm_3: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_1, arg14_1);  permute_1 = arg14_1 = None
        sum_7: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_8, [0], True, dtype = torch.float32);  view_8 = None
        view_9: "f32[768]" = torch.ops.aten.view.default(sum_7, _shape_param_12);  sum_7 = _shape_param_12 = None
        convert_element_type_16: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_9, torch.bfloat16);  view_9 = None
        view_10: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm_2, _shape_param_13);  mm_2 = _shape_param_13 = None
        convert_element_type_17: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(view_10, torch.float32);  view_10 = None
        convert_element_type_18: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_19: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_16, torch.float32);  convert_element_type_16 = None
        return (convert_element_type_8, convert_element_type_9, sum_5, sum_6, convert_element_type_17, convert_element_type_18, convert_element_type_19)



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
