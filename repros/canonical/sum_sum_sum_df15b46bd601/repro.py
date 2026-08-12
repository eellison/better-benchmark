"""
Standalone repro captured via capture_hook.
Label: hf_GPTJForQuestionAnswering_train
Pattern hash: df15b46bd601
Shape hash: 3e32ddcf
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
    def forward(self, arg0_1: "f32[]", arg1_1: "b8[1]", arg2_1: "i64[1]", arg3_1: "bf16[1, 128]", arg4_1: "f32[1, 1]", arg5_1: "f32[1, 1]", arg6_1: "bf16[1, 128]", arg7_1: "b8[1]", arg8_1: "i64[1]", arg9_1: "bf16[1, 128]", arg10_1: "f32[1, 1]", arg11_1: "f32[1, 1]", arg12_1: "bf16[1, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        div: "f32[]" = torch.ops.aten.div.Tensor(arg0_1, 2);  arg0_1 = None
        sum_1: "i64[]" = torch.ops.aten.sum.default(arg1_1);  arg1_1 = None
        convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_1, torch.float32);  sum_1 = None
        div_1: "f32[]" = torch.ops.aten.div.Tensor(div, convert_element_type);  convert_element_type = None
        clamp_min: "i64[1]" = torch.ops.aten.clamp_min.default(arg2_1, 0);  arg2_1 = None
        clamp_max: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min, 128);  clamp_min = None
        unsqueeze: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(clamp_max, 1);  clamp_max = None
        ne: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze, 128)
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[1, 1]" = torch.ops.aten.where.self(ne, unsqueeze, full);  unsqueeze = None
        iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view: "i64[1, 128]" = torch.ops.aten.view.default(iota, _shape_param_0);  iota = _shape_param_0 = None
        expand: "i64[1, 128]" = torch.ops.aten.expand.default(where, _shape_param_1);  where = _shape_param_1 = None
        eq: "b8[1, 128]" = torch.ops.aten.eq.Tensor(expand, view);  expand = None
        scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where_1: "f32[1, 128]" = torch.ops.aten.where.self(eq, scalar_tensor_1, scalar_tensor);  eq = None
        full_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[1, 1]" = torch.ops.aten.where.self(ne, div_1, full_1);  ne = div_1 = None
        mul: "f32[1, 128]" = torch.ops.aten.mul.Tensor(where_1, where_2);  where_1 = where_2 = None
        convert_element_type_1: "bf16[1, 128]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        convert_element_type_2: "f32[1, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        convert_element_type_3: "f32[1, 128]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        sub: "f32[1, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_3, arg4_1);  convert_element_type_3 = arg4_1 = None
        sub_1: "f32[1, 128]" = torch.ops.aten.sub.Tensor(sub, arg5_1);  sub = arg5_1 = None
        convert_element_type_4: "bf16[1, 128]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        convert_element_type_5: "f32[1, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_4, torch.float32);  convert_element_type_4 = None
        exp: "f32[1, 128]" = torch.ops.aten.exp.default(convert_element_type_5);  convert_element_type_5 = None
        sum_2: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(convert_element_type_2, [1], True)
        mul_1: "f32[1, 128]" = torch.ops.aten.mul.Tensor(exp, sum_2);  exp = sum_2 = None
        sub_2: "f32[1, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_2, mul_1);  convert_element_type_2 = mul_1 = None
        convert_element_type_6: "bf16[1, 128]" = torch.ops.prims.convert_element_type.default(sub_2, torch.bfloat16);  sub_2 = None
        add: "bf16[1, 128]" = torch.ops.aten.add.Tensor(arg6_1, convert_element_type_6);  arg6_1 = convert_element_type_6 = None
        sum_3: "i64[]" = torch.ops.aten.sum.default(arg7_1);  arg7_1 = None
        convert_element_type_7: "f32[]" = torch.ops.prims.convert_element_type.default(sum_3, torch.float32);  sum_3 = None
        div_2: "f32[]" = torch.ops.aten.div.Tensor(div, convert_element_type_7);  div = convert_element_type_7 = None
        clamp_min_1: "i64[1]" = torch.ops.aten.clamp_min.default(arg8_1, 0);  arg8_1 = None
        clamp_max_1: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min_1, 128);  clamp_min_1 = None
        unsqueeze_1: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(clamp_max_1, 1);  clamp_max_1 = None
        ne_1: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_1, 128)
        where_3: "i64[1, 1]" = torch.ops.aten.where.self(ne_1, unsqueeze_1, full);  unsqueeze_1 = full = None
        expand_1: "i64[1, 128]" = torch.ops.aten.expand.default(where_3, _shape_param_2);  where_3 = _shape_param_2 = None
        eq_1: "b8[1, 128]" = torch.ops.aten.eq.Tensor(expand_1, view);  expand_1 = view = None
        where_4: "f32[1, 128]" = torch.ops.aten.where.self(eq_1, scalar_tensor_1, scalar_tensor);  eq_1 = scalar_tensor_1 = scalar_tensor = None
        where_5: "f32[1, 1]" = torch.ops.aten.where.self(ne_1, div_2, full_1);  ne_1 = div_2 = full_1 = None
        mul_2: "f32[1, 128]" = torch.ops.aten.mul.Tensor(where_4, where_5);  where_4 = where_5 = None
        convert_element_type_8: "bf16[1, 128]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        convert_element_type_9: "f32[1, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_8, torch.float32);  convert_element_type_8 = None
        convert_element_type_10: "f32[1, 128]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.float32);  arg9_1 = None
        sub_3: "f32[1, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_10, arg10_1);  convert_element_type_10 = arg10_1 = None
        sub_4: "f32[1, 128]" = torch.ops.aten.sub.Tensor(sub_3, arg11_1);  sub_3 = arg11_1 = None
        convert_element_type_11: "bf16[1, 128]" = torch.ops.prims.convert_element_type.default(sub_4, torch.bfloat16);  sub_4 = None
        convert_element_type_12: "f32[1, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_11, torch.float32);  convert_element_type_11 = None
        exp_1: "f32[1, 128]" = torch.ops.aten.exp.default(convert_element_type_12);  convert_element_type_12 = None
        sum_4: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(convert_element_type_9, [1], True)
        mul_3: "f32[1, 128]" = torch.ops.aten.mul.Tensor(exp_1, sum_4);  exp_1 = sum_4 = None
        sub_5: "f32[1, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_9, mul_3);  convert_element_type_9 = mul_3 = None
        convert_element_type_13: "bf16[1, 128]" = torch.ops.prims.convert_element_type.default(sub_5, torch.bfloat16);  sub_5 = None
        add_1: "bf16[1, 128]" = torch.ops.aten.add.Tensor(arg12_1, convert_element_type_13);  arg12_1 = convert_element_type_13 = None
        unsqueeze_2: "bf16[1, 128, 1]" = torch.ops.aten.unsqueeze.default(add, 2);  add = None
        unsqueeze_3: "bf16[1, 128, 1]" = torch.ops.aten.unsqueeze.default(add_1, 2);  add_1 = None
        cat: "bf16[1, 128, 2]" = torch.ops.aten.cat.default([unsqueeze_3, unsqueeze_2], 2);  unsqueeze_3 = unsqueeze_2 = None
        view_1: "bf16[128, 2]" = torch.ops.aten.view.default(cat, _shape_param_3);  cat = _shape_param_3 = None
        constant_pad_nd: "bf16[128, 8]" = torch.ops.aten.constant_pad_nd.default(view_1, _shape_param_4);  _shape_param_4 = None
        permute: "bf16[2, 128]" = torch.ops.aten.permute.default(view_1, [1, 0])
        sum_5: "f32[1, 2]" = torch.ops.aten.sum.dim_IntList(view_1, [0], True, dtype = torch.float32);  view_1 = None
        view_2: "f32[2]" = torch.ops.aten.view.default(sum_5, _shape_param_5);  sum_5 = _shape_param_5 = None
        convert_element_type_14: "bf16[2]" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        convert_element_type_15: "f32[2]" = torch.ops.prims.convert_element_type.default(convert_element_type_14, torch.float32);  convert_element_type_14 = None
        return (constant_pad_nd, permute, convert_element_type_15)



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
