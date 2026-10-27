"""
Standalone repro captured via capture_hook.
Label: gpt-oss-20b
Pattern hash: 3ac61cd6e287
Shape hash: 35159a0f
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
    def forward(self, arg0_1: "i64[4000]", arg1_1: "i64[4000]", arg2_1: "bf16[1000, 2880]", arg3_1: "bf16[32, 2880, 5760]", arg4_1: "i32[32]", arg5_1: "bf16[32, 5760]", arg6_1: "bf16[32, 2880, 2880]", arg7_1: "bf16[32, 2880]", arg8_1: "bf16[1000, 4]", arg9_1: "bf16[1, 1000, 2880]", arg10_1: "bf16[2880]", arg11_1: "bf16[201088, 2880]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        full: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        ge: "b8[4000]" = torch.ops.aten.ge.Scalar(arg0_1, 32)
        unsqueeze: "b8[4000, 1]" = torch.ops.aten.unsqueeze.default(ge, -1);  ge = None
        full_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        div: "i64[4000]" = torch.ops.aten.div.Tensor_mode(arg1_1, 4, rounding_mode = 'floor')
        index: "bf16[4000, 2880]" = torch.ops.aten.index.Tensor(arg2_1, [div]);  arg2_1 = div = None
        where: "bf16[4000, 2880]" = torch.ops.aten.where.self(unsqueeze, full_1, index);  full_1 = index = None
        _grouped_mm: "bf16[4000, 5760]" = torch.ops.aten._grouped_mm.default(where, arg3_1, arg4_1);  where = arg3_1 = None
        clamp_max: "i64[4000]" = torch.ops.aten.clamp_max.default(arg0_1, 31);  arg0_1 = None
        index_1: "bf16[4000, 5760]" = torch.ops.aten.index.Tensor(arg5_1, [clamp_max]);  arg5_1 = None
        add: "bf16[4000, 5760]" = torch.ops.aten.add.Tensor(_grouped_mm, index_1);  _grouped_mm = index_1 = None
        slice_1: "bf16[4000, 2880]" = torch.ops.aten.slice.Tensor(add, 1, 1, 9223372036854775807, 2)
        convert_element_type: "f32[4000, 2880]" = torch.ops.prims.convert_element_type.default(slice_1, torch.float32);  slice_1 = None
        clamp_min: "f32[4000, 2880]" = torch.ops.aten.clamp_min.default(convert_element_type, -7.0);  convert_element_type = None
        clamp_max_1: "f32[4000, 2880]" = torch.ops.aten.clamp_max.default(clamp_min, 7.0);  clamp_min = None
        convert_element_type_1: "bf16[4000, 2880]" = torch.ops.prims.convert_element_type.default(clamp_max_1, torch.bfloat16);  clamp_max_1 = None
        add_1: "bf16[4000, 2880]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1);  convert_element_type_1 = None
        slice_2: "bf16[4000, 2880]" = torch.ops.aten.slice.Tensor(add, 1, 0, 9223372036854775807, 2);  add = None
        convert_element_type_2: "f32[4000, 2880]" = torch.ops.prims.convert_element_type.default(slice_2, torch.float32);  slice_2 = None
        clamp_max_2: "f32[4000, 2880]" = torch.ops.aten.clamp_max.default(convert_element_type_2, 7.0);  convert_element_type_2 = None
        convert_element_type_3: "bf16[4000, 2880]" = torch.ops.prims.convert_element_type.default(clamp_max_2, torch.bfloat16);  clamp_max_2 = None
        mul: "bf16[4000, 2880]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.702)
        sigmoid: "bf16[4000, 2880]" = torch.ops.aten.sigmoid.default(mul);  mul = None
        mul_1: "bf16[4000, 2880]" = torch.ops.aten.mul.Tensor(convert_element_type_3, sigmoid);  convert_element_type_3 = sigmoid = None
        mul_2: "bf16[4000, 2880]" = torch.ops.aten.mul.Tensor(add_1, mul_1);  add_1 = mul_1 = None
        _grouped_mm_1: "bf16[4000, 2880]" = torch.ops.aten._grouped_mm.default(mul_2, arg6_1, arg4_1);  mul_2 = arg6_1 = arg4_1 = None
        index_2: "bf16[4000, 2880]" = torch.ops.aten.index.Tensor(arg7_1, [clamp_max]);  arg7_1 = clamp_max = None
        add_2: "bf16[4000, 2880]" = torch.ops.aten.add.Tensor(_grouped_mm_1, index_2);  _grouped_mm_1 = index_2 = None
        convert_element_type_4: "f32[1000, 4]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.float32);  arg8_1 = None
        amax: "f32[1000, 1]" = torch.ops.aten.amax.default(convert_element_type_4, [1], True)
        sub: "f32[1000, 4]" = torch.ops.aten.sub.Tensor(convert_element_type_4, amax);  convert_element_type_4 = amax = None
        exp: "f32[1000, 4]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[1000, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True)
        div_1: "f32[1000, 4]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_5: "bf16[1000, 4]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        view: "bf16[4000]" = torch.ops.aten.view.default(convert_element_type_5, [-1]);  convert_element_type_5 = None
        index_3: "bf16[4000]" = torch.ops.aten.index.Tensor(view, [arg1_1]);  view = None
        unsqueeze_1: "bf16[4000, 1]" = torch.ops.aten.unsqueeze.default(index_3, -1);  index_3 = None
        mul_3: "bf16[4000, 2880]" = torch.ops.aten.mul.Tensor(add_2, unsqueeze_1);  add_2 = unsqueeze_1 = None
        where_1: "bf16[4000, 2880]" = torch.ops.aten.where.self(unsqueeze, full, mul_3);  unsqueeze = full = mul_3 = None
        empty: "i64[4000]" = torch.ops.aten.empty.memory_format(_shape_param_0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        iota: "i64[4000]" = torch.ops.prims.iota.default(4000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_put: "i64[4000]" = torch.ops.aten.index_put.default(empty, [arg1_1], iota);  empty = arg1_1 = iota = None
        index_4: "bf16[4000, 2880]" = torch.ops.aten.index.Tensor(where_1, [index_put]);  where_1 = index_put = None
        view_1: "bf16[1000, 4, 2880]" = torch.ops.aten.view.default(index_4, _shape_param_1);  index_4 = _shape_param_1 = None
        sum_2: "bf16[1000, 2880]" = torch.ops.aten.sum.dim_IntList(view_1, [1]);  view_1 = None
        view_2: "bf16[1, 1000, 2880]" = torch.ops.aten.view.default(sum_2, _shape_param_2);  sum_2 = _shape_param_2 = None
        add_3: "bf16[1, 1000, 2880]" = torch.ops.aten.add.Tensor(arg9_1, view_2);  arg9_1 = view_2 = None
        convert_element_type_6: "f32[1, 1000, 2880]" = torch.ops.prims.convert_element_type.default(add_3, torch.float32);  add_3 = None
        pow_1: "f32[1, 1000, 2880]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_6, 2)
        mean: "f32[1, 1000, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add_4: "f32[1, 1000, 1]" = torch.ops.aten.add.Tensor(mean, 1e-05);  mean = None
        rsqrt: "f32[1, 1000, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_4: "f32[1, 1000, 2880]" = torch.ops.aten.mul.Tensor(convert_element_type_6, rsqrt);  convert_element_type_6 = rsqrt = None
        mul_5: "f32[1, 1000, 2880]" = torch.ops.aten.mul.Tensor(arg10_1, mul_4);  arg10_1 = mul_4 = None
        convert_element_type_7: "bf16[1, 1000, 2880]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        view_3: "bf16[1000, 2880]" = torch.ops.aten.view.default(convert_element_type_7, _shape_param_3);  convert_element_type_7 = _shape_param_3 = None
        permute: "bf16[2880, 201088]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        mm: "bf16[1000, 201088]" = torch.ops.aten.mm.default(view_3, permute);  view_3 = permute = None
        view_4: "bf16[1, 1000, 201088]" = torch.ops.aten.view.default(mm, _shape_param_4);  mm = _shape_param_4 = None
        return view_4



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
