"""Curated MM repro. Pattern hash: e31835620776."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[4000]", arg1_1: "bf16[32, 5760]", arg2_1: "bf16[4000, 5760]", arg3_1: "bf16[32, 2880, 2880]", arg4_1: "i32[32]", arg5_1: "bf16[32, 2880]", arg6_1: "bf16[1000, 4]", arg7_1: "i64[4000]", arg8_1: "b8[4000, 1]", arg9_1: "bf16[1, 1000, 2880]", arg10_1: "bf16[2880]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        full: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        clamp_max: "i64[4000]" = torch.ops.aten.clamp_max.default(arg0_1, 31);  arg0_1 = None
        index: "bf16[4000, 5760]" = torch.ops.aten.index.Tensor(arg1_1, [clamp_max]);  arg1_1 = None
        add: "bf16[4000, 5760]" = torch.ops.aten.add.Tensor(arg2_1, index);  arg2_1 = index = None
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
        _grouped_mm: "bf16[4000, 2880]" = torch.ops.aten._grouped_mm.default(mul_2, arg3_1, arg4_1);  mul_2 = arg3_1 = arg4_1 = None
        index_1: "bf16[4000, 2880]" = torch.ops.aten.index.Tensor(arg5_1, [clamp_max]);  arg5_1 = clamp_max = None
        add_2: "bf16[4000, 2880]" = torch.ops.aten.add.Tensor(_grouped_mm, index_1);  _grouped_mm = index_1 = None
        convert_element_type_4: "f32[1000, 4]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.float32);  arg6_1 = None
        amax: "f32[1000, 1]" = torch.ops.aten.amax.default(convert_element_type_4, [1], True)
        sub: "f32[1000, 4]" = torch.ops.aten.sub.Tensor(convert_element_type_4, amax);  convert_element_type_4 = amax = None
        exp: "f32[1000, 4]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[1000, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True)
        div: "f32[1000, 4]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_5: "bf16[1000, 4]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        view: "bf16[4000]" = torch.ops.aten.view.default(convert_element_type_5, [-1]);  convert_element_type_5 = None
        index_2: "bf16[4000]" = torch.ops.aten.index.Tensor(view, [arg7_1]);  view = None
        unsqueeze: "bf16[4000, 1]" = torch.ops.aten.unsqueeze.default(index_2, -1);  index_2 = None
        mul_3: "bf16[4000, 2880]" = torch.ops.aten.mul.Tensor(add_2, unsqueeze);  add_2 = unsqueeze = None
        where: "bf16[4000, 2880]" = torch.ops.aten.where.self(arg8_1, full, mul_3);  arg8_1 = full = mul_3 = None
        empty: "i64[4000]" = torch.ops.aten.empty.memory_format(_shape_param_0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        iota: "i64[4000]" = torch.ops.prims.iota.default(4000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_put: "i64[4000]" = torch.ops.aten.index_put.default(empty, [arg7_1], iota);  empty = arg7_1 = iota = None
        index_3: "bf16[4000, 2880]" = torch.ops.aten.index.Tensor(where, [index_put]);  where = index_put = None
        view_1: "bf16[1000, 4, 2880]" = torch.ops.aten.view.default(index_3, _shape_param_1);  index_3 = _shape_param_1 = None
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
        return convert_element_type_7

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
