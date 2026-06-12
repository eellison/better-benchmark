"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train
Pattern hash: 200f6e0136dd
Shape hash: 3d639bb6
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
    def forward(self, arg0_1: "bf16[2048, 768]", arg1_1: "bf16[2048, 768]", arg2_1: "bf16[2048, 768]", arg3_1: "f32[768]", arg4_1: "f32[16, 128, 768]", arg5_1: "f32[16, 128, 1]", arg6_1: "f32[16, 128, 768]", arg7_1: "f32[]", arg8_1: "b8[16, 128, 768]", arg9_1: "i64[16, 128]", arg10_1: "i64[16, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # No stacktrace found for following nodes
        view: "bf16[16, 128, 768]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        view_1: "bf16[16, 128, 768]" = torch.ops.aten.view.default(arg1_1, _shape_param_1);  arg1_1 = _shape_param_1 = None
        convert_element_type_1: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        add: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(convert_element_type, convert_element_type_1);  convert_element_type = convert_element_type_1 = None
        view_2: "bf16[16, 128, 768]" = torch.ops.aten.view.default(arg2_1, _shape_param_2);  arg2_1 = _shape_param_2 = None
        convert_element_type_2: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        add_1: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add, convert_element_type_2);  add = convert_element_type_2 = None
        sum_1: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_1, [0, 1], True, dtype = torch.float32)
        view_3: "f32[768]" = torch.ops.aten.view.default(sum_1, _shape_param_3);  sum_1 = _shape_param_3 = None
        mul: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg3_1, arg4_1)
        add_2: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(arg5_1, 1e-06)
        div: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(mul, add_2);  mul = None
        div_1: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(div, add_2);  div = None
        neg: "f32[16, 128, 768]" = torch.ops.aten.neg.default(add_1)
        mul_1: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(neg, div_1);  neg = div_1 = None
        div_2: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(add_1, add_2);  add_1 = add_2 = None
        sum_2: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_1, [2], True, dtype = torch.float32);  mul_1 = None
        mul_2: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_2, arg3_1);  arg3_1 = None
        mul_3: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_2, arg4_1);  div_2 = None
        sum_3: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_3, [0, 1], True, dtype = torch.float32);  mul_3 = None
        view_4: "f32[768]" = torch.ops.aten.view.default(sum_3, _shape_param_4);  sum_3 = _shape_param_4 = None
        neg_1: "f32[16, 128, 768]" = torch.ops.aten.neg.default(mul_2)
        sum_4: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_1, [2], True, dtype = torch.float32);  neg_1 = None
        add_3: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(arg6_1, mul_2);  arg6_1 = mul_2 = None
        mul_4: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(arg5_1, 2)
        div_3: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(sum_2, mul_4);  sum_2 = mul_4 = None
        eq: "b8[16, 128, 1]" = torch.ops.aten.eq.Scalar(arg5_1, 0);  arg5_1 = None
        where: "f32[16, 128, 1]" = torch.ops.aten.where.self(eq, arg7_1, div_3);  eq = arg7_1 = div_3 = None
        mul_5: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(where, 0.002607561929595828);  where = None
        mul_6: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(mul_5, arg4_1);  mul_5 = arg4_1 = None
        add_4: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_3, mul_6);  add_3 = mul_6 = None
        expand: "f32[16, 128, 768]" = torch.ops.aten.expand.default(sum_4, _shape_param_5);  sum_4 = _shape_param_5 = None
        div_4: "f32[16, 128, 768]" = torch.ops.aten.div.Scalar(expand, 768);  expand = None
        add_5: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_4, div_4);  add_4 = div_4 = None
        convert_element_type_3: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.float32);  arg8_1 = None
        mul_7: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.1111111111111112);  convert_element_type_3 = None
        mul_8: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(add_5, mul_7);  add_5 = mul_7 = None
        ge: "b8[16, 128]" = torch.ops.aten.ge.Scalar(arg9_1, 0)
        lt: "b8[16, 128]" = torch.ops.aten.lt.Scalar(arg9_1, 3)
        bitwise_and: "b8[16, 128]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne: "b8[16, 128]" = torch.ops.aten.ne.Scalar(arg9_1, 0)
        bitwise_and_1: "b8[16, 128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne);  bitwise_and = ne = None
        unsqueeze: "b8[16, 128, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full: "f32[3, 768]" = torch.ops.aten.full.default(_shape_param_6, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_6 = None
        _unsafe_masked_index_put_accumulate: "f32[3, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full, unsqueeze, [arg9_1], mul_8);  full = unsqueeze = arg9_1 = None
        ge_1: "b8[16, 128]" = torch.ops.aten.ge.Scalar(arg10_1, 0)
        lt_1: "b8[16, 128]" = torch.ops.aten.lt.Scalar(arg10_1, 20005)
        bitwise_and_2: "b8[16, 128]" = torch.ops.aten.bitwise_and.Tensor(ge_1, lt_1);  ge_1 = lt_1 = None
        ne_1: "b8[16, 128]" = torch.ops.aten.ne.Scalar(arg10_1, 0)
        bitwise_and_3: "b8[16, 128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, ne_1);  bitwise_and_2 = ne_1 = None
        unsqueeze_1: "b8[16, 128, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_3, -1);  bitwise_and_3 = None
        full_1: "f32[20005, 768]" = torch.ops.aten.full.default(_shape_param_7, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_7 = None
        _unsafe_masked_index_put_accumulate_1: "f32[20005, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_1, unsqueeze_1, [arg10_1], mul_8);  full_1 = unsqueeze_1 = arg10_1 = mul_8 = None
        return (view_3, view_4, _unsafe_masked_index_put_accumulate, _unsafe_masked_index_put_accumulate_1)



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
