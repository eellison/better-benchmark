"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train
Pattern hash: 7ab5c91b014a
Shape hash: 801fb66e
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
    def forward(self, arg0_1: "bf16[2048, 768]", arg1_1: "f32[768]", arg2_1: "f32[16, 128, 768]", arg3_1: "f32[16, 128, 1]", arg4_1: "f32[16, 128, 768]", arg5_1: "b8[16, 128, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        view: "bf16[16, 128, 768]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        sum_1: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(convert_element_type, [0, 1], True, dtype = torch.float32)
        view_1: "f32[768]" = torch.ops.aten.view.default(sum_1, _shape_param_1);  sum_1 = _shape_param_1 = None
        mul: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg1_1, arg2_1)
        add: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(arg3_1, 1e-06)
        div: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(mul, add);  mul = None
        div_1: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(div, add);  div = None
        neg: "f32[16, 128, 768]" = torch.ops.aten.neg.default(convert_element_type)
        mul_1: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(neg, div_1);  neg = div_1 = None
        div_2: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(convert_element_type, add);  convert_element_type = add = None
        sum_2: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_1, [2], True, dtype = torch.float32);  mul_1 = None
        mul_2: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_2, arg1_1);  arg1_1 = None
        mul_3: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_2, arg2_1);  div_2 = None
        sum_3: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_3, [0, 1], True, dtype = torch.float32);  mul_3 = None
        view_2: "f32[768]" = torch.ops.aten.view.default(sum_3, _shape_param_2);  sum_3 = _shape_param_2 = None
        neg_1: "f32[16, 128, 768]" = torch.ops.aten.neg.default(mul_2)
        sum_4: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_1, [2], True, dtype = torch.float32);  neg_1 = None
        add_1: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(arg4_1, mul_2);  arg4_1 = mul_2 = None
        mul_4: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(arg3_1, 2)
        div_3: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(sum_2, mul_4);  sum_2 = mul_4 = None
        eq: "b8[16, 128, 1]" = torch.ops.aten.eq.Scalar(arg3_1, 0);  arg3_1 = None
        full: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[16, 128, 1]" = torch.ops.aten.where.self(eq, full, div_3);  eq = div_3 = None
        mul_5: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(where, 0.002607561929595828);  where = None
        mul_6: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(mul_5, arg2_1);  mul_5 = arg2_1 = None
        add_2: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_1, mul_6);  add_1 = mul_6 = None
        expand: "f32[16, 128, 768]" = torch.ops.aten.expand.default(sum_4, _shape_param_3);  sum_4 = _shape_param_3 = None
        div_4: "f32[16, 128, 768]" = torch.ops.aten.div.Scalar(expand, 768);  expand = None
        add_3: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_2, div_4);  add_2 = div_4 = None
        convert_element_type_1: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16)
        convert_element_type_2: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        mul_7: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
        mul_8: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_1, mul_7);  convert_element_type_1 = mul_7 = None
        view_3: "bf16[2048, 768]" = torch.ops.aten.view.default(mul_8, _shape_param_4);  mul_8 = _shape_param_4 = None
        permute: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_3, [1, 0])
        sum_5: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_3, [0], True, dtype = torch.float32)
        view_4: "f32[768]" = torch.ops.aten.view.default(sum_5, _shape_param_5);  sum_5 = _shape_param_5 = None
        convert_element_type_3: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_4, torch.bfloat16);  view_4 = None
        convert_element_type_4: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        return (view_1, view_2, full, add_3, view_3, permute, convert_element_type_4)



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
