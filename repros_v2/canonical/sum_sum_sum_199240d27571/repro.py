"""
Standalone repro captured via capture_hook.
Label: hf_DebertaV2ForMaskedLM_train
Pattern hash: 199240d27571
Shape hash: 49fb3d4b
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
    def forward(self, arg0_1: "bf16[4096, 1536]", arg1_1: "f32[1536]", arg2_1: "bf16[4096, 1536]", arg3_1: "f32[8, 512, 1]", arg4_1: "f32[8, 512, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        mul: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type, arg1_1);  arg1_1 = None
        mul_1: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul, 1536)
        sum_1: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        view_1: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(arg2_1, _shape_param_1);  arg2_1 = _shape_param_1 = None
        convert_element_type_1: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        mul_2: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 0.5)
        mul_3: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 0.7071067811865476)
        erf: "f32[8, 512, 1536]" = torch.ops.aten.erf.default(mul_3);  mul_3 = None
        add: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_4: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_2, add);  mul_2 = None
        convert_element_type_2: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        convert_element_type_3: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        sub: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(convert_element_type_3, arg3_1);  convert_element_type_3 = arg3_1 = None
        mul_5: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub, arg4_1);  sub = None
        mul_6: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul, mul_5);  mul = None
        sum_2: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_6, [2], True);  mul_6 = None
        mul_7: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_5, sum_2);  sum_2 = None
        sub_1: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_2: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_1, mul_7);  sub_1 = mul_7 = None
        div: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(arg4_1, 1536);  arg4_1 = None
        mul_8: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div, sub_2);  div = sub_2 = None
        mul_9: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type, mul_5);  mul_5 = None
        sum_3: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_9, [0, 1]);  mul_9 = None
        sum_4: "f32[1536]" = torch.ops.aten.sum.dim_IntList(convert_element_type, [0, 1]);  convert_element_type = None
        convert_element_type_4: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None
        convert_element_type_5: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_4, torch.float32);  convert_element_type_4 = None
        mul_10: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add, 0.5);  add = None
        mul_11: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_1, convert_element_type_1)
        mul_12: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_11, -0.5);  mul_11 = None
        exp: "f32[8, 512, 1536]" = torch.ops.aten.exp.default(mul_12);  mul_12 = None
        mul_13: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_14: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_1, mul_13);  convert_element_type_1 = mul_13 = None
        add_1: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_10, mul_14);  mul_10 = mul_14 = None
        mul_15: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_5, add_1);  convert_element_type_5 = add_1 = None
        convert_element_type_6: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_15, torch.bfloat16);  mul_15 = None
        view_2: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_6, _shape_param_2);  convert_element_type_6 = _shape_param_2 = None
        permute: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_2, [1, 0])
        sum_5: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_2, [0], True, dtype = torch.float32)
        view_3: "f32[1536]" = torch.ops.aten.view.default(sum_5, _shape_param_3);  sum_5 = _shape_param_3 = None
        convert_element_type_7: "bf16[1536]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        convert_element_type_8: "f32[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_7, torch.float32);  convert_element_type_7 = None
        return (sum_3, sum_4, view_2, permute, convert_element_type_8)



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
