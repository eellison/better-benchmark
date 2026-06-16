"""
Standalone repro captured via capture_hook.
Label: hf_DebertaV2ForMaskedLM_train
Pattern hash: 9b2fcee49b0a
Shape hash: 8dadb13a
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
    def forward(self, arg0_1: "bf16[192, 512, 512]", arg1_1: "b8[8, 24, 512, 512]", arg2_1: "bf16[192, 512, 512]", arg3_1: "b8[8, 1, 512, 512]", arg4_1: "f32[8, 24, 512, 1]", arg5_1: "f32[8, 24, 512, 1]", arg6_1: "bf16[]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        convert_element_type: "f32[192, 512, 512]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        view: "f32[8, 24, 512, 512]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        convert_element_type_1: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        mul: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_1: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view, mul);  view = mul = None
        view_1: "bf16[8, 24, 512, 512]" = torch.ops.aten.view.default(arg2_1, _shape_param_1);  arg2_1 = _shape_param_1 = None
        full: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(arg3_1, full, view_1);  full = view_1 = None
        convert_element_type_2: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        sub: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_2, arg4_1);  convert_element_type_2 = arg4_1 = None
        exp: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub);  sub = None
        div: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp, arg5_1);  exp = arg5_1 = None
        mul_2: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_1, div);  mul_1 = None
        sum_1: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [-1], True)
        neg: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div);  div = None
        fma: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg, sum_1, mul_2);  neg = sum_1 = mul_2 = None
        convert_element_type_3: "bf16[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        where_1: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(arg3_1, arg6_1, convert_element_type_3);  arg3_1 = arg6_1 = convert_element_type_3 = None
        view_2: "bf16[192, 512, 512]" = torch.ops.aten.view.default(where_1, _shape_param_2);  where_1 = _shape_param_2 = None
        return view_2



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
