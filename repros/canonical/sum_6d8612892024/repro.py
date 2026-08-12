"""
Standalone repro captured via capture_hook.
Label: hf_DistilBertForMaskedLM_train
Pattern hash: 6d8612892024
Shape hash: 931c2d63
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
    def forward(self, arg0_1: "bf16[3072, 128, 128]", arg1_1: "b8[256, 12, 128, 128]", arg2_1: "bf16[3072, 128, 128]", arg3_1: "f32[256, 12, 128, 1]", arg4_1: "f32[256, 12, 128, 1]", arg5_1: "b8[256, 12, 128, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view: "bf16[256, 12, 128, 128]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "bf16[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        mul: "bf16[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
        mul_1: "bf16[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view, mul);  view = mul = None
        convert_element_type_1: "f32[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_1, torch.float32);  mul_1 = None
        view_1: "bf16[256, 12, 128, 128]" = torch.ops.aten.view.default(arg2_1, _shape_param_1);  arg2_1 = _shape_param_1 = None
        convert_element_type_2: "f32[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        sub: "f32[256, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_2, arg3_1);  convert_element_type_2 = arg3_1 = None
        exp: "f32[256, 12, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        div: "f32[256, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp, arg4_1);  exp = arg4_1 = None
        convert_element_type_3: "bf16[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        full: "bf16[256, 12, 128, 128]" = torch.ops.aten.full.default(_shape_param_2, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_2 = None
        where: "bf16[256, 12, 128, 128]" = torch.ops.aten.where.self(arg5_1, full, convert_element_type_3);  arg5_1 = full = convert_element_type_3 = None
        convert_element_type_4: "f32[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        mul_2: "f32[256, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_1, convert_element_type_4);  convert_element_type_1 = None
        sum_1: "f32[256, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [-1], True)
        neg: "f32[256, 12, 128, 128]" = torch.ops.aten.neg.default(convert_element_type_4);  convert_element_type_4 = None
        fma: "f32[256, 12, 128, 128]" = torch.ops.prims.fma.default(neg, sum_1, mul_2);  neg = sum_1 = mul_2 = None
        convert_element_type_5: "bf16[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        view_2: "bf16[3072, 128, 128]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_3);  convert_element_type_5 = _shape_param_3 = None
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
