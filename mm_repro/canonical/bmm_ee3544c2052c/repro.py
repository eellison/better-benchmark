"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/YituTechConvBert/full_graph_001.py
Pattern hash: ee3544c2052c
Shape hash: 7403002d
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
    def forward(self, arg0_1: "bf16[98304, 9, 64]", arg1_1: "bf16[98304, 64, 1]", arg2_1: "f32[54]", arg3_1: "bf16[16384, 54]", arg4_1: "f32[98304, 1, 1]", arg5_1: "f32[98304, 1, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        bmm: "bf16[98304, 9, 1]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "bf16[54]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        view: "bf16[32, 512, 54]" = torch.ops.aten.view.default(arg3_1, _shape_param_0);  arg3_1 = _shape_param_0 = None
        add: "bf16[32, 512, 54]" = torch.ops.aten.add.Tensor(view, convert_element_type);  view = convert_element_type = None
        view_1: "bf16[98304, 9, 1]" = torch.ops.aten.view.default(add, _shape_param_1);  add = _shape_param_1 = None
        convert_element_type_1: "f32[98304, 9, 1]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        sub: "f32[98304, 9, 1]" = torch.ops.aten.sub.Tensor(convert_element_type_1, arg4_1);  convert_element_type_1 = arg4_1 = None
        exp: "f32[98304, 9, 1]" = torch.ops.aten.exp.default(sub);  sub = None
        div: "f32[98304, 9, 1]" = torch.ops.aten.div.Tensor(exp, arg5_1);  exp = arg5_1 = None
        convert_element_type_2: "f32[98304, 9, 1]" = torch.ops.prims.convert_element_type.default(bmm, torch.float32);  bmm = None
        mul: "f32[98304, 9, 1]" = torch.ops.aten.mul.Tensor(convert_element_type_2, div);  convert_element_type_2 = None
        sum_1: "f32[98304, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul, [1], True)
        neg: "f32[98304, 9, 1]" = torch.ops.aten.neg.default(div);  div = None
        fma: "f32[98304, 9, 1]" = torch.ops.prims.fma.default(neg, sum_1, mul);  neg = sum_1 = mul = None
        convert_element_type_3: "bf16[98304, 9, 1]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        view_2: "bf16[32, 512, 54]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_2);  convert_element_type_3 = _shape_param_2 = None
        view_3: "bf16[16384, 54]" = torch.ops.aten.view.default(view_2, _shape_param_3);  _shape_param_3 = None
        permute: "bf16[54, 16384]" = torch.ops.aten.permute.default(view_3, [1, 0])
        return (view_2, view_3, permute)



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
