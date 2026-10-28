"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/T5ForConditionalGeneration/full_graph_001.py
Pattern hash: 8ac43cbc8a33
Shape hash: e3d169cf
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
    def forward(self, arg0_1: "bf16[64, 1024, 64]", arg1_1: "bf16[64, 64, 1024]", arg2_1: "b8[8, 8, 1024, 1024]", arg3_1: "bf16[64, 1024, 1024]", arg4_1: "f32[8, 8, 1024, 1]", arg5_1: "f32[8, 8, 1024, 1]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        bmm: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        convert_element_type: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        mul: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
        mul_1: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view, mul);  view = mul = None
        convert_element_type_1: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_1, torch.float32);  mul_1 = None
        view_1: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(arg3_1, _shape_param_1);  arg3_1 = _shape_param_1 = None
        convert_element_type_2: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        convert_element_type_3: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        sub: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_3, arg4_1);  convert_element_type_3 = arg4_1 = None
        exp: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub);  sub = None
        div: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp, arg5_1);  exp = arg5_1 = None
        mul_2: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1, div);  convert_element_type_1 = div = None
        return mul_2



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
