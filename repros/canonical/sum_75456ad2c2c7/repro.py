"""
Standalone repro captured via capture_hook.
Label: hf_TrOCRForCausalLM_train
Pattern hash: 75456ad2c2c7
Shape hash: b0ff1d38
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
    def forward(self, arg0_1: "bf16[1024, 256, 256]", arg1_1: "bf16[1024, 256, 256]", arg2_1: "f32[64, 1, 256, 256]", arg3_1: "f32[1024, 256, 1]", arg4_1: "f32[1024, 256, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        convert_element_type: "f32[1024, 256, 256]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        view: "bf16[64, 16, 256, 256]" = torch.ops.aten.view.default(arg1_1, _shape_param_0);  arg1_1 = _shape_param_0 = None
        add: "f32[64, 16, 256, 256]" = torch.ops.aten.add.Tensor(view, arg2_1);  view = arg2_1 = None
        view_1: "f32[1024, 256, 256]" = torch.ops.aten.view.default(add, _shape_param_1);  add = _shape_param_1 = None
        sub: "f32[1024, 256, 256]" = torch.ops.aten.sub.Tensor(view_1, arg3_1);  view_1 = arg3_1 = None
        exp: "f32[1024, 256, 256]" = torch.ops.aten.exp.default(sub);  sub = None
        div: "f32[1024, 256, 256]" = torch.ops.aten.div.Tensor(exp, arg4_1);  exp = arg4_1 = None
        mul: "f32[1024, 256, 256]" = torch.ops.aten.mul.Tensor(convert_element_type, div);  convert_element_type = None
        sum_1: "f32[1024, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul, [-1], True)
        neg: "f32[1024, 256, 256]" = torch.ops.aten.neg.default(div);  div = None
        fma: "f32[1024, 256, 256]" = torch.ops.prims.fma.default(neg, sum_1, mul);  neg = sum_1 = mul = None
        view_2: "f32[64, 16, 256, 256]" = torch.ops.aten.view.default(fma, _shape_param_2);  fma = _shape_param_2 = None
        convert_element_type_1: "bf16[64, 16, 256, 256]" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        view_3: "bf16[1024, 256, 256]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_3);  convert_element_type_1 = _shape_param_3 = None
        return view_3



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
