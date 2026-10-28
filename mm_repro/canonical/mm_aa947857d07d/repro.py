"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/OPTForCausalLM/full_graph_005.py
Pattern hash: aa947857d07d
Shape hash: b076df1c
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
    def forward(self, arg0_1: "bf16[8192, 3072]", arg1_1: "bf16[3072, 768]", arg2_1: "f32[768]", arg3_1: "bf16[8192, 768]", arg4_1: "b8[4, 2048, 768]", arg5_1: "f32[4, 2048, 768]", arg6_1: "f32[8192, 1]", arg7_1: "f32[8192, 1]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        mm: "bf16[8192, 768]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[8192, 768]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        mul: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, arg2_1);  arg2_1 = None
        mul_1: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(mul, 768)
        view: "bf16[4, 2048, 768]" = torch.ops.aten.view.default(arg3_1, _shape_param_0);  arg3_1 = _shape_param_0 = None
        mul_2: "bf16[4, 2048, 768]" = torch.ops.aten.mul.Tensor(arg4_1, view);  arg4_1 = view = None
        mul_3: "bf16[4, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None
        add: "f32[4, 2048, 768]" = torch.ops.aten.add.Tensor(arg5_1, mul_3);  arg5_1 = mul_3 = None
        view_1: "f32[8192, 768]" = torch.ops.aten.view.default(add, _shape_param_1);  add = _shape_param_1 = None
        sub: "f32[8192, 768]" = torch.ops.aten.sub.Tensor(view_1, arg6_1);  view_1 = arg6_1 = None
        mul_4: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(sub, arg7_1);  sub = arg7_1 = None
        mul_5: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(mul, mul_4)
        mul_6: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, mul_4);  convert_element_type = mul_4 = None
        return (mul, mul_1, mul_5, mul_6)



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
