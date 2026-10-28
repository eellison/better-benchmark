"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/MT5ForConditionalGeneration/full_graph_001.py
Pattern hash: 97715b15aa18
Shape hash: 5019a1db
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
    def forward(self, arg0_1: "bf16[4096, 384]", arg1_1: "bf16[384, 512]", arg2_1: "bf16[4096, 384]", arg3_1: "bf16[384, 512]", arg4_1: "bf16[4096, 384]", arg5_1: "bf16[384, 512]", arg6_1: "f32[512]", arg7_1: "f32[32, 128, 512]", arg8_1: "f32[32, 128, 1]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        mm: "bf16[4096, 512]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        mm_1: "bf16[4096, 512]" = torch.ops.aten.mm.default(arg2_1, arg3_1);  arg2_1 = arg3_1 = None
        view_1: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_1, _shape_param_1);  mm_1 = _shape_param_1 = None
        convert_element_type_1: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        add: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type, convert_element_type_1);  convert_element_type = convert_element_type_1 = None
        mm_2: "bf16[4096, 512]" = torch.ops.aten.mm.default(arg4_1, arg5_1);  arg4_1 = arg5_1 = None
        view_2: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_2, _shape_param_2);  mm_2 = _shape_param_2 = None
        convert_element_type_2: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        add_1: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add, convert_element_type_2);  add = convert_element_type_2 = None
        mul: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_1, arg6_1);  arg6_1 = None
        mul_1: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg7_1, arg8_1)
        mul_2: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_1, mul_1);  add_1 = mul_1 = None
        mul_3: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul, arg7_1);  arg7_1 = None
        mul_4: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul, arg8_1);  mul = arg8_1 = None
        return (mul_2, mul_3, mul_4)



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
