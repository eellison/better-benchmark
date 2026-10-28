"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/MT5ForConditionalGeneration/full_graph_001.py
Pattern hash: 25e9e8b3aae0
Shape hash: efc2e1c1
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
    def forward(self, arg0_1: "bf16[4096, 250112]", arg1_1: "bf16[250112, 512]", arg2_1: "b8[32, 128, 512]", arg3_1: "f32[512]", arg4_1: "f32[32, 128, 512]", arg5_1: "f32[32, 128, 1]", _shape_param_0):
        # No stacktrace found for following nodes
        mm: "bf16[4096, 512]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        convert_element_type_1: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        mul: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_1: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type, mul);  convert_element_type = mul = None
        mul_2: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_1, arg3_1);  arg3_1 = None
        mul_3: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg4_1, arg5_1)
        mul_4: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_1, mul_3);  mul_1 = mul_3 = None
        mul_5: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_2, arg4_1);  arg4_1 = None
        mul_6: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_2, arg5_1);  mul_2 = arg5_1 = None
        return (mul_4, mul_5, mul_6)



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
