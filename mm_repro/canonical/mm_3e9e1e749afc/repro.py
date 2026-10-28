"""
Standalone repro captured via capture_hook.
Label: repros/models/timm/train/mobilevit_s/full_graph_001.py
Pattern hash: 3e9e1e749afc
Shape hash: 6c5f0460
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
    def forward(self, arg0_1: "bf16[8192, 480]", arg1_1: "bf16[480, 240]", arg2_1: "f32[240]", arg3_1: "bf16[512, 16, 240]", arg4_1: "f32[512, 16, 1]", arg5_1: "f32[512, 16, 1]", _shape_param_0):
        # No stacktrace found for following nodes
        mm: "bf16[8192, 240]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[512, 16, 240]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[512, 16, 240]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        mul: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(convert_element_type, arg2_1);  arg2_1 = None
        mul_1: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(mul, 240)
        convert_element_type_1: "f32[512, 16, 240]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        sub: "f32[512, 16, 240]" = torch.ops.aten.sub.Tensor(convert_element_type_1, arg4_1);  convert_element_type_1 = arg4_1 = None
        mul_2: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(sub, arg5_1);  sub = arg5_1 = None
        mul_3: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(mul, mul_2)
        mul_4: "f32[512, 16, 240]" = torch.ops.aten.mul.Tensor(convert_element_type, mul_2);  convert_element_type = mul_2 = None
        return (mul, mul_1, mul_3, mul_4)



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
