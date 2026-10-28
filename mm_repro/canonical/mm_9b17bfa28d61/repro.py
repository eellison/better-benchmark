"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/DistillGPT2/full_graph_003.py
Pattern hash: 9b17bfa28d61
Shape hash: 93155588
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
    def forward(self, arg0_1: "bf16[16384, 3072]", arg1_1: "bf16[3072, 768]", arg2_1: "f32[768]", arg3_1: "f32[32, 512, 768]", _shape_param_0):
        # No stacktrace found for following nodes
        mm: "bf16[16384, 768]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[16384, 768]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        view: "f32[32, 512, 768]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view, arg2_1);  arg2_1 = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, 768)
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, arg3_1)
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view, arg3_1);  view = arg3_1 = None
        return (mul, mul_1, mul_2, mul_3)



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
