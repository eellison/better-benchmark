"""
Standalone repro captured via capture_hook.
Label: torchbench_tts_angular_train
Pattern hash: b822d095bf9f
Shape hash: 6cbb208b
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
    def forward(self, arg0_1: "bf16[64, 50, 256]", _shape_param_0):
        # No stacktrace found for following nodes
        select: "bf16[64, 256]" = torch.ops.aten.select.int(arg0_1, 1, -1);  arg0_1 = None
        convert_element_type: "f32[64, 256]" = torch.ops.prims.convert_element_type.default(select, torch.float32)
        pow_1: "f32[64, 256]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2);  convert_element_type = None
        sum_1: "f32[64, 1]" = torch.ops.aten.sum.dim_IntList(pow_1, [1], True);  pow_1 = None
        pow_2: "f32[64, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_1, 0.5);  sum_1 = None
        clamp_min: "f32[64, 1]" = torch.ops.aten.clamp_min.default(pow_2, 1e-12)
        expand: "f32[64, 256]" = torch.ops.aten.expand.default(clamp_min, _shape_param_0);  clamp_min = _shape_param_0 = None
        div: "f32[64, 256]" = torch.ops.aten.div.Tensor(select, expand);  expand = None
        return (select, pow_2, div)



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
