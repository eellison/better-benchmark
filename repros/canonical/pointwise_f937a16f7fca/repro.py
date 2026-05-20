"""
Standalone repro captured via capture_hook.
Label: inductor_torchbench_perf-6-6-linux.aws.a100_graph28
Pattern hash: f937a16f7fca
Shape hash: adb46d13
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([256, 1024], f16), T([2], f32), T([2, 1024], f32))"

class Repro(torch.nn.Module):
    def forward(self, addmm_1: "f16[256, 1024]", arg6_1: "f32[2]", arg5_1: "f32[2, 1024]"):
        # No stacktrace found for following nodes
        relu_default: "f16[256, 1024]" = torch.ops.aten.relu.default(addmm_1);  addmm_1 = None
        convert_element_type_default: "f16[2]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.float16);  arg6_1 = None
        convert_element_type_default_1: "f16[2, 1024]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.float16);  arg5_1 = None
        permute_default: "f16[1024, 2]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        return (relu_default, convert_element_type_default, permute_default)


def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
