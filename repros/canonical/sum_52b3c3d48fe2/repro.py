"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s1_g220
Pattern hash: 52b3c3d48fe2
Shape hash: 9384f5f4
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg3_1: "f32[10000, 64]", mm: "f16[64, 64]", _shape_param_0):
        # No stacktrace found for following nodes
        sum_dim_int_list: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(arg3_1, [0], True, dtype = torch.float32);  arg3_1 = None
        reshape_default: "f32[64]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None
        convert_element_type_default: "f32[64, 64]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        _output_to_half_0: "f16[64]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float16);  reshape_default = None
        _output_to_half_1: "f16[64, 64]" = torch.ops.prims.convert_element_type.default(convert_element_type_default, torch.float16);  convert_element_type_default = None
        return (_output_to_half_0, _output_to_half_1)


def _default_make_inputs():
    return [
    torch.randn([10000, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float16, device='cuda'),
    [64],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
