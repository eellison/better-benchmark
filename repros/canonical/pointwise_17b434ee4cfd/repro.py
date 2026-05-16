"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s5_g20
Pattern hash: 17b434ee4cfd
Shape hash: 627f64b9
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_1: "f16[4, 16]", arg8_1: "f32[16]", arg7_1: "f32[16, 16]"):
        # No stacktrace found for following nodes
        tanh_default: "f16[4, 16]" = torch.ops.aten.tanh.default(addmm_1);  addmm_1 = None
        convert_element_type_default: "f16[16]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.float16);  arg8_1 = None
        convert_element_type_default_1: "f16[16, 16]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.float16);  arg7_1 = None
        permute_default: "f16[16, 16]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        return (tanh_default, convert_element_type_default, permute_default)


def _default_make_inputs():
    return [
    torch.randn([4, 16], dtype=torch.float16, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16, 16], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
