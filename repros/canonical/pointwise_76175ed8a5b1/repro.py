"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s5_g20
Pattern hash: 76175ed8a5b1
Shape hash: 4d1cc12d
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_2: "f16[4, 16]", arg10_1: "f32[1]", arg9_1: "f32[1, 16]", permute_3: "f16[16, 16]", permute_2: "f16[16, 16]", permute_1: "f16[16, 16]"):
        # No stacktrace found for following nodes
        tanh_default: "f16[4, 16]" = torch.ops.aten.tanh.default(addmm_2);  addmm_2 = None
        convert_element_type_default: "f16[1]" = torch.ops.prims.convert_element_type.default(arg10_1, torch.float16);  arg10_1 = None
        convert_element_type_default_1: "f16[1, 16]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.float16);  arg9_1 = None
        permute_default: "f16[16, 1]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        permute_default_1: "f16[16, 16]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        permute_default_2: "f16[16, 16]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_default_3: "f16[16, 16]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        return (tanh_default, convert_element_type_default, permute_default, permute_default_1, permute_default_2, permute_default_3)


def _default_make_inputs():
    return [
    torch.randn([4, 16], dtype=torch.float16, device='cuda'),
    torch.randn([1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16], dtype=torch.float32, device='cuda'),
    torch.randn(256, dtype=torch.float16, device='cuda').as_strided([16, 16], [1, 16]),  # permute_3
    torch.randn(256, dtype=torch.float16, device='cuda').as_strided([16, 16], [1, 16]),  # permute_2
    torch.randn(256, dtype=torch.float16, device='cuda').as_strided([16, 16], [1, 16]),  # permute_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
