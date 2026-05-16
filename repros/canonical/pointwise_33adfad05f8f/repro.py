"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s4_g35
Pattern hash: 33adfad05f8f
Shape hash: 4d1fa5d2
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_171: "f16[4096, 2048]", arg64_1: "b8[4, 1024, 2048]", arg417_1: "b8[4, 1024, 2048]", full_2: "f16[]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[4, 1024, 2048]" = torch.ops.aten.reshape.default(mm_171, [4, 1024, 2048]);  mm_171 = None
        convert_element_type_default: "f16[4, 1024, 2048]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float16);  reshape_default = None
        convert_element_type_default_1: "f16[4, 1024, 2048]" = torch.ops.prims.convert_element_type.default(arg64_1, torch.float16);  arg64_1 = None
        mul_tensor: "f16[4, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_1: "f16[4, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor);  convert_element_type_default = mul_tensor = None
        where_self: "f16[4, 1024, 2048]" = torch.ops.aten.where.self(arg417_1, full_2, mul_tensor_1);  arg417_1 = full_2 = mul_tensor_1 = None
        reshape_default_1: "f16[4096, 2048]" = torch.ops.aten.reshape.default(where_self, [4096, 2048]);  where_self = None
        return reshape_default_1


def _default_make_inputs():
    return [
    torch.randn([4096, 2048], dtype=torch.float16, device='cuda'),
    torch.randint(0, 2, [4, 1024, 2048], dtype=torch.bool, device='cuda'),
    torch.randint(0, 2, [4, 1024, 2048], dtype=torch.bool, device='cuda'),
    torch.randn([], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
