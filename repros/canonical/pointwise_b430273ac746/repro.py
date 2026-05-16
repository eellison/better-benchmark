"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s5_g11
Pattern hash: b430273ac746
Shape hash: 15f4cadf
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_109: "bf16[1000, 8192]", mm_110: "bf16[1000, 8192]", arg146_1: "bf16[2048, 8192]"):
        # No stacktrace found for following nodes
        reshape_default: "bf16[1, 1000, 8192]" = torch.ops.aten.reshape.default(mm_109, [1, 1000, 8192]);  mm_109 = None
        convert_element_type_default: "f32[1, 1000, 8192]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        neg_default: "f32[1, 1000, 8192]" = torch.ops.aten.neg.default(convert_element_type_default)
        exp_default: "f32[1, 1000, 8192]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[1, 1000, 8192]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[1, 1000, 8192]" = torch.ops.aten.div.Tensor(convert_element_type_default, add_tensor);  convert_element_type_default = add_tensor = None
        convert_element_type_default_1: "bf16[1, 1000, 8192]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        reshape_default_1: "bf16[1, 1000, 8192]" = torch.ops.aten.reshape.default(mm_110, [1, 1000, 8192]);  mm_110 = None
        mul_tensor: "bf16[1, 1000, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, reshape_default_1);  convert_element_type_default_1 = reshape_default_1 = None
        permute_default: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        reshape_default_2: "bf16[1000, 8192]" = torch.ops.aten.reshape.default(mul_tensor, [1000, 8192]);  mul_tensor = None
        return (permute_default, reshape_default_2)


def _default_make_inputs():
    return [
    torch.randn([1000, 8192], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1000, 8192], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 8192], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
