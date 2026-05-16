"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s3_g62
Pattern hash: 2c617e11db8e
Shape hash: 157e759f
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_70: "f16[1904, 3072]", arg197_1: "f32[768]", arg196_1: "f32[768, 3072]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[4, 476, 3072]" = torch.ops.aten.reshape.default(addmm_70, [4, 476, 3072]);  addmm_70 = None
        mul_tensor: "f16[4, 476, 3072]" = torch.ops.aten.mul.Tensor(reshape_default, 0.5)
        div_tensor: "f16[4, 476, 3072]" = torch.ops.aten.div.Tensor(reshape_default, 1.4142135623730951);  reshape_default = None
        erf_default: "f16[4, 476, 3072]" = torch.ops.aten.erf.default(div_tensor);  div_tensor = None
        add_tensor: "f16[4, 476, 3072]" = torch.ops.aten.add.Tensor(erf_default, 1.0);  erf_default = None
        mul_tensor_1: "f16[4, 476, 3072]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None
        convert_element_type_default: "f16[768]" = torch.ops.prims.convert_element_type.default(arg197_1, torch.float16);  arg197_1 = None
        convert_element_type_default_1: "f16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg196_1, torch.float16);  arg196_1 = None
        reshape_default_1: "f16[1904, 3072]" = torch.ops.aten.reshape.default(mul_tensor_1, [1904, 3072]);  mul_tensor_1 = None
        permute_default: "f16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        return (convert_element_type_default, reshape_default_1, permute_default)


def _default_make_inputs():
    return [
    torch.randn([1904, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
