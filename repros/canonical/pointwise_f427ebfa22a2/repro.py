"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s3_g21
Pattern hash: f427ebfa22a2
Shape hash: 68f1a4dd
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
    def forward(self, addmm_70: "f16[512, 3072]", arg200_1: "f32[768]", arg199_1: "f32[768, 3072]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_70, [1, 512, 3072]);  addmm_70 = None
        convert_element_type_default: "f32[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        mul_tensor: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.5)
        mul_tensor_1: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.7071067811865476);  convert_element_type_default = None
        erf_default: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None
        convert_element_type_default_1: "f16[1, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.float16);  mul_tensor_2 = None
        convert_element_type_default_2: "f16[768]" = torch.ops.prims.convert_element_type.default(arg200_1, torch.float16);  arg200_1 = None
        convert_element_type_default_3: "f16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg199_1, torch.float16);  arg199_1 = None
        reshape_default_1: "f16[512, 3072]" = torch.ops.aten.reshape.default(convert_element_type_default_1, [512, 3072]);  convert_element_type_default_1 = None
        permute_default: "f16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_default_3, [1, 0]);  convert_element_type_default_3 = None
        return (convert_element_type_default_2, reshape_default_1, permute_default)


def _default_make_inputs():
    return [
    torch.randn([512, 3072], dtype=torch.float16, device='cuda'),
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
