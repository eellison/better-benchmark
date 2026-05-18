"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s9_g20
Pattern hash: ed0145e01b1f
Shape hash: 2b0bb86f
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
    def forward(self, addmm_34: "f16[1600, 3072]", arg149_1: "f32[768]", arg148_1: "f32[768, 3072]", addmm_82: "f16[2464, 2048]", arg300_1: "f32[512]", arg299_1: "f32[512, 2048]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[32, 50, 3072]" = torch.ops.aten.reshape.default(addmm_34, [32, 50, 3072]);  addmm_34 = None
        mul_tensor: "f16[32, 50, 3072]" = torch.ops.aten.mul.Tensor(reshape_default, 1.702)
        sigmoid_default: "f16[32, 50, 3072]" = torch.ops.aten.sigmoid.default(mul_tensor);  mul_tensor = None
        mul_tensor_1: "f16[32, 50, 3072]" = torch.ops.aten.mul.Tensor(sigmoid_default, reshape_default);  sigmoid_default = reshape_default = None
        convert_element_type_default: "f16[768]" = torch.ops.prims.convert_element_type.default(arg149_1, torch.float16);  arg149_1 = None
        convert_element_type_default_1: "f16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg148_1, torch.float16);  arg148_1 = None
        reshape_default_1: "f16[1600, 3072]" = torch.ops.aten.reshape.default(mul_tensor_1, [1600, 3072]);  mul_tensor_1 = None
        permute_default: "f16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        reshape_default_2: "f16[77, 32, 2048]" = torch.ops.aten.reshape.default(addmm_82, [77, 32, 2048]);  addmm_82 = None
        mul_tensor_2: "f16[77, 32, 2048]" = torch.ops.aten.mul.Tensor(reshape_default_2, 1.702)
        sigmoid_default_1: "f16[77, 32, 2048]" = torch.ops.aten.sigmoid.default(mul_tensor_2);  mul_tensor_2 = None
        mul_tensor_3: "f16[77, 32, 2048]" = torch.ops.aten.mul.Tensor(sigmoid_default_1, reshape_default_2);  sigmoid_default_1 = reshape_default_2 = None
        convert_element_type_default_2: "f16[512]" = torch.ops.prims.convert_element_type.default(arg300_1, torch.float16);  arg300_1 = None
        convert_element_type_default_3: "f16[512, 2048]" = torch.ops.prims.convert_element_type.default(arg299_1, torch.float16);  arg299_1 = None
        reshape_default_3: "f16[2464, 2048]" = torch.ops.aten.reshape.default(mul_tensor_3, [2464, 2048]);  mul_tensor_3 = None
        permute_default_1: "f16[2048, 512]" = torch.ops.aten.permute.default(convert_element_type_default_3, [1, 0]);  convert_element_type_default_3 = None
        return (convert_element_type_default, reshape_default_1, permute_default, convert_element_type_default_2, reshape_default_3, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([1600, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([2464, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
