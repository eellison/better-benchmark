"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s4_g20
Pattern hash: 1b6e8b05ef3b
Shape hash: a9eaae6d
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_94: "f16[4096, 2048]", inductor_seeds: "i64[64]", arg132_1: "f32[512, 2048]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[4, 1024, 2048]" = torch.ops.aten.reshape.default(mm_94, [4, 1024, 2048]);  mm_94 = None
        relu_default: "f16[4, 1024, 2048]" = torch.ops.aten.relu.default(reshape_default);  reshape_default = None
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 61);  inductor_seeds = None
        inductor_random_default: "f32[4, 1024, 2048]" = torch.ops.prims.inductor_random.default([4, 1024, 2048], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        convert_element_type_default: "f16[4, 1024, 2048]" = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.float16);  inductor_random_default = None
        gt_scalar: "b8[4, 1024, 2048]" = torch.ops.aten.gt.Scalar(convert_element_type_default, 0.1);  convert_element_type_default = None
        mul_tensor: "f16[4, 1024, 2048]" = torch.ops.aten.mul.Tensor(gt_scalar, relu_default);  gt_scalar = relu_default = None
        mul_tensor_1: "f16[4, 1024, 2048]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        convert_element_type_default_1: "f16[512, 2048]" = torch.ops.prims.convert_element_type.default(arg132_1, torch.float16);  arg132_1 = None
        convert_element_type_default_2: "f16[4, 1024, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.float16);  mul_tensor_1 = None
        permute_default: "f16[2048, 512]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        reshape_default_1: "f16[4096, 2048]" = torch.ops.aten.reshape.default(convert_element_type_default_2, [4096, 2048]);  convert_element_type_default_2 = None
        return (permute_default, reshape_default_1)


def _default_make_inputs():
    return [
    torch.randn([4096, 2048], dtype=torch.float16, device='cuda'),
    torch.randint(0, 2, [64], dtype=torch.int64, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
