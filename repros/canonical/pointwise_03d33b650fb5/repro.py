"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s4_g20
Pattern hash: 03d33b650fb5
Shape hash: ff7d42f0
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
    def forward(self, mm_34: "f16[4096, 2048]", inductor_seeds: "i64[64]", arg50_1: "f32[512, 2048]", bmm_13: "f16[32, 1024, 64]", arg59_1: "f32[512, 512]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[4, 1024, 2048]" = torch.ops.aten.reshape.default(mm_34, [4, 1024, 2048]);  mm_34 = None
        relu_default: "f16[4, 1024, 2048]" = torch.ops.aten.relu.default(reshape_default);  reshape_default = None
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 23);  inductor_seeds = None
        inductor_random_default: "f32[4, 1024, 2048]" = torch.ops.prims.inductor_random.default([4, 1024, 2048], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        convert_element_type_default: "f16[4, 1024, 2048]" = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.float16);  inductor_random_default = None
        gt_scalar: "b8[4, 1024, 2048]" = torch.ops.aten.gt.Scalar(convert_element_type_default, 0.1);  convert_element_type_default = None
        mul_tensor: "f16[4, 1024, 2048]" = torch.ops.aten.mul.Tensor(gt_scalar, relu_default);  gt_scalar = relu_default = None
        mul_tensor_1: "f16[4, 1024, 2048]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        convert_element_type_default_1: "f16[512, 2048]" = torch.ops.prims.convert_element_type.default(arg50_1, torch.float16);  arg50_1 = None
        convert_element_type_default_2: "f16[4, 1024, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.float16);  mul_tensor_1 = None
        permute_default: "f16[2048, 512]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        reshape_default_1: "f16[4096, 2048]" = torch.ops.aten.reshape.default(convert_element_type_default_2, [4096, 2048]);  convert_element_type_default_2 = None
        reshape_default_2: "f16[4, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_13, [4, 8, 1024, 64]);  bmm_13 = None
        permute_default_1: "f16[4, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1, 3]);  reshape_default_2 = None
        clone_default: "f16[4, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_3: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(clone_default, [4, 1024, -1]);  clone_default = None
        convert_element_type_default_3: "f16[512, 512]" = torch.ops.prims.convert_element_type.default(arg59_1, torch.float16);  arg59_1 = None
        permute_default_2: "f16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_default_3, [1, 0]);  convert_element_type_default_3 = None
        reshape_default_4: "f16[4096, 512]" = torch.ops.aten.reshape.default(reshape_default_3, [4096, 512]);  reshape_default_3 = None
        return (permute_default, reshape_default_1, permute_default_2, reshape_default_4)


def _default_make_inputs():
    return [
    torch.randn([4096, 2048], dtype=torch.float16, device='cuda'),
    torch.randint(0, 2, [64], dtype=torch.int64, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1024, 64], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
