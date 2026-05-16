"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s4_g20
Pattern hash: 0ceab191d8e4
Shape hash: 9f153739
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
    def forward(self, mm_93: "f16[4096, 512]", inductor_seeds: "i64[64]", add_83: "f32[4, 1024, 512]", arg130_1: "f32[512]", arg131_1: "f32[2048, 512]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_93, [4, 1024, 512]);  mm_93 = None
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 60);  inductor_seeds = None
        inductor_random_default: "f32[4, 1024, 512]" = torch.ops.prims.inductor_random.default([4, 1024, 512], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        convert_element_type_default: "f16[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.float16);  inductor_random_default = None
        gt_scalar: "b8[4, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_default, 0.1);  convert_element_type_default = None
        mul_tensor: "f16[4, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_scalar, reshape_default);  gt_scalar = reshape_default = None
        mul_tensor_1: "f16[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        add_tensor: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_83, mul_tensor_1);  add_83 = mul_tensor_1 = None
        pow_tensor_scalar: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_tensor, 2)
        mean_dim: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None
        add_tensor_1: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_2: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor, rsqrt_default);  add_tensor = rsqrt_default = None
        mul_tensor_3: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg130_1, mul_tensor_2);  arg130_1 = mul_tensor_2 = None
        convert_element_type_default_1: "f16[2048, 512]" = torch.ops.prims.convert_element_type.default(arg131_1, torch.float16);  arg131_1 = None
        convert_element_type_default_2: "f16[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.float16);  mul_tensor_3 = None
        permute_default: "f16[512, 2048]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        reshape_default_1: "f16[4096, 512]" = torch.ops.aten.reshape.default(convert_element_type_default_2, [4096, 512]);  convert_element_type_default_2 = None
        return (permute_default, reshape_default_1)


def _default_make_inputs():
    return [
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randint(0, 2, [64], dtype=torch.int64, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
