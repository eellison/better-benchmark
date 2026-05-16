"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s4_g20
Pattern hash: b9736a91f42a
Shape hash: 9bf86746
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
    def forward(self, arg1_1: "f32[32128, 512]", arg0_1: "i64[4, 1024]", arg2_1: "f32[512]", arg3_1: "f32[512, 512]", arg4_1: "f32[512, 512]"):
        # No stacktrace found for following nodes
        embedding_default: "f32[4, 1024, 512]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        inductor_seeds_default: "i64[64]" = torch.ops.prims.inductor_seeds.default(64, device(type='cuda', index=0))
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[4, 1024, 512]" = torch.ops.prims.inductor_random.default([4, 1024, 512], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[4, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_scalar, embedding_default);  gt_scalar = embedding_default = None
        mul_tensor_1: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        pow_tensor_scalar: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(mul_tensor_1, 2)
        mean_dim: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None
        add_tensor: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor_2: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, rsqrt_default);  mul_tensor_1 = rsqrt_default = None
        mul_tensor_3: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg2_1, mul_tensor_2);  arg2_1 = mul_tensor_2 = None
        convert_element_type_default: "f16[512, 512]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float16);  arg3_1 = None
        convert_element_type_default_1: "f16[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.float16);  mul_tensor_3 = None
        permute_default: "f16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_default, [1, 0]);  convert_element_type_default = None
        reshape_default: "f16[4096, 512]" = torch.ops.aten.reshape.default(convert_element_type_default_1, [4096, 512]);  convert_element_type_default_1 = None
        convert_element_type_default_2: "f16[512, 512]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float16);  arg4_1 = None
        permute_default_1: "f16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_default_2, [1, 0]);  convert_element_type_default_2 = None
        return (permute_default, reshape_default, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([32128, 512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [4, 1024], dtype=torch.int64, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
