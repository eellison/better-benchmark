"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s4_g20
Pattern hash: 590318f4803d
Shape hash: c6fd6e62
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_32: "f16[4096, 512]", bmm_10: "f16[32, 1024, 1024]", add_6: "f32[4, 8, 1024, 1024]", inductor_seeds: "i64[64]", arg1_1: "f32[32128, 512]", arg53_1: "i64[4, 1024]", arg54_1: "f32[512]", arg55_1: "f32[512, 512]", arg56_1: "f32[512, 512]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_32, [4, 1024, 512]);  mm_32 = None
        reshape_default_1: "f16[4, 1024, 8, 64]" = torch.ops.aten.reshape.default(reshape_default, [4, 1024, -1, 64]);  reshape_default = None
        permute_default: "f16[4, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None
        reshape_default_2: "f16[4, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_10, [4, 8, 1024, 1024]);  bmm_10 = None
        add_tensor: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(reshape_default_2, add_6);  reshape_default_2 = add_6 = None
        convert_element_type_default: "f32[4, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        amax_default: "f32[4, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_default, [-1], True)
        sub_tensor: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_default, amax_default);  convert_element_type_default = amax_default = None
        exp_default: "f32[4, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[4, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        convert_element_type_default_1: "f16[4, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.float16);  div_tensor = None
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 21)
        inductor_random_default: "f32[4, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default([4, 8, 1024, 1024], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        convert_element_type_default_2: "f16[4, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.float16);  inductor_random_default = None
        gt_scalar: "b8[4, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_default_2, 0.1);  convert_element_type_default_2 = None
        mul_tensor: "f16[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_scalar, convert_element_type_default_1);  gt_scalar = convert_element_type_default_1 = None
        mul_tensor_1: "f16[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        expand_default: "f16[4, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_tensor_1, [4, 8, 1024, 1024]);  mul_tensor_1 = None
        reshape_default_3: "f16[32, 1024, 1024]" = torch.ops.aten.reshape.default(expand_default, [32, 1024, 1024]);  expand_default = None
        expand_default_1: "f16[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_default, [4, 8, 1024, 64]);  permute_default = None
        clone_default: "f16[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_4: "f16[32, 1024, 64]" = torch.ops.aten.reshape.default(clone_default, [32, 1024, 64]);  clone_default = None
        embedding_default: "f32[4, 1024, 512]" = torch.ops.aten.embedding.default(arg1_1, arg53_1);  arg1_1 = arg53_1 = None
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 26);  inductor_seeds = None
        inductor_random_default_1: "f32[4, 1024, 512]" = torch.ops.prims.inductor_random.default([4, 1024, 512], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_scalar_1: "b8[4, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.1);  inductor_random_default_1 = None
        mul_tensor_2: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_scalar_1, embedding_default);  gt_scalar_1 = embedding_default = None
        mul_tensor_3: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.1111111111111112);  mul_tensor_2 = None
        pow_tensor_scalar: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(mul_tensor_3, 2)
        mean_dim: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None
        add_tensor_1: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_4: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_3, rsqrt_default);  mul_tensor_3 = rsqrt_default = None
        mul_tensor_5: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg54_1, mul_tensor_4);  arg54_1 = mul_tensor_4 = None
        convert_element_type_default_3: "f16[512, 512]" = torch.ops.prims.convert_element_type.default(arg55_1, torch.float16);  arg55_1 = None
        convert_element_type_default_4: "f16[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_5, torch.float16);  mul_tensor_5 = None
        permute_default_1: "f16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_default_3, [1, 0]);  convert_element_type_default_3 = None
        reshape_default_5: "f16[4096, 512]" = torch.ops.aten.reshape.default(convert_element_type_default_4, [4096, 512]);  convert_element_type_default_4 = None
        convert_element_type_default_5: "f16[512, 512]" = torch.ops.prims.convert_element_type.default(arg56_1, torch.float16);  arg56_1 = None
        permute_default_2: "f16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_default_5, [1, 0]);  convert_element_type_default_5 = None
        return (reshape_default_3, reshape_default_4, permute_default_1, reshape_default_5, permute_default_2)


def _default_make_inputs():
    return [
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randn(33554432, dtype=torch.float32, device='cuda').as_strided([4, 8, 1024, 1024], [8388608, 1, 8192, 8]),  # add_6
    torch.randint(0, 2, [64], dtype=torch.int64, device='cuda'),
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
