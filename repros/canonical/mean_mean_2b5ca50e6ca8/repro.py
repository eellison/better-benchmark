"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s4_g20
Pattern hash: 2b5ca50e6ca8
Shape hash: 513383a4
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
    def forward(self, mm_35: "f16[4096, 512]", inductor_seeds: "i64[64]", add_33: "f32[4, 1024, 512]", arg51_1: "f32[512]", mm_39: "f16[4096, 512]", mul_81: "f32[4, 1024, 512]", arg60_1: "f32[512]", arg61_1: "f32[512, 512]", arg62_1: "f32[512, 512]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_35, [4, 1024, 512]);  mm_35 = None
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 24)
        inductor_random_default: "f32[4, 1024, 512]" = torch.ops.prims.inductor_random.default([4, 1024, 512], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        convert_element_type_default: "f16[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.float16);  inductor_random_default = None
        gt_scalar: "b8[4, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_default, 0.1);  convert_element_type_default = None
        mul_tensor: "f16[4, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_scalar, reshape_default);  gt_scalar = reshape_default = None
        mul_tensor_1: "f16[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        add_tensor: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_33, mul_tensor_1);  add_33 = mul_tensor_1 = None
        pow_tensor_scalar: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_tensor, 2)
        mean_dim: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None
        add_tensor_1: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_2: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor, rsqrt_default);  add_tensor = rsqrt_default = None
        mul_tensor_3: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg51_1, mul_tensor_2);  arg51_1 = mul_tensor_2 = None
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 25)
        inductor_random_default_1: "f32[4, 1024, 512]" = torch.ops.prims.inductor_random.default([4, 1024, 512], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_scalar_1: "b8[4, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.1);  inductor_random_default_1 = None
        mul_tensor_4: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_scalar_1, mul_tensor_3);  gt_scalar_1 = mul_tensor_3 = None
        mul_tensor_5: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_4, 1.1111111111111112);  mul_tensor_4 = None
        reshape_default_1: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_39, [4, 1024, 512]);  mm_39 = None
        inductor_lookup_seed_default_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 28);  inductor_seeds = None
        inductor_random_default_2: "f32[4, 1024, 512]" = torch.ops.prims.inductor_random.default([4, 1024, 512], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        convert_element_type_default_1: "f16[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_default_2, torch.float16);  inductor_random_default_2 = None
        gt_scalar_2: "b8[4, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_default_1, 0.1);  convert_element_type_default_1 = None
        mul_tensor_6: "f16[4, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_scalar_2, reshape_default_1);  gt_scalar_2 = reshape_default_1 = None
        mul_tensor_7: "f16[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_6, 1.1111111111111112);  mul_tensor_6 = None
        add_tensor_2: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(mul_81, mul_tensor_7);  mul_81 = mul_tensor_7 = None
        pow_tensor_scalar_1: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_tensor_2, 2)
        mean_dim_1: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar_1, [-1], True);  pow_tensor_scalar_1 = None
        add_tensor_3: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_dim_1, 1e-06);  mean_dim_1 = None
        rsqrt_default_1: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor_3);  add_tensor_3 = None
        mul_tensor_8: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor_2, rsqrt_default_1);  add_tensor_2 = rsqrt_default_1 = None
        mul_tensor_9: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg60_1, mul_tensor_8);  arg60_1 = mul_tensor_8 = None
        convert_element_type_default_2: "f16[512, 512]" = torch.ops.prims.convert_element_type.default(arg61_1, torch.float16);  arg61_1 = None
        convert_element_type_default_3: "f16[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_9, torch.float16);  mul_tensor_9 = None
        permute_default: "f16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_default_2, [1, 0]);  convert_element_type_default_2 = None
        reshape_default_2: "f16[4096, 512]" = torch.ops.aten.reshape.default(convert_element_type_default_3, [4096, 512]);  convert_element_type_default_3 = None
        convert_element_type_default_4: "f16[512, 512]" = torch.ops.prims.convert_element_type.default(arg62_1, torch.float16);  arg62_1 = None
        convert_element_type_default_5: "f16[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_5, torch.float16);  mul_tensor_5 = None
        permute_default_1: "f16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_default_4, [1, 0]);  convert_element_type_default_4 = None
        reshape_default_3: "f16[4096, 512]" = torch.ops.aten.reshape.default(convert_element_type_default_5, [4096, 512]);  convert_element_type_default_5 = None
        return (permute_default, reshape_default_2, permute_default_1, reshape_default_3)


def _default_make_inputs():
    return [
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randint(0, 2, [64], dtype=torch.int64, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
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
