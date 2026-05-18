"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s4_g20
Pattern hash: 6ef723c3f65e
Shape hash: 7fc33421
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
    def forward(self, mm_33: "f16[4096, 512]", inductor_seeds: "i64[64]", add_30: "f32[4, 1024, 512]", arg48_1: "f32[512]", arg49_1: "f32[2048, 512]", unsqueeze_1: "i64[1, 1, 1024]", unsqueeze_2: "i64[1, 1, 1024, 1]", full: "f32[]", full_1: "f32[]", mm_38: "f16[4096, 512]", bmm_12: "f16[32, 1024, 1024]", sub: "i64[1024, 1024]", arg58_1: "f32[32, 8]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_33, [4, 1024, 512]);  mm_33 = None
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 22)
        inductor_random_default: "f32[4, 1024, 512]" = torch.ops.prims.inductor_random.default([4, 1024, 512], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        convert_element_type_default: "f16[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.float16);  inductor_random_default = None
        gt_scalar: "b8[4, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_default, 0.1);  convert_element_type_default = None
        mul_tensor: "f16[4, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_scalar, reshape_default);  gt_scalar = reshape_default = None
        mul_tensor_1: "f16[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        add_tensor: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_30, mul_tensor_1);  add_30 = mul_tensor_1 = None
        pow_tensor_scalar: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_tensor, 2)
        mean_dim: "f32[4, 1024, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None
        add_tensor_1: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_2: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor, rsqrt_default);  add_tensor = rsqrt_default = None
        mul_tensor_3: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg48_1, mul_tensor_2);  arg48_1 = mul_tensor_2 = None
        convert_element_type_default_1: "f16[2048, 512]" = torch.ops.prims.convert_element_type.default(arg49_1, torch.float16);  arg49_1 = None
        convert_element_type_default_2: "f16[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.float16);  mul_tensor_3 = None
        permute_default: "f16[512, 2048]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        reshape_default_1: "f16[4096, 512]" = torch.ops.aten.reshape.default(convert_element_type_default_2, [4096, 512]);  convert_element_type_default_2 = None
        unsqueeze_default: "i64[1, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None
        le_tensor: "b8[1, 1, 1024, 1024]" = torch.ops.aten.le.Tensor(unsqueeze_default, unsqueeze_2);  unsqueeze_default = unsqueeze_2 = None
        expand_default: "b8[4, 1, 1024, 1024]" = torch.ops.aten.expand.default(le_tensor, [4, -1, 1024, 1024]);  le_tensor = None
        where_self: "f32[4, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_default, full, full_1);  expand_default = full = full_1 = None
        reshape_default_2: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_38, [4, 1024, 512]);  mm_38 = None
        reshape_default_3: "f16[4, 1024, 8, 64]" = torch.ops.aten.reshape.default(reshape_default_2, [4, 1024, -1, 64]);  reshape_default_2 = None
        permute_default_1: "f16[4, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None
        reshape_default_4: "f16[4, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_12, [4, 8, 1024, 1024]);  bmm_12 = None
        full_default: "i64[1024, 1024]" = torch.ops.aten.full.default([1024, 1024], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_default: "i64[1024, 1024]" = torch.ops.aten.minimum.default(sub, full_default);  sub = full_default = None
        neg_default: "i64[1024, 1024]" = torch.ops.aten.neg.default(minimum_default);  minimum_default = None
        lt_scalar: "b8[1024, 1024]" = torch.ops.aten.lt.Scalar(neg_default, 16)
        convert_element_type_default_3: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(neg_default, torch.float32)
        div_tensor: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(convert_element_type_default_3, 16);  convert_element_type_default_3 = None
        log_default: "f32[1024, 1024]" = torch.ops.aten.log.default(div_tensor);  div_tensor = None
        div_tensor_1: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(log_default, 2.0794415416798357);  log_default = None
        mul_tensor_4: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(div_tensor_1, 16);  div_tensor_1 = None
        convert_element_type_default_4: "i64[1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.int64);  mul_tensor_4 = None
        add_tensor_2: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_default_4, 16);  convert_element_type_default_4 = None
        full_default_1: "i64[1024, 1024]" = torch.ops.aten.full.default([1024, 1024], 31, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_default_1: "i64[1024, 1024]" = torch.ops.aten.minimum.default(add_tensor_2, full_default_1);  add_tensor_2 = full_default_1 = None
        where_self_1: "i64[1024, 1024]" = torch.ops.aten.where.self(lt_scalar, neg_default, minimum_default_1);  lt_scalar = neg_default = minimum_default_1 = None
        add_tensor_3: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(where_self_1, 0);  where_self_1 = None
        embedding_default: "f32[1024, 1024, 8]" = torch.ops.aten.embedding.default(arg58_1, add_tensor_3);  arg58_1 = add_tensor_3 = None
        permute_default_2: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(embedding_default, [2, 0, 1]);  embedding_default = None
        unsqueeze_default_1: "f32[1, 8, 1024, 1024]" = torch.ops.aten.unsqueeze.default(permute_default_2, 0);  permute_default_2 = None
        add_tensor_4: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_default_1, where_self);  unsqueeze_default_1 = where_self = None
        add_tensor_5: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(reshape_default_4, add_tensor_4);  reshape_default_4 = add_tensor_4 = None
        convert_element_type_default_5: "f32[4, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor_5, torch.float32);  add_tensor_5 = None
        amax_default: "f32[4, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_default_5, [-1], True)
        sub_tensor: "f32[4, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_default_5, amax_default);  convert_element_type_default_5 = amax_default = None
        exp_default: "f32[4, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor_2: "f32[4, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        convert_element_type_default_6: "f16[4, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_tensor_2, torch.float16);  div_tensor_2 = None
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 27);  inductor_seeds = None
        inductor_random_default_1: "f32[4, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default([4, 8, 1024, 1024], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        convert_element_type_default_7: "f16[4, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_default_1, torch.float16);  inductor_random_default_1 = None
        gt_scalar_1: "b8[4, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_default_7, 0.1);  convert_element_type_default_7 = None
        mul_tensor_5: "f16[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_scalar_1, convert_element_type_default_6);  gt_scalar_1 = convert_element_type_default_6 = None
        mul_tensor_6: "f16[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_5, 1.1111111111111112);  mul_tensor_5 = None
        expand_default_1: "f16[4, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_tensor_6, [4, 8, 1024, 1024]);  mul_tensor_6 = None
        reshape_default_5: "f16[32, 1024, 1024]" = torch.ops.aten.reshape.default(expand_default_1, [32, 1024, 1024]);  expand_default_1 = None
        expand_default_2: "f16[4, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_default_1, [4, 8, 1024, 64]);  permute_default_1 = None
        clone_default: "f16[4, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_default_2, memory_format = torch.contiguous_format);  expand_default_2 = None
        reshape_default_6: "f16[32, 1024, 64]" = torch.ops.aten.reshape.default(clone_default, [32, 1024, 64]);  clone_default = None
        return (permute_default, reshape_default_1, reshape_default_5, reshape_default_6)


def _default_make_inputs():
    return [
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randint(0, 2, [64], dtype=torch.int64, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [1, 1, 1024], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, [1, 1, 1024, 1], dtype=torch.int64, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([32, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randint(0, 2, [1024, 1024], dtype=torch.int64, device='cuda'),
    torch.randn([32, 8], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
