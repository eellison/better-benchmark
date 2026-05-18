"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s5_g11
Pattern hash: 81199804097e
Shape hash: 557c1248
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
    def forward(self, arg2_1: "bf16[32]", mm: "bf16[1000, 2048]", mm_1: "bf16[1000, 512]", mm_2: "bf16[1000, 512]"):
        # No stacktrace found for following nodes
        iota_default: "i64[1000]" = torch.ops.prims.iota.default(1000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[1000]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None
        unsqueeze_default: "i64[1, 1000]" = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None
        iota_default_1: "i64[1000]" = torch.ops.prims.iota.default(1000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor_1: "i64[1000]" = torch.ops.aten.add.Tensor(iota_default_1, 0);  iota_default_1 = None
        iota_default_2: "i64[1000]" = torch.ops.prims.iota.default(1000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor_2: "i64[1000]" = torch.ops.aten.add.Tensor(iota_default_2, 0);  iota_default_2 = None
        unsqueeze_default_1: "i64[1, 1000]" = torch.ops.aten.unsqueeze.default(add_tensor_1, 0);  add_tensor_1 = None
        unsqueeze_default_2: "i64[1, 1, 1000]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 1);  unsqueeze_default_1 = None
        unsqueeze_default_3: "i64[1, 1, 1000, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 3);  unsqueeze_default_2 = None
        unsqueeze_default_4: "i64[1, 1000]" = torch.ops.aten.unsqueeze.default(add_tensor_2, 0);  add_tensor_2 = None
        unsqueeze_default_5: "i64[1, 1, 1000]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 1);  unsqueeze_default_4 = None
        unsqueeze_default_6: "i64[1, 1, 1, 1000]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 2);  unsqueeze_default_5 = None
        le_tensor: "b8[1, 1, 1000, 1000]" = torch.ops.aten.le.Tensor(unsqueeze_default_6, unsqueeze_default_3);  unsqueeze_default_6 = unsqueeze_default_3 = None
        expand_default: "b8[1, 1, 1000, 1000]" = torch.ops.aten.expand.default(le_tensor, [1, -1, 1000, 1000]);  le_tensor = None
        unsqueeze_default_7: "bf16[1, 32]" = torch.ops.aten.unsqueeze.default(arg2_1, 0);  arg2_1 = None
        unsqueeze_default_8: "bf16[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        convert_element_type_default: "f32[1, 32, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_default_8, torch.float32);  unsqueeze_default_8 = None
        expand_default_1: "f32[1, 32, 1]" = torch.ops.aten.expand.default(convert_element_type_default, [1, -1, 1]);  convert_element_type_default = None
        unsqueeze_default_9: "i64[1, 1, 1000]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 1);  unsqueeze_default = None
        convert_element_type_default_1: "f32[1, 1, 1000]" = torch.ops.prims.convert_element_type.default(unsqueeze_default_9, torch.float32);  unsqueeze_default_9 = None
        expand_default_2: "f32[1, 32, 1]" = torch.ops.aten.expand.default(expand_default_1, [1, 32, 1]);  expand_default_1 = None
        expand_default_3: "f32[1, 1, 1000]" = torch.ops.aten.expand.default(convert_element_type_default_1, [1, 1, 1000]);  convert_element_type_default_1 = None
        mul_tensor: "f32[1, 32, 1000]" = torch.ops.aten.mul.Tensor(expand_default_2, expand_default_3);  expand_default_2 = expand_default_3 = None
        permute_default: "f32[1, 1000, 32]" = torch.ops.aten.permute.default(mul_tensor, [0, 2, 1]);  mul_tensor = None
        unsqueeze_default_10: "f32[1, 1000, 1, 32]" = torch.ops.aten.unsqueeze.default(permute_default, 2);  permute_default = None
        expand_default_4: "f32[1, 1000, 2, 32]" = torch.ops.aten.expand.default(unsqueeze_default_10, [1, 1000, 2, 32]);  unsqueeze_default_10 = None
        clone_default: "f32[1, 1000, 2, 32]" = torch.ops.aten.clone.default(expand_default_4, memory_format = torch.contiguous_format);  expand_default_4 = None
        reshape_default: "f32[1, 1000, 64]" = torch.ops.aten.reshape.default(clone_default, [1, 1000, 64]);  clone_default = None
        cos_default: "f32[1, 1000, 64]" = torch.ops.aten.cos.default(reshape_default)
        mul_tensor_1: "f32[1, 1000, 64]" = torch.ops.aten.mul.Tensor(cos_default, 1.0);  cos_default = None
        sin_default: "f32[1, 1000, 64]" = torch.ops.aten.sin.default(reshape_default);  reshape_default = None
        mul_tensor_2: "f32[1, 1000, 64]" = torch.ops.aten.mul.Tensor(sin_default, 1.0);  sin_default = None
        convert_element_type_default_2: "bf16[1, 1000, 64]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None
        convert_element_type_default_3: "bf16[1, 1000, 64]" = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.bfloat16);  mul_tensor_2 = None
        reshape_default_1: "bf16[1, 1000, 2048]" = torch.ops.aten.reshape.default(mm, [1, 1000, 2048]);  mm = None
        reshape_default_2: "bf16[1, 1000, 32, 64]" = torch.ops.aten.reshape.default(reshape_default_1, [1, 1000, -1, 64]);  reshape_default_1 = None
        permute_default_1: "bf16[1, 32, 1000, 64]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1, 3]);  reshape_default_2 = None
        reshape_default_3: "bf16[1, 1000, 512]" = torch.ops.aten.reshape.default(mm_1, [1, 1000, 512]);  mm_1 = None
        reshape_default_4: "bf16[1, 1000, 8, 64]" = torch.ops.aten.reshape.default(reshape_default_3, [1, 1000, -1, 64]);  reshape_default_3 = None
        permute_default_2: "bf16[1, 8, 1000, 64]" = torch.ops.aten.permute.default(reshape_default_4, [0, 2, 1, 3]);  reshape_default_4 = None
        reshape_default_5: "bf16[1, 1000, 512]" = torch.ops.aten.reshape.default(mm_2, [1, 1000, 512]);  mm_2 = None
        reshape_default_6: "bf16[1, 1000, 8, 64]" = torch.ops.aten.reshape.default(reshape_default_5, [1, 1000, -1, 64]);  reshape_default_5 = None
        permute_default_3: "bf16[1, 8, 1000, 64]" = torch.ops.aten.permute.default(reshape_default_6, [0, 2, 1, 3]);  reshape_default_6 = None
        unsqueeze_default_11: "bf16[1, 1, 1000, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 1);  convert_element_type_default_2 = None
        unsqueeze_default_12: "bf16[1, 1, 1000, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_3, 1);  convert_element_type_default_3 = None
        mul_tensor_3: "bf16[1, 32, 1000, 64]" = torch.ops.aten.mul.Tensor(permute_default_1, unsqueeze_default_11)
        slice_tensor: "bf16[1, 32, 1000, 32]" = torch.ops.aten.slice.Tensor(permute_default_1, 3, 0, 32)
        slice_tensor_1: "bf16[1, 32, 1000, 32]" = torch.ops.aten.slice.Tensor(permute_default_1, 3, 32, 9223372036854775807);  permute_default_1 = None
        neg_default: "bf16[1, 32, 1000, 32]" = torch.ops.aten.neg.default(slice_tensor_1);  slice_tensor_1 = None
        cat_default: "bf16[1, 32, 1000, 64]" = torch.ops.aten.cat.default([neg_default, slice_tensor], -1);  neg_default = slice_tensor = None
        mul_tensor_4: "bf16[1, 32, 1000, 64]" = torch.ops.aten.mul.Tensor(cat_default, unsqueeze_default_12);  cat_default = None
        add_tensor_3: "bf16[1, 32, 1000, 64]" = torch.ops.aten.add.Tensor(mul_tensor_3, mul_tensor_4);  mul_tensor_3 = mul_tensor_4 = None
        mul_tensor_5: "bf16[1, 8, 1000, 64]" = torch.ops.aten.mul.Tensor(permute_default_2, unsqueeze_default_11);  unsqueeze_default_11 = None
        slice_tensor_2: "bf16[1, 8, 1000, 32]" = torch.ops.aten.slice.Tensor(permute_default_2, 3, 0, 32)
        slice_tensor_3: "bf16[1, 8, 1000, 32]" = torch.ops.aten.slice.Tensor(permute_default_2, 3, 32, 9223372036854775807);  permute_default_2 = None
        neg_default_1: "bf16[1, 8, 1000, 32]" = torch.ops.aten.neg.default(slice_tensor_3);  slice_tensor_3 = None
        cat_default_1: "bf16[1, 8, 1000, 64]" = torch.ops.aten.cat.default([neg_default_1, slice_tensor_2], -1);  neg_default_1 = slice_tensor_2 = None
        mul_tensor_6: "bf16[1, 8, 1000, 64]" = torch.ops.aten.mul.Tensor(cat_default_1, unsqueeze_default_12);  cat_default_1 = unsqueeze_default_12 = None
        add_tensor_4: "bf16[1, 8, 1000, 64]" = torch.ops.aten.add.Tensor(mul_tensor_5, mul_tensor_6);  mul_tensor_5 = mul_tensor_6 = None
        unsqueeze_default_13: "bf16[1, 8, 1, 1000, 64]" = torch.ops.aten.unsqueeze.default(add_tensor_4, 2);  add_tensor_4 = None
        expand_default_5: "bf16[1, 8, 4, 1000, 64]" = torch.ops.aten.expand.default(unsqueeze_default_13, [1, 8, 4, 1000, 64]);  unsqueeze_default_13 = None
        clone_default_1: "bf16[1, 8, 4, 1000, 64]" = torch.ops.aten.clone.default(expand_default_5, memory_format = torch.contiguous_format);  expand_default_5 = None
        reshape_default_7: "bf16[1, 32, 1000, 64]" = torch.ops.aten.reshape.default(clone_default_1, [1, 32, 1000, 64]);  clone_default_1 = None
        unsqueeze_default_14: "bf16[1, 8, 1, 1000, 64]" = torch.ops.aten.unsqueeze.default(permute_default_3, 2);  permute_default_3 = None
        expand_default_6: "bf16[1, 8, 4, 1000, 64]" = torch.ops.aten.expand.default(unsqueeze_default_14, [1, 8, 4, 1000, 64]);  unsqueeze_default_14 = None
        clone_default_2: "bf16[1, 8, 4, 1000, 64]" = torch.ops.aten.clone.default(expand_default_6, memory_format = torch.contiguous_format);  expand_default_6 = None
        reshape_default_8: "bf16[1, 32, 1000, 64]" = torch.ops.aten.reshape.default(clone_default_2, [1, 32, 1000, 64]);  clone_default_2 = None
        full_default: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand_default, full_default_1, full_default);  expand_default = full_default_1 = full_default = None
        expand_default_7: "bf16[1, 32, 1000, 1000]" = torch.ops.aten.expand.default(where_self, [1, 32, 1000, 1000]);  where_self = None
        return (add_tensor_3, reshape_default_7, reshape_default_8, expand_default_7)


def _default_make_inputs():
    return [
    torch.randn([32], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1000, 2048], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1000, 512], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1000, 512], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
