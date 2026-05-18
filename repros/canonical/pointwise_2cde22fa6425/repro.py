"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s5_g11
Pattern hash: 2cde22fa6425
Shape hash: d339fbc9
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
    def forward(self, mm_105: "bf16[1000, 2048]", mm_106: "bf16[1000, 512]", mm_107: "bf16[1000, 512]", convert_element_type_2: "bf16[1, 1000, 64]", convert_element_type_3: "bf16[1, 1000, 64]", expand: "b8[1, 1, 1000, 1000]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10):
        # No stacktrace found for following nodes
        reshape_default: "bf16[1, 1000, 2048]" = torch.ops.aten.reshape.default(mm_105, _shape_param_0);  mm_105 = _shape_param_0 = None
        reshape_default_1: "bf16[1, 1000, 32, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "bf16[1, 32, 1000, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None
        reshape_default_2: "bf16[1, 1000, 512]" = torch.ops.aten.reshape.default(mm_106, _shape_param_2);  mm_106 = _shape_param_2 = None
        reshape_default_3: "bf16[1, 1000, 8, 64]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_1: "bf16[1, 8, 1000, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None
        reshape_default_4: "bf16[1, 1000, 512]" = torch.ops.aten.reshape.default(mm_107, _shape_param_4);  mm_107 = _shape_param_4 = None
        reshape_default_5: "bf16[1, 1000, 8, 64]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_5);  reshape_default_4 = _shape_param_5 = None
        permute_default_2: "bf16[1, 8, 1000, 64]" = torch.ops.aten.permute.default(reshape_default_5, [0, 2, 1, 3]);  reshape_default_5 = None
        unsqueeze_default: "bf16[1, 1, 1000, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1);  convert_element_type_2 = None
        unsqueeze_default_1: "bf16[1, 1, 1000, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1);  convert_element_type_3 = None
        mul_tensor: "bf16[1, 32, 1000, 64]" = torch.ops.aten.mul.Tensor(permute_default, unsqueeze_default)
        slice_tensor: "bf16[1, 32, 1000, 32]" = torch.ops.aten.slice.Tensor(permute_default, 3, 0, 32)
        slice_tensor_1: "bf16[1, 32, 1000, 32]" = torch.ops.aten.slice.Tensor(permute_default, 3, 32, 9223372036854775807);  permute_default = None
        neg_default: "bf16[1, 32, 1000, 32]" = torch.ops.aten.neg.default(slice_tensor_1);  slice_tensor_1 = None
        cat_default: "bf16[1, 32, 1000, 64]" = torch.ops.aten.cat.default([neg_default, slice_tensor], -1);  neg_default = slice_tensor = None
        mul_tensor_1: "bf16[1, 32, 1000, 64]" = torch.ops.aten.mul.Tensor(cat_default, unsqueeze_default_1);  cat_default = None
        add_tensor: "bf16[1, 32, 1000, 64]" = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        mul_tensor_2: "bf16[1, 8, 1000, 64]" = torch.ops.aten.mul.Tensor(permute_default_1, unsqueeze_default);  unsqueeze_default = None
        slice_tensor_2: "bf16[1, 8, 1000, 32]" = torch.ops.aten.slice.Tensor(permute_default_1, 3, 0, 32)
        slice_tensor_3: "bf16[1, 8, 1000, 32]" = torch.ops.aten.slice.Tensor(permute_default_1, 3, 32, 9223372036854775807);  permute_default_1 = None
        neg_default_1: "bf16[1, 8, 1000, 32]" = torch.ops.aten.neg.default(slice_tensor_3);  slice_tensor_3 = None
        cat_default_1: "bf16[1, 8, 1000, 64]" = torch.ops.aten.cat.default([neg_default_1, slice_tensor_2], -1);  neg_default_1 = slice_tensor_2 = None
        mul_tensor_3: "bf16[1, 8, 1000, 64]" = torch.ops.aten.mul.Tensor(cat_default_1, unsqueeze_default_1);  cat_default_1 = unsqueeze_default_1 = None
        add_tensor_1: "bf16[1, 8, 1000, 64]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_2: "bf16[1, 8, 1, 1000, 64]" = torch.ops.aten.unsqueeze.default(add_tensor_1, 2);  add_tensor_1 = None
        expand_default: "bf16[1, 8, 4, 1000, 64]" = torch.ops.aten.expand.default(unsqueeze_default_2, _shape_param_6);  unsqueeze_default_2 = _shape_param_6 = None
        clone_default: "bf16[1, 8, 4, 1000, 64]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_6: "bf16[1, 32, 1000, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_7);  clone_default = _shape_param_7 = None
        unsqueeze_default_3: "bf16[1, 8, 1, 1000, 64]" = torch.ops.aten.unsqueeze.default(permute_default_2, 2);  permute_default_2 = None
        expand_default_1: "bf16[1, 8, 4, 1000, 64]" = torch.ops.aten.expand.default(unsqueeze_default_3, _shape_param_8);  unsqueeze_default_3 = _shape_param_8 = None
        clone_default_1: "bf16[1, 8, 4, 1000, 64]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_7: "bf16[1, 32, 1000, 64]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_9);  clone_default_1 = _shape_param_9 = None
        full_default: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_default_1, full_default);  expand = full_default_1 = full_default = None
        expand_default_2: "bf16[1, 32, 1000, 1000]" = torch.ops.aten.expand.default(where_self, _shape_param_10);  where_self = _shape_param_10 = None
        return (add_tensor, reshape_default_6, reshape_default_7, expand_default_2)


def _default_make_inputs():
    return [
    torch.randn([1000, 2048], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1000, 512], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1000, 512], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 1000, 64], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 1000, 64], dtype=torch.bfloat16, device='cuda'),
    torch.randint(0, 2, [1, 1, 1000, 1000], dtype=torch.bool, device='cuda'),
    [1, 1000, 2048],  # _shape_param_0
    [1, 1000, -1, 64],  # _shape_param_1
    [1, 1000, 512],  # _shape_param_2
    [1, 1000, -1, 64],  # _shape_param_3
    [1, 1000, 512],  # _shape_param_4
    [1, 1000, -1, 64],  # _shape_param_5
    [1, 8, 4, 1000, 64],  # _shape_param_6
    [1, 32, 1000, 64],  # _shape_param_7
    [1, 8, 4, 1000, 64],  # _shape_param_8
    [1, 32, 1000, 64],  # _shape_param_9
    [1, 32, 1000, 1000],  # _shape_param_10
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
