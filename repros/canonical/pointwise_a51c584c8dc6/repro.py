"""
Standalone repro captured via capture_hook.
Label: torchbench_llava_infer
Pattern hash: a51c584c8dc6
Shape hash: b6b9b8e8
"""
import sys
from pathlib import Path

import sys
from pathlib import Path
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm: "f16[512, 4096]", arg2_1: "f16[64]", mm_1: "f16[512, 4096]", mm_2: "f16[512, 4096]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/utils/_device.py:122 in __torch_function__, code: return func(*args, **kwargs)
        reshape_default: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        reshape_default_1: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None
        unsqueeze_default: "f16[1, 64]" = torch.ops.aten.unsqueeze.default(arg2_1, 0);  arg2_1 = None
        unsqueeze_default_1: "f16[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        convert_element_type_default: "f32[1, 64, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_default_1, torch.float32);  unsqueeze_default_1 = None
        expand_default: "f32[1, 64, 1]" = torch.ops.aten.expand.default(convert_element_type_default, [1, -1, 1]);  convert_element_type_default = None
        expand_default_1: "f32[1, 64, 1]" = torch.ops.aten.expand.default(expand_default, _shape_param_2);  expand_default = _shape_param_2 = None
        iota_default: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[512]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None
        unsqueeze_default_2: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None
        unsqueeze_default_3: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 1);  unsqueeze_default_2 = None
        convert_element_type_default_1: "f32[1, 1, 512]" = torch.ops.prims.convert_element_type.default(unsqueeze_default_3, torch.float32);  unsqueeze_default_3 = None
        expand_default_2: "f32[1, 1, 512]" = torch.ops.aten.expand.default(convert_element_type_default_1, _shape_param_3);  convert_element_type_default_1 = _shape_param_3 = None
        mul_tensor: "f32[1, 64, 512]" = torch.ops.aten.mul.Tensor(expand_default_1, expand_default_2);  expand_default_1 = expand_default_2 = None
        permute_default_1: "f32[1, 512, 64]" = torch.ops.aten.permute.default(mul_tensor, [0, 2, 1]);  mul_tensor = None
        unsqueeze_default_4: "f32[1, 512, 1, 64]" = torch.ops.aten.unsqueeze.default(permute_default_1, 2);  permute_default_1 = None
        expand_default_3: "f32[1, 512, 2, 64]" = torch.ops.aten.expand.default(unsqueeze_default_4, _shape_param_4);  unsqueeze_default_4 = _shape_param_4 = None
        clone_default: "f32[1, 512, 2, 64]" = torch.ops.aten.clone.default(expand_default_3, memory_format = torch.contiguous_format);  expand_default_3 = None
        reshape_default_2: "f32[1, 512, 128]" = torch.ops.aten.reshape.default(clone_default, _shape_param_5);  clone_default = _shape_param_5 = None
        cos_default: "f32[1, 512, 128]" = torch.ops.aten.cos.default(reshape_default_2)
        mul_tensor_1: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(cos_default, 1.0);  cos_default = None
        convert_element_type_default_2: "f16[1, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.float16);  mul_tensor_1 = None
        unsqueeze_default_5: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 1);  convert_element_type_default_2 = None
        mul_tensor_2: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_default, unsqueeze_default_5)
        slice_tensor: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default, 3, 64, 9223372036854775807)
        neg_default: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_tensor);  slice_tensor = None
        slice_tensor_1: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default, 3, 0, 64);  permute_default = None
        cat_default: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_default, slice_tensor_1], -1);  neg_default = slice_tensor_1 = None
        sin_default: "f32[1, 512, 128]" = torch.ops.aten.sin.default(reshape_default_2);  reshape_default_2 = None
        mul_tensor_3: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sin_default, 1.0);  sin_default = None
        convert_element_type_default_3: "f16[1, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.float16);  mul_tensor_3 = None
        unsqueeze_default_6: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_3, 1);  convert_element_type_default_3 = None
        mul_tensor_4: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_default, unsqueeze_default_6);  cat_default = None
        add_tensor_1: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_4);  mul_tensor_2 = mul_tensor_4 = None
        reshape_default_3: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_1, _shape_param_6);  mm_1 = _shape_param_6 = None
        reshape_default_4: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_7);  reshape_default_3 = _shape_param_7 = None
        permute_default_2: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(reshape_default_4, [0, 2, 1, 3]);  reshape_default_4 = None
        mul_tensor_5: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_default_2, unsqueeze_default_5);  unsqueeze_default_5 = None
        slice_tensor_2: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default_2, 3, 64, 9223372036854775807)
        neg_default_1: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_tensor_2);  slice_tensor_2 = None
        slice_tensor_3: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default_2, 3, 0, 64);  permute_default_2 = None
        cat_default_1: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_default_1, slice_tensor_3], -1);  neg_default_1 = slice_tensor_3 = None
        mul_tensor_6: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_default_1, unsqueeze_default_6);  cat_default_1 = unsqueeze_default_6 = None
        add_tensor_2: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_tensor_5, mul_tensor_6);  mul_tensor_5 = mul_tensor_6 = None
        reshape_default_5: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_2, _shape_param_8);  mm_2 = _shape_param_8 = None
        reshape_default_6: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(reshape_default_5, _shape_param_9);  reshape_default_5 = _shape_param_9 = None
        permute_default_3: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(reshape_default_6, [0, 2, 1, 3]);  reshape_default_6 = None
        iota_default_1: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor_3: "i64[512]" = torch.ops.aten.add.Tensor(iota_default_1, 0);  iota_default_1 = None
        unsqueeze_default_7: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_tensor_3, 0);  add_tensor_3 = None
        unsqueeze_default_8: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 1);  unsqueeze_default_7 = None
        unsqueeze_default_9: "i64[1, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 2);  unsqueeze_default_8 = None
        iota_default_2: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor_4: "i64[512]" = torch.ops.aten.add.Tensor(iota_default_2, 0);  iota_default_2 = None
        unsqueeze_default_10: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_tensor_4, 0);  add_tensor_4 = None
        unsqueeze_default_11: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 1);  unsqueeze_default_10 = None
        unsqueeze_default_12: "i64[1, 1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        le_tensor: "b8[1, 1, 512, 512]" = torch.ops.aten.le.Tensor(unsqueeze_default_9, unsqueeze_default_12);  unsqueeze_default_9 = unsqueeze_default_12 = None
        expand_default_4: "b8[1, 1, 512, 512]" = torch.ops.aten.expand.default(le_tensor, _shape_param_10);  le_tensor = _shape_param_10 = None
        full_default: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand_default_4, full_default, full_default_1);  expand_default_4 = full_default = full_default_1 = None
        return (add_tensor_1, add_tensor_2, permute_default_3, where_self)


def _default_make_inputs():
    return []


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
