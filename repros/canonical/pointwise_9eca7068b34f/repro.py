"""
Standalone repro captured via capture_hook.
Label: torchbench_moondream_infer
Pattern hash: 9eca7068b34f
Shape hash: 72c549a6
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
    def forward(self, addmm_138: "f16[512, 2048]", convert_element_type_2: "f16[1, 512, 32]", convert_element_type_3: "f16[1, 512, 32]", addmm_139: "f16[512, 2048]", addmm_140: "f16[512, 2048]", expand: "b8[1, 1, 512, 512]", convert_element_type_465: "f16[1, 512, 2048]", arg335_1: "f16[8192, 2048]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/utils/_device.py:122 in __torch_function__, code: return func(*args, **kwargs)
        reshape_default: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_138, _shape_param_0);  addmm_138 = _shape_param_0 = None
        reshape_default_1: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None
        slice_tensor: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_default, 3, 0, 32)
        unsqueeze_default: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1);  convert_element_type_2 = None
        mul_tensor: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_tensor, unsqueeze_default)
        slice_tensor_1: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_tensor, 3, 16, 9223372036854775807)
        neg_default: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_tensor_1);  slice_tensor_1 = None
        slice_tensor_2: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_tensor, 3, 0, 16);  slice_tensor = None
        cat_default: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_default, slice_tensor_2], -1);  neg_default = slice_tensor_2 = None
        unsqueeze_default_1: "f16[1, 1, 512, 32]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1);  convert_element_type_3 = None
        mul_tensor_1: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_default, unsqueeze_default_1);  cat_default = None
        add_tensor: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        slice_tensor_3: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_default, 3, 32, 9223372036854775807);  permute_default = None
        cat_default_1: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_tensor, slice_tensor_3], -1);  add_tensor = slice_tensor_3 = None
        reshape_default_2: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_139, _shape_param_2);  addmm_139 = _shape_param_2 = None
        reshape_default_3: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_1: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None
        slice_tensor_4: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_default_1, 3, 0, 32)
        mul_tensor_2: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(slice_tensor_4, unsqueeze_default);  unsqueeze_default = None
        slice_tensor_5: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_tensor_4, 3, 16, 9223372036854775807)
        neg_default_1: "f16[1, 32, 512, 16]" = torch.ops.aten.neg.default(slice_tensor_5);  slice_tensor_5 = None
        slice_tensor_6: "f16[1, 32, 512, 16]" = torch.ops.aten.slice.Tensor(slice_tensor_4, 3, 0, 16);  slice_tensor_4 = None
        cat_default_2: "f16[1, 32, 512, 32]" = torch.ops.aten.cat.default([neg_default_1, slice_tensor_6], -1);  neg_default_1 = slice_tensor_6 = None
        mul_tensor_3: "f16[1, 32, 512, 32]" = torch.ops.aten.mul.Tensor(cat_default_2, unsqueeze_default_1);  cat_default_2 = unsqueeze_default_1 = None
        add_tensor_1: "f16[1, 32, 512, 32]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        slice_tensor_7: "f16[1, 32, 512, 32]" = torch.ops.aten.slice.Tensor(permute_default_1, 3, 32, 9223372036854775807);  permute_default_1 = None
        cat_default_3: "f16[1, 32, 512, 64]" = torch.ops.aten.cat.default([add_tensor_1, slice_tensor_7], -1);  add_tensor_1 = slice_tensor_7 = None
        reshape_default_4: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_140, _shape_param_4);  addmm_140 = _shape_param_4 = None
        reshape_default_5: "f16[1, 512, 32, 64]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_5);  reshape_default_4 = _shape_param_5 = None
        permute_default_2: "f16[1, 32, 512, 64]" = torch.ops.aten.permute.default(reshape_default_5, [0, 2, 1, 3]);  reshape_default_5 = None
        full_default: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default, full_default_1);  expand = full_default = full_default_1 = None
        reshape_default_6: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_465, _shape_param_6);  convert_element_type_465 = _shape_param_6 = None
        permute_default_3: "f16[2048, 8192]" = torch.ops.aten.permute.default(arg335_1, [1, 0]);  arg335_1 = None
        return (cat_default_1, cat_default_3, permute_default_2, where_self, reshape_default_6, permute_default_3)


def _default_make_inputs():
    return []


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
