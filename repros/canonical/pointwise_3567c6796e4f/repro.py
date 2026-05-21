"""
Standalone repro captured via capture_hook.
Label: torchbench_llava_infer
Pattern hash: 3567c6796e4f
Shape hash: 1e82871f
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
    def forward(self, mm_217: "f16[512, 4096]", convert_element_type_2: "f16[1, 512, 128]", convert_element_type_3: "f16[1, 512, 128]", mm_218: "f16[512, 4096]", mm_219: "f16[512, 4096]", expand: "b8[1, 1, 512, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/utils/_device.py:122 in __torch_function__, code: return func(*args, **kwargs)
        reshape_default: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_217, _shape_param_0);  mm_217 = _shape_param_0 = None
        reshape_default_1: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None
        unsqueeze_default: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1);  convert_element_type_2 = None
        mul_tensor: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_default, unsqueeze_default)
        slice_tensor: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default, 3, 64, 9223372036854775807)
        neg_default: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_tensor);  slice_tensor = None
        slice_tensor_1: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default, 3, 0, 64);  permute_default = None
        cat_default: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_default, slice_tensor_1], -1);  neg_default = slice_tensor_1 = None
        unsqueeze_default_1: "f16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1);  convert_element_type_3 = None
        mul_tensor_1: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_default, unsqueeze_default_1);  cat_default = None
        add_tensor: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        reshape_default_2: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_218, _shape_param_2);  mm_218 = _shape_param_2 = None
        reshape_default_3: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_1: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None
        mul_tensor_2: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(permute_default_1, unsqueeze_default);  unsqueeze_default = None
        slice_tensor_2: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default_1, 3, 64, 9223372036854775807)
        neg_default_1: "f16[1, 32, 512, 64]" = torch.ops.aten.neg.default(slice_tensor_2);  slice_tensor_2 = None
        slice_tensor_3: "f16[1, 32, 512, 64]" = torch.ops.aten.slice.Tensor(permute_default_1, 3, 0, 64);  permute_default_1 = None
        cat_default_1: "f16[1, 32, 512, 128]" = torch.ops.aten.cat.default([neg_default_1, slice_tensor_3], -1);  neg_default_1 = slice_tensor_3 = None
        mul_tensor_3: "f16[1, 32, 512, 128]" = torch.ops.aten.mul.Tensor(cat_default_1, unsqueeze_default_1);  cat_default_1 = unsqueeze_default_1 = None
        add_tensor_1: "f16[1, 32, 512, 128]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        reshape_default_4: "f16[1, 512, 4096]" = torch.ops.aten.reshape.default(mm_219, _shape_param_4);  mm_219 = _shape_param_4 = None
        reshape_default_5: "f16[1, 512, 32, 128]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_5);  reshape_default_4 = _shape_param_5 = None
        permute_default_2: "f16[1, 32, 512, 128]" = torch.ops.aten.permute.default(reshape_default_5, [0, 2, 1, 3]);  reshape_default_5 = None
        full_default: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f16[1, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default, full_default_1);  expand = full_default = full_default_1 = None
        return (add_tensor, add_tensor_1, permute_default_2, where_self)


def _default_make_inputs():
    return []


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
