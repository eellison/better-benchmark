"""
Standalone repro captured via capture_hook.
Label: hf_GPTJForCausalLM_train_001
Pattern hash: 303cef4ec79e
Shape hash: 17e942a6
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([1, 16, 128, 256], f32, stride=(524288, 256, 4096, 1)), T([1, 16, 128, 256], f32, stride=(524288, 256, 4096, 1)), T([1, 128, 1, 32, 1], f32), T([1, 128, 1, 32, 1], f32), S([1, 128, 1, 32, 2]), S([1, 128, 1, 64]), S([1, 128, 16, 32, 2]), S([1, 128, 1, 32, 2]), S([1, 128, 1, 64]), S([1, 128, 16, 32, 2]), S([1, 128, 4096]), S([1, 128, 4096]), S([128, 4096]), S([128, 4096]))"

class Repro(torch.nn.Module):
    def forward(self, getitem: "f32[1, 16, 128, 256]", getitem_1: "f32[1, 16, 128, 256]", arg583_1: "f32[1, 128, 1, 32, 1]", arg584_1: "f32[1, 128, 1, 32, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9):
        # No stacktrace found for following nodes
        permute_default: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3]);  getitem = None
        permute_default_1: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_1, [0, 2, 1, 3]);  getitem_1 = None
        slice_tensor: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_default, 3, 0, 64)
        slice_tensor_1: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_default, 3, 64, 256);  permute_default = None
        slice_tensor_2: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_default_1, 3, 0, 64)
        slice_tensor_3: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_default_1, 3, 64, 256);  permute_default_1 = None
        expand_default: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg583_1, _shape_param_0);  arg583_1 = _shape_param_0 = None
        clone_default: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default: "f32[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None
        mul_tensor: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_tensor, view_default)
        view_default_1: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_tensor, _shape_param_2);  mul_tensor = _shape_param_2 = None
        select_int: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_default_1, -1, 0)
        select_int_1: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_default_1, -1, 1);  view_default_1 = None
        neg_default: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_int);  select_int = None
        full_default: "f32[1, 128, 16, 64]" = torch.ops.aten.full.default([1, 128, 16, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default, neg_default, 3, 1, 9223372036854775807, 2);  neg_default = None
        slice_scatter_default_1: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default, select_int_1, 3, 0, 9223372036854775807, 2);  select_int_1 = None
        add_tensor: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_default, slice_scatter_default_1);  slice_scatter_default = slice_scatter_default_1 = None
        expand_default_1: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg584_1, _shape_param_3);  arg584_1 = _shape_param_3 = None
        clone_default_1: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        view_default_2: "f32[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_default_1, _shape_param_4);  clone_default_1 = _shape_param_4 = None
        mul_tensor_1: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_tensor, view_default_2);  slice_tensor = None
        add_tensor_1: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_tensor, mul_tensor_1);  add_tensor = mul_tensor_1 = None
        mul_tensor_2: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_tensor_2, view_default);  view_default = None
        view_default_3: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_tensor_2, _shape_param_5);  mul_tensor_2 = _shape_param_5 = None
        select_int_2: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_default_3, -1, 0)
        select_int_3: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_default_3, -1, 1);  view_default_3 = None
        neg_default_1: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_int_2);  select_int_2 = None
        slice_scatter_default_2: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default, neg_default_1, 3, 1, 9223372036854775807, 2);  neg_default_1 = None
        slice_scatter_default_3: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default, select_int_3, 3, 0, 9223372036854775807, 2);  full_default = select_int_3 = None
        add_tensor_2: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_default_2, slice_scatter_default_3);  slice_scatter_default_2 = slice_scatter_default_3 = None
        mul_tensor_3: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_tensor_2, view_default_2);  slice_tensor_2 = view_default_2 = None
        add_tensor_3: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_3);  add_tensor_2 = mul_tensor_3 = None
        full_default_1: "f32[1, 128, 16, 256]" = torch.ops.aten.full.default([1, 128, 16, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default_4: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_1, slice_tensor_1, 3, 64, 9223372036854775807);  slice_tensor_1 = None
        slice_scatter_default_5: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_1, add_tensor_1, 3, 0, 64);  add_tensor_1 = None
        add_tensor_4: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_default_4, slice_scatter_default_5);  slice_scatter_default_4 = slice_scatter_default_5 = None
        slice_scatter_default_6: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_1, slice_tensor_3, 3, 64, 9223372036854775807);  slice_tensor_3 = None
        slice_scatter_default_7: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_1, add_tensor_3, 3, 0, 64);  full_default_1 = add_tensor_3 = None
        add_tensor_5: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_default_6, slice_scatter_default_7);  slice_scatter_default_6 = slice_scatter_default_7 = None
        view_default_4: "f32[1, 128, 4096]" = torch.ops.aten.view.default(add_tensor_5, _shape_param_6);  add_tensor_5 = _shape_param_6 = None
        view_default_5: "f32[1, 128, 4096]" = torch.ops.aten.view.default(add_tensor_4, _shape_param_7);  add_tensor_4 = _shape_param_7 = None
        view_default_6: "f32[128, 4096]" = torch.ops.aten.view.default(view_default_4, _shape_param_8);  view_default_4 = _shape_param_8 = None
        view_default_7: "f32[128, 4096]" = torch.ops.aten.view.default(view_default_5, _shape_param_9);  view_default_5 = _shape_param_9 = None
        return (view_default_7, view_default_6)



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
