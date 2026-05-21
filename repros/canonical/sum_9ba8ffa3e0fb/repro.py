"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_train_005
Pattern hash: 9ba8ffa3e0fb
Shape hash: bb3bb760
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
_shapes_config = "(T([384, 256, 768], f32), T([96, 4, 256, 769], f32), T([96, 4, 197120], f32), T([8, 1024, 12, 513], b8), T([8, 1024, 1, 1], b8), T([], f32), T([8, 1024, 12, 513], f32), T([8, 256, 12, 257], f32), T([8, 256, 12, 257], b8), T([8, 256, 12, 513], f32), T([8, 1024, 12, 513], f32), T([8, 256, 12, 257], b8), T([96, 255, 255], f32), T([96, 255, 513], f32), T([96, 512, 513], f32), T([96, 3, 512, 513], f32), T([96, 3, 256, 256], f32), T([96, 3, 256, 513], f32), T([96, 256, 257], f32), T([96, 256, 513], f32), S([96, 4, 256, 768, 1]), S([96, 4, 196864]), S([96, 4, 256, 770]), S([8, 12, 1024, 513]), S([96, 4, 256, 513]), S([8, 12, 1024, 513]), S([96, 4, 256, 513]), S([8, 12, 1024, 513]), S([96, 4, 256, 513]), S([96, 4, 256, 513]), S([8, 12, 1024, 513]), S([96, 4, 256, 513]), S([96, 4, 256, 513]), S([96, 3, 513, 512]), S([96, 3, 512, 512, 1]), S([288, 512, 512]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_45: "f32[384, 256, 768]", full: "f32[96, 4, 256, 769]", full_1: "f32[96, 4, 197120]", arg101_1: "b8[8, 1024, 12, 513]", unsqueeze_1: "b8[8, 1024, 1, 1]", full_3: "f32[]", arg100_1: "f32[8, 1024, 12, 513]", full_4: "f32[8, 256, 12, 257]", convert_element_type_3: "b8[8, 256, 12, 257]", full_5: "f32[8, 256, 12, 513]", full_6: "f32[8, 1024, 12, 513]", convert_element_type_4: "b8[8, 256, 12, 257]", full_7: "f32[96, 255, 255]", full_8: "f32[96, 255, 513]", full_9: "f32[96, 512, 513]", full_10: "f32[96, 3, 512, 513]", full_11: "f32[96, 3, 256, 256]", full_12: "f32[96, 3, 256, 513]", full_13: "f32[96, 256, 257]", full_14: "f32[96, 256, 513]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15):
        # No stacktrace found for following nodes
        view_default: "f32[96, 4, 256, 768, 1]" = torch.ops.aten.view.default(bmm_45, _shape_param_0);  bmm_45 = _shape_param_0 = None
        squeeze_dim: "f32[96, 4, 256, 768]" = torch.ops.aten.squeeze.dim(view_default, 4);  view_default = None
        slice_scatter_default: "f32[96, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full, squeeze_dim, 3, 0, -1);  full = squeeze_dim = None
        view_default_1: "f32[96, 4, 196864]" = torch.ops.aten.view.default(slice_scatter_default, _shape_param_1);  slice_scatter_default = _shape_param_1 = None
        slice_scatter_default_1: "f32[96, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_1, view_default_1, 2, 0, -256);  full_1 = view_default_1 = None
        view_default_2: "f32[96, 4, 256, 770]" = torch.ops.aten.view.default(slice_scatter_default_1, _shape_param_2);  slice_scatter_default_1 = _shape_param_2 = None
        constant_pad_nd_default: "f32[96, 4, 256, 513]" = torch.ops.aten.constant_pad_nd.default(view_default_2, [0, -257]);  view_default_2 = None
        view_default_3: "f32[8, 12, 1024, 513]" = torch.ops.aten.view.default(constant_pad_nd_default, _shape_param_3);  constant_pad_nd_default = _shape_param_3 = None
        permute_default: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_default_3, [0, 2, 1, 3]);  view_default_3 = None
        convert_element_type_default: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(arg101_1, torch.float32);  arg101_1 = None
        mul_tensor: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.0);  convert_element_type_default = None
        mul_tensor_1: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(permute_default, mul_tensor);  permute_default = mul_tensor = None
        clone_default: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(mul_tensor_1, memory_format = torch.contiguous_format);  mul_tensor_1 = None
        where_self: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_1, full_3, clone_default);  unsqueeze_1 = clone_default = None
        mul_tensor_2: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(where_self, arg100_1);  where_self = None
        sum_dim_int_list: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [-1], True)
        neg_default: "f32[8, 1024, 12, 513]" = torch.ops.aten.neg.default(arg100_1);  arg100_1 = None
        fma_default: "f32[8, 1024, 12, 513]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_2);  neg_default = sum_dim_int_list = mul_tensor_2 = None
        permute_default_1: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(fma_default, [0, 2, 1, 3]);  fma_default = None
        clone_default_1: "f32[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_4: "f32[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_default_1, _shape_param_4);  clone_default_1 = _shape_param_4 = None
        view_default_5: "f32[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_default_4, _shape_param_5);  view_default_4 = _shape_param_5 = None
        permute_default_2: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_default_5, [0, 2, 1, 3]);  view_default_5 = None
        clone_default_2: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format)
        copy_default: "f32[8, 1024, 12, 513]" = torch.ops.aten.copy.default(permute_default_2, clone_default_2);  permute_default_2 = clone_default_2 = None
        permute_default_3: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(copy_default, [0, 2, 1, 3]);  copy_default = None
        view_default_6: "f32[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_default_3, _shape_param_6);  permute_default_3 = _shape_param_6 = None
        view_default_7: "f32[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_default_6, _shape_param_7);  view_default_6 = _shape_param_7 = None
        permute_default_4: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_default_7, [0, 2, 1, 3]);  view_default_7 = None
        slice_tensor: "f32[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_default_4, 1, -256, 9223372036854775807)
        slice_tensor_1: "f32[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_tensor, 3, -257, 9223372036854775807)
        clone_default_3: "f32[8, 256, 12, 257]" = torch.ops.aten.clone.default(slice_tensor_1, memory_format = torch.contiguous_format)
        copy_default_1: "f32[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_tensor_1, full_4);  slice_tensor_1 = None
        slice_scatter_default_2: "f32[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor, copy_default_1, 3, -257, 9223372036854775807);  slice_tensor = copy_default_1 = None
        slice_scatter_default_3: "f32[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_default_4, slice_scatter_default_2, 1, -256, 9223372036854775807);  permute_default_4 = slice_scatter_default_2 = None
        permute_default_5: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_3, [0, 2, 1, 3]);  slice_scatter_default_3 = None
        view_default_8: "f32[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_default_5, _shape_param_8);  permute_default_5 = _shape_param_8 = None
        where_self_1: "f32[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_3, full_3, clone_default_3);  convert_element_type_3 = clone_default_3 = None
        slice_scatter_default_4: "f32[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_5, where_self_1, 3, -257, 9223372036854775807);  where_self_1 = None
        slice_scatter_default_5: "f32[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_6, slice_scatter_default_4, 1, -256, 9223372036854775807);  slice_scatter_default_4 = None
        permute_default_6: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_5, [0, 2, 1, 3]);  slice_scatter_default_5 = None
        clone_default_4: "f32[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_default_6, memory_format = torch.contiguous_format);  permute_default_6 = None
        view_default_9: "f32[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_default_4, _shape_param_9);  clone_default_4 = _shape_param_9 = None
        add_tensor: "f32[96, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_default_8, view_default_9);  view_default_8 = view_default_9 = None
        view_default_10: "f32[8, 12, 1024, 513]" = torch.ops.aten.view.default(add_tensor, _shape_param_10);  add_tensor = _shape_param_10 = None
        permute_default_7: "f32[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_default_10, [0, 2, 1, 3]);  view_default_10 = None
        slice_tensor_2: "f32[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_default_7, 1, 0, 256)
        slice_tensor_3: "f32[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_2, 3, 0, 257)
        clone_default_5: "f32[8, 256, 12, 257]" = torch.ops.aten.clone.default(slice_tensor_3, memory_format = torch.contiguous_format)
        copy_default_2: "f32[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_tensor_3, full_4);  slice_tensor_3 = full_4 = None
        slice_scatter_default_6: "f32[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_2, copy_default_2, 3, 0, 257);  slice_tensor_2 = copy_default_2 = None
        slice_scatter_default_7: "f32[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_default_7, slice_scatter_default_6, 1, 0, 256);  permute_default_7 = slice_scatter_default_6 = None
        permute_default_8: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_7, [0, 2, 1, 3]);  slice_scatter_default_7 = None
        view_default_11: "f32[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_default_8, _shape_param_11);  permute_default_8 = _shape_param_11 = None
        where_self_2: "f32[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_4, full_3, clone_default_5);  convert_element_type_4 = full_3 = clone_default_5 = None
        slice_scatter_default_8: "f32[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_5, where_self_2, 3, 0, 257);  full_5 = where_self_2 = None
        slice_scatter_default_9: "f32[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_6, slice_scatter_default_8, 1, 0, 256);  full_6 = slice_scatter_default_8 = None
        permute_default_9: "f32[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_9, [0, 2, 1, 3]);  slice_scatter_default_9 = None
        clone_default_6: "f32[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_default_9, memory_format = torch.contiguous_format);  permute_default_9 = None
        view_default_12: "f32[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_default_6, _shape_param_12);  clone_default_6 = _shape_param_12 = None
        add_tensor_1: "f32[96, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_default_11, view_default_12);  view_default_11 = view_default_12 = None
        select_int: "f32[96, 256, 513]" = torch.ops.aten.select.int(add_tensor_1, 1, 0)
        slice_tensor_4: "f32[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_int, 1, 1, 256)
        slice_tensor_5: "f32[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_tensor_4, 2, 1, 256)
        clone_default_7: "f32[96, 255, 255]" = torch.ops.aten.clone.default(slice_tensor_5, memory_format = torch.contiguous_format)
        copy_default_3: "f32[96, 255, 255]" = torch.ops.aten.copy.default(slice_tensor_5, full_7);  slice_tensor_5 = full_7 = None
        slice_scatter_default_10: "f32[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_4, copy_default_3, 2, 1, 256);  slice_tensor_4 = copy_default_3 = None
        slice_scatter_default_11: "f32[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int, slice_scatter_default_10, 1, 1, 256);  select_int = slice_scatter_default_10 = None
        select_scatter_default: "f32[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(add_tensor_1, slice_scatter_default_11, 1, 0);  add_tensor_1 = slice_scatter_default_11 = None
        slice_scatter_default_12: "f32[96, 255, 513]" = torch.ops.aten.slice_scatter.default(full_8, clone_default_7, 2, -255, 9223372036854775807);  full_8 = clone_default_7 = None
        slice_scatter_default_13: "f32[96, 512, 513]" = torch.ops.aten.slice_scatter.default(full_9, slice_scatter_default_12, 1, 0, 255);  slice_scatter_default_12 = None
        select_scatter_default_1: "f32[96, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_10, slice_scatter_default_13, 1, 0);  slice_scatter_default_13 = None
        slice_tensor_6: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default, 1, 1, 9223372036854775807)
        slice_tensor_7: "f32[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_tensor_6, 3, 0, 256)
        clone_default_8: "f32[96, 3, 256, 256]" = torch.ops.aten.clone.default(slice_tensor_7, memory_format = torch.contiguous_format)
        copy_default_4: "f32[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_tensor_7, full_11);  slice_tensor_7 = full_11 = None
        slice_scatter_default_14: "f32[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_6, copy_default_4, 3, 0, 256);  slice_tensor_6 = copy_default_4 = None
        slice_scatter_default_15: "f32[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default, slice_scatter_default_14, 1, 1, 9223372036854775807);  select_scatter_default = slice_scatter_default_14 = None
        slice_scatter_default_16: "f32[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_12, clone_default_8, 3, 257, 9223372036854775807);  clone_default_8 = None
        slice_scatter_default_17: "f32[96, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_10, slice_scatter_default_16, 2, -257, -1);  slice_scatter_default_16 = None
        add_tensor_2: "f32[96, 3, 512, 513]" = torch.ops.aten.add.Tensor(select_scatter_default_1, slice_scatter_default_17);  select_scatter_default_1 = slice_scatter_default_17 = None
        select_int_1: "f32[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_15, 1, -1)
        slice_tensor_8: "f32[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_int_1, 2, 256, 9223372036854775807)
        clone_default_9: "f32[96, 256, 257]" = torch.ops.aten.clone.default(slice_tensor_8, memory_format = torch.contiguous_format)
        copy_default_5: "f32[96, 256, 257]" = torch.ops.aten.copy.default(slice_tensor_8, full_13);  slice_tensor_8 = full_13 = None
        slice_scatter_default_18: "f32[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_1, copy_default_5, 2, 256, 9223372036854775807);  select_int_1 = copy_default_5 = None
        select_scatter_default_2: "f32[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_15, slice_scatter_default_18, 1, -1);  slice_scatter_default_15 = slice_scatter_default_18 = None
        slice_scatter_default_19: "f32[96, 256, 513]" = torch.ops.aten.slice_scatter.default(full_14, clone_default_9, 2, 0, 257);  full_14 = clone_default_9 = None
        slice_scatter_default_20: "f32[96, 512, 513]" = torch.ops.aten.slice_scatter.default(full_9, slice_scatter_default_19, 1, 256, 9223372036854775807);  full_9 = slice_scatter_default_19 = None
        select_scatter_default_3: "f32[96, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_10, slice_scatter_default_20, 1, -1);  slice_scatter_default_20 = None
        add_tensor_3: "f32[96, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_tensor_2, select_scatter_default_3);  add_tensor_2 = select_scatter_default_3 = None
        slice_tensor_9: "f32[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_2, 1, 0, -1);  select_scatter_default_2 = None
        slice_tensor_10: "f32[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_9, 3, 256, 9223372036854775807);  slice_tensor_9 = None
        clone_default_10: "f32[96, 3, 256, 257]" = torch.ops.aten.clone.default(slice_tensor_10, memory_format = torch.contiguous_format);  slice_tensor_10 = None
        slice_scatter_default_21: "f32[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_12, clone_default_10, 3, 0, 257);  full_12 = clone_default_10 = None
        slice_scatter_default_22: "f32[96, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_10, slice_scatter_default_21, 2, 0, 256);  full_10 = slice_scatter_default_21 = None
        add_tensor_4: "f32[96, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_tensor_3, slice_scatter_default_22);  add_tensor_3 = slice_scatter_default_22 = None
        view_default_13: "f32[96, 3, 513, 512]" = torch.ops.aten.view.default(add_tensor_4, _shape_param_13);  add_tensor_4 = _shape_param_13 = None
        constant_pad_nd_default_1: "f32[96, 3, 512, 512]" = torch.ops.aten.constant_pad_nd.default(view_default_13, [0, 0, 0, -1]);  view_default_13 = None
        view_default_14: "f32[96, 3, 512, 512, 1]" = torch.ops.aten.view.default(constant_pad_nd_default_1, _shape_param_14);  constant_pad_nd_default_1 = _shape_param_14 = None
        permute_default_10: "f32[96, 3, 512, 1, 512]" = torch.ops.aten.permute.default(view_default_14, [0, 1, 2, 4, 3]);  view_default_14 = None
        view_default_15: "f32[288, 512, 512]" = torch.ops.aten.view.default(permute_default_10, _shape_param_15);  permute_default_10 = _shape_param_15 = None
        return view_default_15



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
