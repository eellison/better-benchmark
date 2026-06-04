"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Longformer_train_005
Pattern hash: 5a4992885bd6
Shape hash: 5edba0a3
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([96, 256, 768], f32), T([2, 1024, 12, 513], b8), T([2, 1024], b8), T([2, 1024, 12, 513], f32), T([1, 256, 1, 257], f32), T([1, 256, 1, 257], f32), S([24, 4, 256, 768, 1]), S([24, 4, 196864]), S([24, 4, 256, 770]), S([2, 12, 1024, 513]), S([24, 4, 256, 513]), S([2, 12, 1024, 513]), S([24, 4, 256, 513]), S([2, 12, 1024, 513]), S([24, 4, 256, 513]), S([2, 256, 12, 257]), S([24, 4, 256, 513]), S([2, 12, 1024, 513]), S([24, 4, 256, 513]), S([2, 256, 12, 257]), S([24, 4, 256, 513]), S([24, 3, 513, 512]), S([24, 3, 512, 512, 1]), S([72, 512, 512]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_1: "f32[96, 256, 768]", arg222_1: "b8[2, 1024, 12, 513]", arg3_1: "b8[2, 1024]", arg221_1: "f32[2, 1024, 12, 513]", arg99_1: "f32[1, 256, 1, 257]", arg98_1: "f32[1, 256, 1, 257]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17):
        # No stacktrace found for following nodes
        view_default: "f32[24, 4, 256, 768, 1]" = torch.ops.aten.view.default(bmm_1, _shape_param_0);  bmm_1 = _shape_param_0 = None
        squeeze_dim: "f32[24, 4, 256, 768]" = torch.ops.aten.squeeze.dim(view_default, 4);  view_default = None
        full_default: "f32[24, 4, 256, 769]" = torch.ops.aten.full.default([24, 4, 256, 769], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default: "f32[24, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default, squeeze_dim, 3, 0, -1);  full_default = squeeze_dim = None
        view_default_1: "f32[24, 4, 196864]" = torch.ops.aten.view.default(slice_scatter_default, _shape_param_1);  slice_scatter_default = _shape_param_1 = None
        full_default_1: "f32[24, 4, 197120]" = torch.ops.aten.full.default([24, 4, 197120], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default_1: "f32[24, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_1, view_default_1, 2, 0, -256);  full_default_1 = view_default_1 = None
        view_default_2: "f32[24, 4, 256, 770]" = torch.ops.aten.view.default(slice_scatter_default_1, _shape_param_2);  slice_scatter_default_1 = _shape_param_2 = None
        constant_pad_nd_default: "f32[24, 4, 256, 513]" = torch.ops.aten.constant_pad_nd.default(view_default_2, [0, -257]);  view_default_2 = None
        view_default_3: "f32[2, 12, 1024, 513]" = torch.ops.aten.view.default(constant_pad_nd_default, _shape_param_3);  constant_pad_nd_default = _shape_param_3 = None
        permute_default: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_default_3, [0, 2, 1, 3]);  view_default_3 = None
        convert_element_type_default: "f32[2, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(arg222_1, torch.float32);  arg222_1 = None
        mul_tensor: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(permute_default, mul_tensor);  permute_default = mul_tensor = None
        clone_default: "f32[2, 1024, 12, 513]" = torch.ops.aten.clone.default(mul_tensor_1, memory_format = torch.contiguous_format);  mul_tensor_1 = None
        full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        unsqueeze_default: "b8[2, 1024, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, 2);  arg3_1 = None
        unsqueeze_default_1: "b8[2, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 3);  unsqueeze_default = None
        where_self: "f32[2, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default_2, clone_default);  unsqueeze_default_1 = clone_default = None
        mul_tensor_2: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(where_self, arg221_1);  where_self = None
        sum_dim_int_list: "f32[2, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [-1], True)
        neg_default: "f32[2, 1024, 12, 513]" = torch.ops.aten.neg.default(arg221_1);  arg221_1 = None
        fma_default: "f32[2, 1024, 12, 513]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_2);  neg_default = sum_dim_int_list = mul_tensor_2 = None
        permute_default_1: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(fma_default, [0, 2, 1, 3]);  fma_default = None
        clone_default_1: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_4: "f32[24, 4, 256, 513]" = torch.ops.aten.view.default(clone_default_1, _shape_param_4);  clone_default_1 = _shape_param_4 = None
        view_default_5: "f32[2, 12, 1024, 513]" = torch.ops.aten.view.default(view_default_4, _shape_param_5);  view_default_4 = _shape_param_5 = None
        permute_default_2: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_default_5, [0, 2, 1, 3]);  view_default_5 = None
        clone_default_2: "f32[2, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format)
        copy_default: "f32[2, 1024, 12, 513]" = torch.ops.aten.copy.default(permute_default_2, clone_default_2);  permute_default_2 = clone_default_2 = None
        permute_default_3: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(copy_default, [0, 2, 1, 3]);  copy_default = None
        view_default_6: "f32[24, 4, 256, 513]" = torch.ops.aten.view.default(permute_default_3, _shape_param_6);  permute_default_3 = _shape_param_6 = None
        view_default_7: "f32[2, 12, 1024, 513]" = torch.ops.aten.view.default(view_default_6, _shape_param_7);  view_default_6 = _shape_param_7 = None
        permute_default_4: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_default_7, [0, 2, 1, 3]);  view_default_7 = None
        slice_tensor: "f32[2, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_default_4, 1, -256, 9223372036854775807)
        slice_tensor_1: "f32[2, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_tensor, 3, -257, 9223372036854775807)
        clone_default_3: "f32[2, 256, 12, 257]" = torch.ops.aten.clone.default(slice_tensor_1, memory_format = torch.contiguous_format)
        full_default_3: "f32[2, 256, 12, 257]" = torch.ops.aten.full.default([2, 256, 12, 257], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        copy_default_1: "f32[2, 256, 12, 257]" = torch.ops.aten.copy.default(slice_tensor_1, full_default_3);  slice_tensor_1 = None
        slice_scatter_default_2: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor, copy_default_1, 3, -257, 9223372036854775807);  slice_tensor = copy_default_1 = None
        slice_scatter_default_3: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_default_4, slice_scatter_default_2, 1, -256, 9223372036854775807);  permute_default_4 = slice_scatter_default_2 = None
        permute_default_5: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_3, [0, 2, 1, 3]);  slice_scatter_default_3 = None
        view_default_8: "f32[24, 4, 256, 513]" = torch.ops.aten.view.default(permute_default_5, _shape_param_8);  permute_default_5 = _shape_param_8 = None
        expand_default: "f32[2, 256, 12, 257]" = torch.ops.aten.expand.default(arg99_1, _shape_param_9);  arg99_1 = _shape_param_9 = None
        convert_element_type_default_1: "b8[2, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_default, torch.bool);  expand_default = None
        where_self_1: "f32[2, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_default_1, full_default_2, clone_default_3);  convert_element_type_default_1 = clone_default_3 = None
        full_default_4: "f32[2, 256, 12, 513]" = torch.ops.aten.full.default([2, 256, 12, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default_4: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_4, where_self_1, 3, -257, 9223372036854775807);  where_self_1 = None
        full_default_5: "f32[2, 1024, 12, 513]" = torch.ops.aten.full.default([2, 1024, 12, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default_5: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_5, slice_scatter_default_4, 1, -256, 9223372036854775807);  slice_scatter_default_4 = None
        permute_default_6: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_5, [0, 2, 1, 3]);  slice_scatter_default_5 = None
        clone_default_4: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_default_6, memory_format = torch.contiguous_format);  permute_default_6 = None
        view_default_9: "f32[24, 4, 256, 513]" = torch.ops.aten.view.default(clone_default_4, _shape_param_10);  clone_default_4 = _shape_param_10 = None
        add_tensor: "f32[24, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_default_8, view_default_9);  view_default_8 = view_default_9 = None
        view_default_10: "f32[2, 12, 1024, 513]" = torch.ops.aten.view.default(add_tensor, _shape_param_11);  add_tensor = _shape_param_11 = None
        permute_default_7: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_default_10, [0, 2, 1, 3]);  view_default_10 = None
        slice_tensor_2: "f32[2, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_default_7, 1, 0, 256)
        slice_tensor_3: "f32[2, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_2, 3, 0, 257)
        clone_default_5: "f32[2, 256, 12, 257]" = torch.ops.aten.clone.default(slice_tensor_3, memory_format = torch.contiguous_format)
        copy_default_2: "f32[2, 256, 12, 257]" = torch.ops.aten.copy.default(slice_tensor_3, full_default_3);  slice_tensor_3 = full_default_3 = None
        slice_scatter_default_6: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_2, copy_default_2, 3, 0, 257);  slice_tensor_2 = copy_default_2 = None
        slice_scatter_default_7: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_default_7, slice_scatter_default_6, 1, 0, 256);  permute_default_7 = slice_scatter_default_6 = None
        permute_default_8: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_7, [0, 2, 1, 3]);  slice_scatter_default_7 = None
        view_default_11: "f32[24, 4, 256, 513]" = torch.ops.aten.view.default(permute_default_8, _shape_param_12);  permute_default_8 = _shape_param_12 = None
        expand_default_1: "f32[2, 256, 12, 257]" = torch.ops.aten.expand.default(arg98_1, _shape_param_13);  arg98_1 = _shape_param_13 = None
        convert_element_type_default_2: "b8[2, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_default_1, torch.bool);  expand_default_1 = None
        where_self_2: "f32[2, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_default_2, full_default_2, clone_default_5);  convert_element_type_default_2 = full_default_2 = clone_default_5 = None
        slice_scatter_default_8: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_4, where_self_2, 3, 0, 257);  full_default_4 = where_self_2 = None
        slice_scatter_default_9: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_5, slice_scatter_default_8, 1, 0, 256);  full_default_5 = slice_scatter_default_8 = None
        permute_default_9: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_9, [0, 2, 1, 3]);  slice_scatter_default_9 = None
        clone_default_6: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_default_9, memory_format = torch.contiguous_format);  permute_default_9 = None
        view_default_12: "f32[24, 4, 256, 513]" = torch.ops.aten.view.default(clone_default_6, _shape_param_14);  clone_default_6 = _shape_param_14 = None
        add_tensor_1: "f32[24, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_default_11, view_default_12);  view_default_11 = view_default_12 = None
        select_int: "f32[24, 256, 513]" = torch.ops.aten.select.int(add_tensor_1, 1, 0)
        slice_tensor_4: "f32[24, 255, 513]" = torch.ops.aten.slice.Tensor(select_int, 1, 1, 256)
        slice_tensor_5: "f32[24, 255, 255]" = torch.ops.aten.slice.Tensor(slice_tensor_4, 2, 1, 256)
        clone_default_7: "f32[24, 255, 255]" = torch.ops.aten.clone.default(slice_tensor_5, memory_format = torch.contiguous_format)
        full_default_6: "f32[24, 255, 255]" = torch.ops.aten.full.default([24, 255, 255], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        copy_default_3: "f32[24, 255, 255]" = torch.ops.aten.copy.default(slice_tensor_5, full_default_6);  slice_tensor_5 = full_default_6 = None
        slice_scatter_default_10: "f32[24, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_4, copy_default_3, 2, 1, 256);  slice_tensor_4 = copy_default_3 = None
        slice_scatter_default_11: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int, slice_scatter_default_10, 1, 1, 256);  select_int = slice_scatter_default_10 = None
        select_scatter_default: "f32[24, 4, 256, 513]" = torch.ops.aten.select_scatter.default(add_tensor_1, slice_scatter_default_11, 1, 0);  add_tensor_1 = slice_scatter_default_11 = None
        full_default_7: "f32[24, 255, 513]" = torch.ops.aten.full.default([24, 255, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default_12: "f32[24, 255, 513]" = torch.ops.aten.slice_scatter.default(full_default_7, clone_default_7, 2, -255, 9223372036854775807);  full_default_7 = clone_default_7 = None
        full_default_8: "f32[24, 512, 513]" = torch.ops.aten.full.default([24, 512, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default_13: "f32[24, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_8, slice_scatter_default_12, 1, 0, 255);  slice_scatter_default_12 = None
        full_default_9: "f32[24, 3, 512, 513]" = torch.ops.aten.full.default([24, 3, 512, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_default_1: "f32[24, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_9, slice_scatter_default_13, 1, 0);  slice_scatter_default_13 = None
        slice_tensor_6: "f32[24, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default, 1, 1, 9223372036854775807)
        slice_tensor_7: "f32[24, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_tensor_6, 3, 0, 256)
        clone_default_8: "f32[24, 3, 256, 256]" = torch.ops.aten.clone.default(slice_tensor_7, memory_format = torch.contiguous_format)
        full_default_10: "f32[24, 3, 256, 256]" = torch.ops.aten.full.default([24, 3, 256, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        copy_default_4: "f32[24, 3, 256, 256]" = torch.ops.aten.copy.default(slice_tensor_7, full_default_10);  slice_tensor_7 = full_default_10 = None
        slice_scatter_default_14: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_6, copy_default_4, 3, 0, 256);  slice_tensor_6 = copy_default_4 = None
        slice_scatter_default_15: "f32[24, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default, slice_scatter_default_14, 1, 1, 9223372036854775807);  select_scatter_default = slice_scatter_default_14 = None
        full_default_11: "f32[24, 3, 256, 513]" = torch.ops.aten.full.default([24, 3, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default_16: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_11, clone_default_8, 3, 257, 9223372036854775807);  clone_default_8 = None
        slice_scatter_default_17: "f32[24, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_9, slice_scatter_default_16, 2, -257, -1);  slice_scatter_default_16 = None
        add_tensor_2: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(select_scatter_default_1, slice_scatter_default_17);  select_scatter_default_1 = slice_scatter_default_17 = None
        select_int_1: "f32[24, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_15, 1, -1)
        slice_tensor_8: "f32[24, 256, 257]" = torch.ops.aten.slice.Tensor(select_int_1, 2, 256, 9223372036854775807)
        clone_default_9: "f32[24, 256, 257]" = torch.ops.aten.clone.default(slice_tensor_8, memory_format = torch.contiguous_format)
        full_default_12: "f32[24, 256, 257]" = torch.ops.aten.full.default([24, 256, 257], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        copy_default_5: "f32[24, 256, 257]" = torch.ops.aten.copy.default(slice_tensor_8, full_default_12);  slice_tensor_8 = full_default_12 = None
        slice_scatter_default_18: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_1, copy_default_5, 2, 256, 9223372036854775807);  select_int_1 = copy_default_5 = None
        select_scatter_default_2: "f32[24, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_15, slice_scatter_default_18, 1, -1);  slice_scatter_default_15 = slice_scatter_default_18 = None
        full_default_13: "f32[24, 256, 513]" = torch.ops.aten.full.default([24, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default_19: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_13, clone_default_9, 2, 0, 257);  full_default_13 = clone_default_9 = None
        slice_scatter_default_20: "f32[24, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_8, slice_scatter_default_19, 1, 256, 9223372036854775807);  full_default_8 = slice_scatter_default_19 = None
        select_scatter_default_3: "f32[24, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_9, slice_scatter_default_20, 1, -1);  slice_scatter_default_20 = None
        add_tensor_3: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_tensor_2, select_scatter_default_3);  add_tensor_2 = select_scatter_default_3 = None
        slice_tensor_9: "f32[24, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_2, 1, 0, -1);  select_scatter_default_2 = None
        slice_tensor_10: "f32[24, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_9, 3, 256, 9223372036854775807);  slice_tensor_9 = None
        clone_default_10: "f32[24, 3, 256, 257]" = torch.ops.aten.clone.default(slice_tensor_10, memory_format = torch.contiguous_format);  slice_tensor_10 = None
        slice_scatter_default_21: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_11, clone_default_10, 3, 0, 257);  full_default_11 = clone_default_10 = None
        slice_scatter_default_22: "f32[24, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_9, slice_scatter_default_21, 2, 0, 256);  full_default_9 = slice_scatter_default_21 = None
        add_tensor_4: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_tensor_3, slice_scatter_default_22);  add_tensor_3 = slice_scatter_default_22 = None
        view_default_13: "f32[24, 3, 513, 512]" = torch.ops.aten.view.default(add_tensor_4, _shape_param_15);  add_tensor_4 = _shape_param_15 = None
        constant_pad_nd_default_1: "f32[24, 3, 512, 512]" = torch.ops.aten.constant_pad_nd.default(view_default_13, [0, 0, 0, -1]);  view_default_13 = None
        view_default_14: "f32[24, 3, 512, 512, 1]" = torch.ops.aten.view.default(constant_pad_nd_default_1, _shape_param_16);  constant_pad_nd_default_1 = _shape_param_16 = None
        permute_default_10: "f32[24, 3, 512, 1, 512]" = torch.ops.aten.permute.default(view_default_14, [0, 1, 2, 4, 3]);  view_default_14 = None
        view_default_15: "f32[72, 512, 512]" = torch.ops.aten.view.default(permute_default_10, _shape_param_17);  permute_default_10 = _shape_param_17 = None
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
