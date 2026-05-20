"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-1-5-linux.aws.a100_graph2
Pattern hash: 75e05cdc9884
Shape hash: 45005d83
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([1024, 768], bf16), T([36, 512, 512], bf16), T([1, 1024], bf16), T([1, 1024], b8), S([1024, 1, 768]), S([12, 3, 512, 1, 512]), S([12, 3, 512, 512]), S([12, 3, 512, 513]), S([1, 256, 12, 257]), S([1, 12, 1024, 513]), S([1, 12, 1024, 513]), S([1, 12, 1024, 513]), S([12, 4, 256, 513]), S([1, 256, 12, 257]), S([1, 12, 1024, 513]), S([1, 12, 1024, 513]), S([1, 12, 1024, 513]), S([1, 1024, 1]), S([1, 1024, 1]), S([1, 2, 512, 1]), S([1, 2, 512, 1]), S([1, 3, 512, 512]), S([1, 3, 512, 513]), S([1, 256, 1, 257]), S([1, 1, 1024, 513]), S([1, 1, 1024, 513]), S([1, 1, 1024, 513]), S([1, 4, 256, 513]), S([1, 256, 1, 257]), S([1, 1, 1024, 513]), S([1, 1, 1024, 513]), S([1, 1, 1024, 513]), S([1024, 1, 12, 64]), S([12, 4, 256, 513]), S([12, 1024, 64]), S([12, 4, -1]), S([12, 4, 256, 769]), S([48, 256, 768]), S([48, 768, 64]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_68: "bf16[1024, 768]", bmm_22: "bf16[36, 512, 512]", arg7_1: "bf16[1, 1024]", arg8_1: "b8[1, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34):
        # No stacktrace found for following nodes
        view_default: "bf16[1024, 1, 768]" = torch.ops.aten.view.default(addmm_68, _shape_param_0);  addmm_68 = _shape_param_0 = None
        view_default_1: "bf16[12, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_22, _shape_param_1);  bmm_22 = _shape_param_1 = None
        permute_default: "bf16[12, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_default_1, [0, 1, 2, 4, 3]);  view_default_1 = None
        view_default_2: "bf16[12, 3, 512, 512]" = torch.ops.aten.view.default(permute_default, _shape_param_2);  permute_default = _shape_param_2 = None
        constant_pad_nd_default: "bf16[12, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_default_2, [0, 0, 0, 1], 0.0);  view_default_2 = None
        view_default_3: "bf16[12, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_default, _shape_param_3);  constant_pad_nd_default = _shape_param_3 = None
        full_default: "bf16[12, 4, 256, 513]" = torch.ops.aten.full.default([12, 4, 256, 513], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor: "bf16[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_default_3, 2, 0, 256)
        slice_tensor_1: "bf16[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_tensor, 3, 0, 257);  slice_tensor = None
        slice_tensor_2: "bf16[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_default, 1, 0, -1)
        slice_tensor_3: "bf16[12, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_2, 3, 256, 9223372036854775807);  slice_tensor_2 = None
        copy_default: "bf16[12, 3, 256, 257]" = torch.ops.aten.copy.default(slice_tensor_3, slice_tensor_1);  slice_tensor_3 = slice_tensor_1 = None
        slice_tensor_4: "bf16[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_default, 1, 0, -1)
        slice_scatter_default: "bf16[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_4, copy_default, 3, 256, 9223372036854775807);  slice_tensor_4 = copy_default = None
        slice_scatter_default_1: "bf16[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default, slice_scatter_default, 1, 0, -1);  full_default = slice_scatter_default = None
        select_int: "bf16[12, 512, 513]" = torch.ops.aten.select.int(view_default_3, 1, -1)
        slice_tensor_5: "bf16[12, 256, 513]" = torch.ops.aten.slice.Tensor(select_int, 1, 256, 9223372036854775807);  select_int = None
        slice_tensor_6: "bf16[12, 256, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_5, 2, 0, 257);  slice_tensor_5 = None
        select_int_1: "bf16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_1, 1, -1)
        slice_tensor_7: "bf16[12, 256, 257]" = torch.ops.aten.slice.Tensor(select_int_1, 2, 256, 9223372036854775807);  select_int_1 = None
        copy_default_1: "bf16[12, 256, 257]" = torch.ops.aten.copy.default(slice_tensor_7, slice_tensor_6);  slice_tensor_7 = slice_tensor_6 = None
        select_int_2: "bf16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_1, 1, -1)
        slice_scatter_default_2: "bf16[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_2, copy_default_1, 2, 256, 9223372036854775807);  select_int_2 = copy_default_1 = None
        select_scatter_default: "bf16[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_1, slice_scatter_default_2, 1, -1);  slice_scatter_default_1 = slice_scatter_default_2 = None
        slice_tensor_8: "bf16[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_default_3, 2, -257, -1)
        slice_tensor_9: "bf16[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_tensor_8, 3, 257, 9223372036854775807);  slice_tensor_8 = None
        slice_tensor_10: "bf16[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default, 1, 1, 9223372036854775807)
        slice_tensor_11: "bf16[12, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_tensor_10, 3, 0, 256);  slice_tensor_10 = None
        copy_default_2: "bf16[12, 3, 256, 256]" = torch.ops.aten.copy.default(slice_tensor_11, slice_tensor_9);  slice_tensor_11 = slice_tensor_9 = None
        slice_tensor_12: "bf16[12, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default, 1, 1, 9223372036854775807)
        slice_scatter_default_3: "bf16[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_12, copy_default_2, 3, 0, 256);  slice_tensor_12 = copy_default_2 = None
        slice_scatter_default_4: "bf16[12, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default, slice_scatter_default_3, 1, 1, 9223372036854775807);  select_scatter_default = slice_scatter_default_3 = None
        select_int_3: "bf16[12, 512, 513]" = torch.ops.aten.select.int(view_default_3, 1, 0);  view_default_3 = None
        slice_tensor_13: "bf16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_3, 1, 0, 255);  select_int_3 = None
        slice_tensor_14: "bf16[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_tensor_13, 2, -255, 9223372036854775807);  slice_tensor_13 = None
        select_int_4: "bf16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_4, 1, 0)
        slice_tensor_15: "bf16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_4, 1, 1, 256);  select_int_4 = None
        slice_tensor_16: "bf16[12, 255, 255]" = torch.ops.aten.slice.Tensor(slice_tensor_15, 2, 1, 256);  slice_tensor_15 = None
        copy_default_3: "bf16[12, 255, 255]" = torch.ops.aten.copy.default(slice_tensor_16, slice_tensor_14);  slice_tensor_16 = slice_tensor_14 = None
        select_int_5: "bf16[12, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_4, 1, 0)
        slice_tensor_17: "bf16[12, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_5, 1, 1, 256)
        slice_scatter_default_5: "bf16[12, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_17, copy_default_3, 2, 1, 256);  slice_tensor_17 = copy_default_3 = None
        slice_scatter_default_6: "bf16[12, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_5, slice_scatter_default_5, 1, 1, 256);  select_int_5 = slice_scatter_default_5 = None
        select_scatter_default_1: "bf16[12, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_4, slice_scatter_default_6, 1, 0);  slice_scatter_default_4 = slice_scatter_default_6 = None
        full_default_1: "bf16[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_default: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_default, -2);  iota_default = None
        iota_default_1: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_1: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_default_1, -1);  iota_default_1 = None
        sub_tensor: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_default, unsqueeze_default_1);  unsqueeze_default = unsqueeze_default_1 = None
        le_scalar: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_tensor, 0);  sub_tensor = None
        full_default_2: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "bf16[256, 257]" = torch.ops.aten.where.self(le_scalar, full_default_1, full_default_2);  le_scalar = full_default_1 = full_default_2 = None
        rev_default: "bf16[256, 257]" = torch.ops.prims.rev.default(where_self, [0]);  where_self = None
        unsqueeze_default_2: "bf16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_default, 0);  rev_default = None
        unsqueeze_default_3: "bf16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 2);  unsqueeze_default_2 = None
        rev_default_1: "bf16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_default_3, [1, 3])
        expand_default: "bf16[1, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_default_3, _shape_param_4);  unsqueeze_default_3 = _shape_param_4 = None
        view_default_4: "bf16[1, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_default_1, _shape_param_5);  _shape_param_5 = None
        permute_default_1: "bf16[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_default_4, [0, 2, 1, 3]);  view_default_4 = None
        slice_tensor_18: "bf16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_default_1, 1, 0, 256);  permute_default_1 = None
        slice_tensor_19: "bf16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_18, 3, 0, 257);  slice_tensor_18 = None
        full_default_3: "bf16[1, 12, 256, 257]" = torch.ops.aten.full.default([1, 12, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_default_2: "bf16[1, 256, 12, 257]" = torch.ops.aten.permute.default(full_default_3, [0, 2, 1, 3]);  full_default_3 = None
        convert_element_type_default: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_default, torch.bool);  expand_default = None
        where_self_1: "bf16[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_default, permute_default_2, slice_tensor_19);  convert_element_type_default = permute_default_2 = slice_tensor_19 = None
        view_default_5: "bf16[1, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_default_1, _shape_param_6);  _shape_param_6 = None
        permute_default_3: "bf16[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_default_5, [0, 2, 1, 3]);  view_default_5 = None
        slice_tensor_20: "bf16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_default_3, 1, 0, 256);  permute_default_3 = None
        slice_tensor_21: "bf16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_20, 3, 0, 257);  slice_tensor_20 = None
        copy_default_4: "bf16[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_tensor_21, where_self_1);  slice_tensor_21 = where_self_1 = None
        view_default_6: "bf16[1, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_default_1, _shape_param_7);  select_scatter_default_1 = _shape_param_7 = None
        permute_default_4: "bf16[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_default_6, [0, 2, 1, 3]);  view_default_6 = None
        slice_tensor_22: "bf16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_default_4, 1, 0, 256)
        slice_scatter_default_7: "bf16[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_22, copy_default_4, 3, 0, 257);  slice_tensor_22 = copy_default_4 = None
        slice_scatter_default_8: "bf16[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_default_4, slice_scatter_default_7, 1, 0, 256);  permute_default_4 = slice_scatter_default_7 = None
        permute_default_5: "bf16[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_8, [0, 2, 1, 3]);  slice_scatter_default_8 = None
        view_default_7: "bf16[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_default_5, _shape_param_8);  permute_default_5 = _shape_param_8 = None
        expand_default_1: "bf16[1, 256, 12, 257]" = torch.ops.aten.expand.default(rev_default_1, _shape_param_9);  rev_default_1 = _shape_param_9 = None
        view_default_8: "bf16[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_default_7, _shape_param_10);  _shape_param_10 = None
        permute_default_6: "bf16[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_default_8, [0, 2, 1, 3]);  view_default_8 = None
        slice_tensor_23: "bf16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_default_6, 1, -256, 9223372036854775807);  permute_default_6 = None
        slice_tensor_24: "bf16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_23, 3, -257, 9223372036854775807);  slice_tensor_23 = None
        full_default_4: "bf16[1, 12, 256, 257]" = torch.ops.aten.full.default([1, 12, 256, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_default_7: "bf16[1, 256, 12, 257]" = torch.ops.aten.permute.default(full_default_4, [0, 2, 1, 3]);  full_default_4 = None
        convert_element_type_default_1: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_default_1, torch.bool);  expand_default_1 = None
        where_self_2: "bf16[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_default_1, permute_default_7, slice_tensor_24);  convert_element_type_default_1 = permute_default_7 = slice_tensor_24 = None
        view_default_9: "bf16[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_default_7, _shape_param_11);  _shape_param_11 = None
        permute_default_8: "bf16[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_default_9, [0, 2, 1, 3]);  view_default_9 = None
        slice_tensor_25: "bf16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_default_8, 1, -256, 9223372036854775807);  permute_default_8 = None
        slice_tensor_26: "bf16[1, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_25, 3, -257, 9223372036854775807);  slice_tensor_25 = None
        copy_default_5: "bf16[1, 256, 12, 257]" = torch.ops.aten.copy.default(slice_tensor_26, where_self_2);  slice_tensor_26 = where_self_2 = None
        view_default_10: "bf16[1, 12, 1024, 513]" = torch.ops.aten.view.default(view_default_7, _shape_param_12);  view_default_7 = _shape_param_12 = None
        permute_default_9: "bf16[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_default_10, [0, 2, 1, 3]);  view_default_10 = None
        slice_tensor_27: "bf16[1, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_default_9, 1, -256, 9223372036854775807)
        slice_scatter_default_9: "bf16[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_27, copy_default_5, 3, -257, 9223372036854775807);  slice_tensor_27 = copy_default_5 = None
        slice_scatter_default_10: "bf16[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_default_9, slice_scatter_default_9, 1, -256, 9223372036854775807);  permute_default_9 = slice_scatter_default_9 = None
        permute_default_10: "bf16[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_10, [0, 2, 1, 3]);  slice_scatter_default_10 = None
        ne_scalar: "b8[1, 1024]" = torch.ops.aten.ne.Scalar(arg7_1, 0);  arg7_1 = None
        unsqueeze_default_4: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(ne_scalar, 2);  ne_scalar = None
        unsqueeze_default_5: "b8[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        convert_element_type_default_2: "bf16[1, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_default_5, torch.bfloat16)
        full_default_5: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_3: "bf16[1, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_default_5, full_default_5, convert_element_type_default_2);  unsqueeze_default_5 = full_default_5 = convert_element_type_default_2 = None
        full_default_6: "bf16[1, 1024, 1, 1]" = torch.ops.aten.full.default([1, 1024, 1, 1], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_default_11: "bf16[1, 1, 1024, 1]" = torch.ops.aten.permute.default(full_default_6, [0, 2, 1, 3]);  full_default_6 = None
        view_default_11: "bf16[1, 1024, 1]" = torch.ops.aten.view.default(permute_default_11, _shape_param_13);  permute_default_11 = _shape_param_13 = None
        permute_default_12: "bf16[1, 1, 1024, 1]" = torch.ops.aten.permute.default(where_self_3, [0, 2, 1, 3]);  where_self_3 = None
        view_default_12: "bf16[1, 1024, 1]" = torch.ops.aten.view.default(permute_default_12, _shape_param_14);  permute_default_12 = _shape_param_14 = None
        view_default_13: "bf16[1, 2, 512, 1]" = torch.ops.aten.view.default(view_default_11, _shape_param_15);  view_default_11 = _shape_param_15 = None
        as_strided_default: "bf16[1, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_default_13, [1, 3, 512, 1], [1024, 256, 1, 1]);  view_default_13 = None
        view_default_14: "bf16[1, 2, 512, 1]" = torch.ops.aten.view.default(view_default_12, _shape_param_16);  view_default_12 = _shape_param_16 = None
        as_strided_default_1: "bf16[1, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_default_14, [1, 3, 512, 1], [1024, 256, 1, 1]);  view_default_14 = None
        unsqueeze_default_6: "bf16[1, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_default, 4);  as_strided_default = None
        permute_default_13: "bf16[1, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_default_6, [0, 1, 2, 4, 3]);  unsqueeze_default_6 = None
        unsqueeze_default_7: "bf16[1, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_default_1, 4);  as_strided_default_1 = None
        permute_default_14: "bf16[1, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_default_7, [0, 1, 4, 2, 3]);  unsqueeze_default_7 = None
        mul_tensor: "bf16[1, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_default_13, permute_default_14);  permute_default_13 = permute_default_14 = None
        view_default_15: "bf16[1, 3, 512, 512]" = torch.ops.aten.view.default(mul_tensor, _shape_param_17);  mul_tensor = _shape_param_17 = None
        constant_pad_nd_default_1: "bf16[1, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_default_15, [0, 0, 0, 1], 0.0);  view_default_15 = None
        view_default_16: "bf16[1, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_default_1, _shape_param_18);  constant_pad_nd_default_1 = _shape_param_18 = None
        full_default_7: "bf16[1, 4, 256, 513]" = torch.ops.aten.full.default([1, 4, 256, 513], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_28: "bf16[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_default_16, 2, 0, 256)
        slice_tensor_29: "bf16[1, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_28, 3, 0, 257);  slice_tensor_28 = None
        slice_tensor_30: "bf16[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_default_7, 1, 0, -1)
        slice_tensor_31: "bf16[1, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_30, 3, 256, 9223372036854775807);  slice_tensor_30 = None
        copy_default_6: "bf16[1, 3, 256, 257]" = torch.ops.aten.copy.default(slice_tensor_31, slice_tensor_29);  slice_tensor_31 = slice_tensor_29 = None
        slice_tensor_32: "bf16[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_default_7, 1, 0, -1)
        slice_scatter_default_11: "bf16[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_32, copy_default_6, 3, 256, 9223372036854775807);  slice_tensor_32 = copy_default_6 = None
        slice_scatter_default_12: "bf16[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_7, slice_scatter_default_11, 1, 0, -1);  full_default_7 = slice_scatter_default_11 = None
        select_int_6: "bf16[1, 512, 513]" = torch.ops.aten.select.int(view_default_16, 1, -1)
        slice_tensor_33: "bf16[1, 256, 513]" = torch.ops.aten.slice.Tensor(select_int_6, 1, 256, 9223372036854775807);  select_int_6 = None
        slice_tensor_34: "bf16[1, 256, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_33, 2, 0, 257);  slice_tensor_33 = None
        select_int_7: "bf16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_12, 1, -1)
        slice_tensor_35: "bf16[1, 256, 257]" = torch.ops.aten.slice.Tensor(select_int_7, 2, 256, 9223372036854775807);  select_int_7 = None
        copy_default_7: "bf16[1, 256, 257]" = torch.ops.aten.copy.default(slice_tensor_35, slice_tensor_34);  slice_tensor_35 = slice_tensor_34 = None
        select_int_8: "bf16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_12, 1, -1)
        slice_scatter_default_13: "bf16[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_8, copy_default_7, 2, 256, 9223372036854775807);  select_int_8 = copy_default_7 = None
        select_scatter_default_2: "bf16[1, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_12, slice_scatter_default_13, 1, -1);  slice_scatter_default_12 = slice_scatter_default_13 = None
        slice_tensor_36: "bf16[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_default_16, 2, -257, -1)
        slice_tensor_37: "bf16[1, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_tensor_36, 3, 257, 9223372036854775807);  slice_tensor_36 = None
        slice_tensor_38: "bf16[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_2, 1, 1, 9223372036854775807)
        slice_tensor_39: "bf16[1, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_tensor_38, 3, 0, 256);  slice_tensor_38 = None
        copy_default_8: "bf16[1, 3, 256, 256]" = torch.ops.aten.copy.default(slice_tensor_39, slice_tensor_37);  slice_tensor_39 = slice_tensor_37 = None
        slice_tensor_40: "bf16[1, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_2, 1, 1, 9223372036854775807)
        slice_scatter_default_14: "bf16[1, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_40, copy_default_8, 3, 0, 256);  slice_tensor_40 = copy_default_8 = None
        slice_scatter_default_15: "bf16[1, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default_2, slice_scatter_default_14, 1, 1, 9223372036854775807);  select_scatter_default_2 = slice_scatter_default_14 = None
        select_int_9: "bf16[1, 512, 513]" = torch.ops.aten.select.int(view_default_16, 1, 0);  view_default_16 = None
        slice_tensor_41: "bf16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_9, 1, 0, 255);  select_int_9 = None
        slice_tensor_42: "bf16[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_tensor_41, 2, -255, 9223372036854775807);  slice_tensor_41 = None
        select_int_10: "bf16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_15, 1, 0)
        slice_tensor_43: "bf16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_10, 1, 1, 256);  select_int_10 = None
        slice_tensor_44: "bf16[1, 255, 255]" = torch.ops.aten.slice.Tensor(slice_tensor_43, 2, 1, 256);  slice_tensor_43 = None
        copy_default_9: "bf16[1, 255, 255]" = torch.ops.aten.copy.default(slice_tensor_44, slice_tensor_42);  slice_tensor_44 = slice_tensor_42 = None
        select_int_11: "bf16[1, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_15, 1, 0)
        slice_tensor_45: "bf16[1, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_11, 1, 1, 256)
        slice_scatter_default_16: "bf16[1, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_45, copy_default_9, 2, 1, 256);  slice_tensor_45 = copy_default_9 = None
        slice_scatter_default_17: "bf16[1, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_11, slice_scatter_default_16, 1, 1, 256);  select_int_11 = slice_scatter_default_16 = None
        select_scatter_default_3: "bf16[1, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_15, slice_scatter_default_17, 1, 0);  slice_scatter_default_15 = slice_scatter_default_17 = None
        full_default_8: "bf16[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_default_2: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_8: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_default_2, -2);  iota_default_2 = None
        iota_default_3: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_9: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_default_3, -1);  iota_default_3 = None
        sub_tensor_1: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_default_8, unsqueeze_default_9);  unsqueeze_default_8 = unsqueeze_default_9 = None
        le_scalar_1: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_tensor_1, 0);  sub_tensor_1 = None
        full_default_9: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_4: "bf16[256, 257]" = torch.ops.aten.where.self(le_scalar_1, full_default_8, full_default_9);  le_scalar_1 = full_default_8 = full_default_9 = None
        rev_default_2: "bf16[256, 257]" = torch.ops.prims.rev.default(where_self_4, [0]);  where_self_4 = None
        unsqueeze_default_10: "bf16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_default_2, 0);  rev_default_2 = None
        unsqueeze_default_11: "bf16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        rev_default_3: "bf16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_default_11, [1, 3])
        expand_default_2: "bf16[1, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_default_11, _shape_param_19);  unsqueeze_default_11 = _shape_param_19 = None
        view_default_17: "bf16[1, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_default_3, _shape_param_20);  _shape_param_20 = None
        permute_default_15: "bf16[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_default_17, [0, 2, 1, 3]);  view_default_17 = None
        slice_tensor_46: "bf16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_default_15, 1, 0, 256);  permute_default_15 = None
        slice_tensor_47: "bf16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_46, 3, 0, 257);  slice_tensor_46 = None
        full_default_10: "bf16[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_default_3: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_default_2, torch.bool);  expand_default_2 = None
        where_self_5: "bf16[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_default_3, full_default_10, slice_tensor_47);  convert_element_type_default_3 = full_default_10 = slice_tensor_47 = None
        view_default_18: "bf16[1, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_default_3, _shape_param_21);  _shape_param_21 = None
        permute_default_16: "bf16[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_default_18, [0, 2, 1, 3]);  view_default_18 = None
        slice_tensor_48: "bf16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_default_16, 1, 0, 256);  permute_default_16 = None
        slice_tensor_49: "bf16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_48, 3, 0, 257);  slice_tensor_48 = None
        copy_default_10: "bf16[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_tensor_49, where_self_5);  slice_tensor_49 = where_self_5 = None
        view_default_19: "bf16[1, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_default_3, _shape_param_22);  select_scatter_default_3 = _shape_param_22 = None
        permute_default_17: "bf16[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_default_19, [0, 2, 1, 3]);  view_default_19 = None
        slice_tensor_50: "bf16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_default_17, 1, 0, 256)
        slice_scatter_default_18: "bf16[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_50, copy_default_10, 3, 0, 257);  slice_tensor_50 = copy_default_10 = None
        slice_scatter_default_19: "bf16[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_default_17, slice_scatter_default_18, 1, 0, 256);  permute_default_17 = slice_scatter_default_18 = None
        permute_default_18: "bf16[1, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_19, [0, 2, 1, 3]);  slice_scatter_default_19 = None
        view_default_20: "bf16[1, 4, 256, 513]" = torch.ops.aten.view.default(permute_default_18, _shape_param_23);  permute_default_18 = _shape_param_23 = None
        expand_default_3: "bf16[1, 256, 1, 257]" = torch.ops.aten.expand.default(rev_default_3, _shape_param_24);  rev_default_3 = _shape_param_24 = None
        view_default_21: "bf16[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_default_20, _shape_param_25);  _shape_param_25 = None
        permute_default_19: "bf16[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_default_21, [0, 2, 1, 3]);  view_default_21 = None
        slice_tensor_51: "bf16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_default_19, 1, -256, 9223372036854775807);  permute_default_19 = None
        slice_tensor_52: "bf16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_51, 3, -257, 9223372036854775807);  slice_tensor_51 = None
        full_default_11: "bf16[1, 256, 1, 257]" = torch.ops.aten.full.default([1, 256, 1, 257], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_default_4: "b8[1, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_default_3, torch.bool);  expand_default_3 = None
        where_self_6: "bf16[1, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_default_4, full_default_11, slice_tensor_52);  convert_element_type_default_4 = full_default_11 = slice_tensor_52 = None
        view_default_22: "bf16[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_default_20, _shape_param_26);  _shape_param_26 = None
        permute_default_20: "bf16[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_default_22, [0, 2, 1, 3]);  view_default_22 = None
        slice_tensor_53: "bf16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_default_20, 1, -256, 9223372036854775807);  permute_default_20 = None
        slice_tensor_54: "bf16[1, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_tensor_53, 3, -257, 9223372036854775807);  slice_tensor_53 = None
        copy_default_11: "bf16[1, 256, 1, 257]" = torch.ops.aten.copy.default(slice_tensor_54, where_self_6);  slice_tensor_54 = where_self_6 = None
        view_default_23: "bf16[1, 1, 1024, 513]" = torch.ops.aten.view.default(view_default_20, _shape_param_27);  view_default_20 = _shape_param_27 = None
        permute_default_21: "bf16[1, 1024, 1, 513]" = torch.ops.aten.permute.default(view_default_23, [0, 2, 1, 3]);  view_default_23 = None
        slice_tensor_55: "bf16[1, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_default_21, 1, -256, 9223372036854775807)
        slice_scatter_default_20: "bf16[1, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_55, copy_default_11, 3, -257, 9223372036854775807);  slice_tensor_55 = copy_default_11 = None
        slice_scatter_default_21: "bf16[1, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_default_21, slice_scatter_default_20, 1, -256, 9223372036854775807);  permute_default_21 = slice_scatter_default_20 = None
        permute_default_22: "bf16[1, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_21, [0, 2, 1, 3]);  slice_scatter_default_21 = None
        permute_default_23: "bf16[1, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_default_10, [0, 2, 1, 3]);  permute_default_10 = None
        permute_default_24: "bf16[1, 1024, 1, 513]" = torch.ops.aten.permute.default(permute_default_22, [0, 2, 1, 3]);  permute_default_22 = None
        add_tensor: "bf16[1, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_default_23, permute_default_24);  permute_default_23 = permute_default_24 = None
        permute_default_25: "bf16[1, 12, 1024, 513]" = torch.ops.aten.permute.default(add_tensor, [0, 2, 1, 3]);  add_tensor = None
        permute_default_26: "bf16[1, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_default_25, [0, 2, 1, 3]);  permute_default_25 = None
        convert_element_type_default_5: "f32[1, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(permute_default_26, torch.float32);  permute_default_26 = None
        clone_default: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type_default_5, memory_format = torch.contiguous_format);  convert_element_type_default_5 = None
        amax_default: "f32[1, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_default, [-1], True)
        sub_tensor_2: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_default, amax_default);  clone_default = amax_default = None
        exp_default: "f32[1, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_tensor_2);  sub_tensor_2 = None
        sum_dim_int_list: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[1, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        unsqueeze_default_12: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, 2);  arg8_1 = None
        unsqueeze_default_13: "b8[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 3);  unsqueeze_default_12 = None
        full_default_12: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_7: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_default_13, full_default_12, div_tensor);  unsqueeze_default_13 = full_default_12 = div_tensor = None
        convert_element_type_default_6: "bf16[1, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(where_self_7, torch.bfloat16);  where_self_7 = None
        view_default_24: "bf16[1024, 1, 12, 64]" = torch.ops.aten.view.default(view_default, _shape_param_28);  view_default = _shape_param_28 = None
        permute_default_27: "bf16[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_default_24, [1, 0, 2, 3]);  view_default_24 = None
        permute_default_28: "bf16[1, 12, 1024, 513]" = torch.ops.aten.permute.default(convert_element_type_default_6, [0, 2, 1, 3]);  convert_element_type_default_6 = None
        view_default_25: "bf16[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_default_28, _shape_param_29);  permute_default_28 = _shape_param_29 = None
        permute_default_29: "bf16[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_default_27, [0, 2, 1, 3]);  permute_default_27 = None
        view_default_26: "bf16[12, 1024, 64]" = torch.ops.aten.view.default(permute_default_29, _shape_param_30);  permute_default_29 = _shape_param_30 = None
        constant_pad_nd_default_2: "bf16[12, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_default_26, [0, 0, 256, 256], -1.0);  view_default_26 = None
        as_strided_default_2: "bf16[12, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_default_2, [12, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_default_2 = None
        constant_pad_nd_default_3: "bf16[12, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_default_25, [0, 257], 0.0);  view_default_25 = None
        view_default_27: "bf16[12, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_default_3, _shape_param_31);  constant_pad_nd_default_3 = _shape_param_31 = None
        slice_tensor_56: "bf16[12, 4, 196864]" = torch.ops.aten.slice.Tensor(view_default_27, 2, 0, -256);  view_default_27 = None
        view_default_28: "bf16[12, 4, 256, 769]" = torch.ops.aten.view.default(slice_tensor_56, _shape_param_32);  slice_tensor_56 = _shape_param_32 = None
        slice_tensor_57: "bf16[12, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_default_28, 3, 0, -1);  view_default_28 = None
        unsqueeze_default_14: "bf16[12, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_57, 4);  slice_tensor_57 = None
        unsqueeze_default_15: "bf16[12, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_default_2, 4);  as_strided_default_2 = None
        view_default_29: "bf16[48, 256, 768]" = torch.ops.aten.view.default(unsqueeze_default_14, _shape_param_33);  unsqueeze_default_14 = _shape_param_33 = None
        clone_default_1: "bf16[12, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_default_15, memory_format = torch.contiguous_format);  unsqueeze_default_15 = None
        view_default_30: "bf16[48, 768, 64]" = torch.ops.aten.view.default(clone_default_1, _shape_param_34);  clone_default_1 = _shape_param_34 = None
        return (view_default_29, view_default_30)


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
