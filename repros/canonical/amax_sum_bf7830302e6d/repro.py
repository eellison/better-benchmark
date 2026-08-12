"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_train
Pattern hash: bf7830302e6d
Shape hash: 50ff733e
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[288, 512, 512]", arg1_1: "f32[8, 1024]", arg2_1: "b8[8, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34):
        # No stacktrace found for following nodes
        view: "bf16[96, 3, 512, 1, 512]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        permute: "bf16[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view, [0, 1, 2, 4, 3]);  view = None
        view_1: "bf16[96, 3, 512, 512]" = torch.ops.aten.view.default(permute, _shape_param_1);  permute = _shape_param_1 = None
        constant_pad_nd: "bf16[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_1, [0, 0, 0, 1], 0.0);  view_1 = None
        view_2: "bf16[96, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd, _shape_param_2);  constant_pad_nd = _shape_param_2 = None
        full: "bf16[96, 4, 256, 513]" = torch.ops.aten.full.default(_shape_param_3, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_3 = None
        slice_1: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_2, 2, 0, 256)
        slice_2: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1, 3, 0, 257);  slice_1 = None
        slice_3: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full, 1, 0, -1)
        slice_4: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_3, 3, 256, 9223372036854775807)
        copy: "bf16[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_4, slice_2);  slice_2 = None
        slice_scatter: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_3, copy, 3, 256, 9223372036854775807);  copy = None
        slice_scatter_1: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter, 1, 0, -1);  slice_scatter = None
        select: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_2, 1, -1)
        slice_5: "bf16[96, 256, 513]" = torch.ops.aten.slice.Tensor(select, 1, 256, 9223372036854775807);  select = None
        slice_6: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_5, 2, 0, 257);  slice_5 = None
        select_1: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_1, 1, -1)
        slice_7: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_1, 2, 256, 9223372036854775807)
        copy_1: "bf16[96, 256, 257]" = torch.ops.aten.copy.default(slice_7, slice_6);  slice_7 = slice_6 = None
        slice_scatter_2: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_1, copy_1, 2, 256, 9223372036854775807);  select_1 = copy_1 = None
        select_scatter: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_1, slice_scatter_2, 1, -1);  slice_scatter_1 = slice_scatter_2 = None
        slice_8: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_2, 2, -257, -1)
        slice_9: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_8, 3, 257, 9223372036854775807);  slice_8 = None
        slice_10: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter, 1, 1, 9223372036854775807)
        slice_11: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_10, 3, 0, 256)
        copy_2: "bf16[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_11, slice_9);  slice_11 = slice_9 = None
        slice_scatter_3: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_10, copy_2, 3, 0, 256);  slice_10 = copy_2 = None
        slice_scatter_4: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter, slice_scatter_3, 1, 1, 9223372036854775807);  select_scatter = slice_scatter_3 = None
        select_2: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_2, 1, 0);  view_2 = None
        slice_12: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_2, 1, 0, 255);  select_2 = None
        slice_13: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_12, 2, -255, 9223372036854775807);  slice_12 = None
        select_3: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_4, 1, 0)
        slice_14: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_3, 1, 1, 256)
        slice_15: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_14, 2, 1, 256)
        copy_3: "bf16[96, 255, 255]" = torch.ops.aten.copy.default(slice_15, slice_13);  slice_15 = slice_13 = None
        slice_scatter_5: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_14, copy_3, 2, 1, 256);  slice_14 = copy_3 = None
        slice_scatter_6: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_3, slice_scatter_5, 1, 1, 256);  select_3 = slice_scatter_5 = None
        select_scatter_1: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_4, slice_scatter_6, 1, 0);  slice_scatter_4 = slice_scatter_6 = None
        full_1: "bf16[256, 257]" = torch.ops.aten.full.default(_shape_param_4, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_4 = None
        iota: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota, -2);  iota = None
        iota_1: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_1: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
        sub: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
        le: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub, 0);  sub = None
        scalar_tensor: "bf16[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where: "bf16[256, 257]" = torch.ops.aten.where.self(le, full_1, scalar_tensor);  le = full_1 = scalar_tensor = None
        rev: "bf16[256, 257]" = torch.ops.prims.rev.default(where, [0]);  where = None
        unsqueeze_2: "bf16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev, 0);  rev = None
        unsqueeze_3: "bf16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 2);  unsqueeze_2 = None
        rev_1: "bf16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_3, [1, 3])
        expand: "bf16[8, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_3, _shape_param_5);  _shape_param_5 = None
        view_3: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_1, _shape_param_6);  select_scatter_1 = _shape_param_6 = None
        permute_1: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        slice_16: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_1, 1, 0, 256)
        slice_17: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_16, 3, 0, 257)
        full_2: "bf16[8, 12, 256, 257]" = torch.ops.aten.full.default(_shape_param_7, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_7 = None
        permute_2: "bf16[8, 256, 12, 257]" = torch.ops.aten.permute.default(full_2, [0, 2, 1, 3]);  full_2 = None
        convert_element_type: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand, torch.bool);  expand = None
        where_1: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, permute_2, slice_17)
        copy_4: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_17, where_1);  slice_17 = where_1 = None
        slice_scatter_7: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_16, copy_4, 3, 0, 257);  slice_16 = copy_4 = None
        slice_scatter_8: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_1, slice_scatter_7, 1, 0, 256);  permute_1 = slice_scatter_7 = None
        permute_3: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_8, [0, 2, 1, 3]);  slice_scatter_8 = None
        view_4: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_3, _shape_param_8);  permute_3 = _shape_param_8 = None
        expand_1: "bf16[8, 256, 12, 257]" = torch.ops.aten.expand.default(rev_1, _shape_param_9);  _shape_param_9 = None
        view_5: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_4, _shape_param_10);  view_4 = _shape_param_10 = None
        permute_4: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        slice_18: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_4, 1, -256, 9223372036854775807)
        slice_19: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_18, 3, -257, 9223372036854775807)
        convert_element_type_1: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_1, torch.bool);  expand_1 = None
        where_2: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, permute_2, slice_19)
        copy_5: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_19, where_2);  slice_19 = where_2 = None
        slice_scatter_9: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_18, copy_5, 3, -257, 9223372036854775807);  slice_18 = copy_5 = None
        slice_scatter_10: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_4, slice_scatter_9, 1, -256, 9223372036854775807);  permute_4 = slice_scatter_9 = None
        permute_5: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_10, [0, 2, 1, 3])
        ne: "b8[8, 1024]" = torch.ops.aten.ne.Scalar(arg1_1, 0);  arg1_1 = None
        unsqueeze_4: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(ne, 2);  ne = None
        unsqueeze_5: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        convert_element_type_2: "bf16[8, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_5, torch.bfloat16)
        full_3: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "bf16[8, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_5, full_3, convert_element_type_2);  unsqueeze_5 = full_3 = convert_element_type_2 = None
        full_4: "bf16[8, 1024, 1, 1]" = torch.ops.aten.full.default(_shape_param_11, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_11 = None
        permute_6: "bf16[8, 1, 1024, 1]" = torch.ops.aten.permute.default(full_4, [0, 2, 1, 3]);  full_4 = None
        view_6: "bf16[8, 1024, 1]" = torch.ops.aten.view.default(permute_6, _shape_param_12);  permute_6 = _shape_param_12 = None
        permute_7: "bf16[8, 1, 1024, 1]" = torch.ops.aten.permute.default(where_3, [0, 2, 1, 3]);  where_3 = None
        view_7: "bf16[8, 1024, 1]" = torch.ops.aten.view.default(permute_7, _shape_param_13);  permute_7 = _shape_param_13 = None
        view_8: "bf16[8, 2, 512, 1]" = torch.ops.aten.view.default(view_6, _shape_param_14);  view_6 = _shape_param_14 = None
        as_strided: "bf16[8, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_8, _shape_param_15, _shape_param_16);  view_8 = _shape_param_15 = _shape_param_16 = None
        view_9: "bf16[8, 2, 512, 1]" = torch.ops.aten.view.default(view_7, _shape_param_17);  view_7 = _shape_param_17 = None
        as_strided_1: "bf16[8, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_9, _shape_param_18, _shape_param_19);  view_9 = _shape_param_18 = _shape_param_19 = None
        unsqueeze_6: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided, 4);  as_strided = None
        permute_8: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_6, [0, 1, 2, 4, 3]);  unsqueeze_6 = None
        unsqueeze_7: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_1, 4);  as_strided_1 = None
        permute_9: "bf16[8, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_7, [0, 1, 4, 2, 3]);  unsqueeze_7 = None
        mul: "bf16[8, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_8, permute_9);  permute_8 = permute_9 = None
        view_10: "bf16[8, 3, 512, 512]" = torch.ops.aten.view.default(mul, _shape_param_20);  mul = _shape_param_20 = None
        constant_pad_nd_1: "bf16[8, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_10, [0, 0, 0, 1], 0.0);  view_10 = None
        view_11: "bf16[8, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_1, _shape_param_21);  constant_pad_nd_1 = _shape_param_21 = None
        full_5: "bf16[8, 4, 256, 513]" = torch.ops.aten.full.default(_shape_param_22, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_22 = None
        slice_20: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_11, 2, 0, 256)
        slice_21: "bf16[8, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_20, 3, 0, 257);  slice_20 = None
        slice_22: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_5, 1, 0, -1)
        slice_23: "bf16[8, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_22, 3, 256, 9223372036854775807)
        copy_6: "bf16[8, 3, 256, 257]" = torch.ops.aten.copy.default(slice_23, slice_21);  slice_23 = slice_21 = None
        slice_scatter_11: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_22, copy_6, 3, 256, 9223372036854775807);  slice_22 = copy_6 = None
        slice_scatter_12: "bf16[8, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_5, slice_scatter_11, 1, 0, -1);  full_5 = slice_scatter_11 = None
        select_4: "bf16[8, 512, 513]" = torch.ops.aten.select.int(view_11, 1, -1)
        slice_24: "bf16[8, 256, 513]" = torch.ops.aten.slice.Tensor(select_4, 1, 256, 9223372036854775807);  select_4 = None
        slice_25: "bf16[8, 256, 257]" = torch.ops.aten.slice.Tensor(slice_24, 2, 0, 257);  slice_24 = None
        select_5: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_12, 1, -1)
        slice_26: "bf16[8, 256, 257]" = torch.ops.aten.slice.Tensor(select_5, 2, 256, 9223372036854775807)
        copy_7: "bf16[8, 256, 257]" = torch.ops.aten.copy.default(slice_26, slice_25);  slice_26 = slice_25 = None
        slice_scatter_13: "bf16[8, 256, 513]" = torch.ops.aten.slice_scatter.default(select_5, copy_7, 2, 256, 9223372036854775807);  select_5 = copy_7 = None
        select_scatter_2: "bf16[8, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_12, slice_scatter_13, 1, -1);  slice_scatter_12 = slice_scatter_13 = None
        slice_27: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_11, 2, -257, -1)
        slice_28: "bf16[8, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_27, 3, 257, 9223372036854775807);  slice_27 = None
        slice_29: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_2, 1, 1, 9223372036854775807)
        slice_30: "bf16[8, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_29, 3, 0, 256)
        copy_8: "bf16[8, 3, 256, 256]" = torch.ops.aten.copy.default(slice_30, slice_28);  slice_30 = slice_28 = None
        slice_scatter_14: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_29, copy_8, 3, 0, 256);  slice_29 = copy_8 = None
        slice_scatter_15: "bf16[8, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_2, slice_scatter_14, 1, 1, 9223372036854775807);  select_scatter_2 = slice_scatter_14 = None
        select_6: "bf16[8, 512, 513]" = torch.ops.aten.select.int(view_11, 1, 0);  view_11 = None
        slice_31: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_6, 1, 0, 255);  select_6 = None
        slice_32: "bf16[8, 255, 255]" = torch.ops.aten.slice.Tensor(slice_31, 2, -255, 9223372036854775807);  slice_31 = None
        select_7: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_15, 1, 0)
        slice_33: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_7, 1, 1, 256)
        slice_34: "bf16[8, 255, 255]" = torch.ops.aten.slice.Tensor(slice_33, 2, 1, 256)
        copy_9: "bf16[8, 255, 255]" = torch.ops.aten.copy.default(slice_34, slice_32);  slice_34 = slice_32 = None
        slice_scatter_16: "bf16[8, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_33, copy_9, 2, 1, 256);  slice_33 = copy_9 = None
        slice_scatter_17: "bf16[8, 256, 513]" = torch.ops.aten.slice_scatter.default(select_7, slice_scatter_16, 1, 1, 256);  select_7 = slice_scatter_16 = None
        select_scatter_3: "bf16[8, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_15, slice_scatter_17, 1, 0);  slice_scatter_15 = slice_scatter_17 = None
        expand_2: "bf16[8, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_3, _shape_param_23);  _shape_param_23 = None
        view_12: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_3, _shape_param_24);  select_scatter_3 = _shape_param_24 = None
        permute_10: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
        slice_35: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_10, 1, 0, 256)
        slice_36: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_35, 3, 0, 257)
        full_6: "bf16[8, 256, 1, 257]" = torch.ops.aten.full.default(_shape_param_25, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_25 = None
        convert_element_type_3: "b8[8, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_2, torch.bool);  expand_2 = None
        where_4: "bf16[8, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_3, full_6, slice_36);  convert_element_type_3 = None
        copy_10: "bf16[8, 256, 1, 257]" = torch.ops.aten.copy.default(slice_36, where_4);  slice_36 = where_4 = None
        slice_scatter_18: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_35, copy_10, 3, 0, 257);  slice_35 = copy_10 = None
        slice_scatter_19: "bf16[8, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_10, slice_scatter_18, 1, 0, 256);  permute_10 = slice_scatter_18 = None
        permute_11: "bf16[8, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_19, [0, 2, 1, 3]);  slice_scatter_19 = None
        view_13: "bf16[8, 4, 256, 513]" = torch.ops.aten.view.default(permute_11, _shape_param_26);  permute_11 = _shape_param_26 = None
        expand_3: "bf16[8, 256, 1, 257]" = torch.ops.aten.expand.default(rev_1, _shape_param_27);  _shape_param_27 = None
        view_14: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_13, _shape_param_28);  view_13 = _shape_param_28 = None
        permute_12: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        slice_37: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_12, 1, -256, 9223372036854775807)
        slice_38: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_37, 3, -257, 9223372036854775807)
        convert_element_type_4: "b8[8, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_3, torch.bool);  expand_3 = None
        where_5: "bf16[8, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_4, full_6, slice_38);  convert_element_type_4 = full_6 = None
        copy_11: "bf16[8, 256, 1, 257]" = torch.ops.aten.copy.default(slice_38, where_5);  slice_38 = where_5 = None
        slice_scatter_20: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_37, copy_11, 3, -257, 9223372036854775807);  slice_37 = copy_11 = None
        slice_scatter_21: "bf16[8, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_12, slice_scatter_20, 1, -256, 9223372036854775807);  permute_12 = slice_scatter_20 = None
        permute_13: "bf16[8, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_21, [0, 2, 1, 3])
        permute_14: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_5, [0, 2, 1, 3]);  permute_5 = None
        permute_15: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(permute_13, [0, 2, 1, 3]);  permute_13 = None
        add: "bf16[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_14, permute_15);  permute_14 = None
        permute_16: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add, [0, 2, 1, 3]);  add = None
        permute_17: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_16, [0, 2, 1, 3]);  permute_16 = None
        convert_element_type_5: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(permute_17, torch.float32);  permute_17 = None
        clone: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type_5, memory_format = torch.contiguous_format);  convert_element_type_5 = None
        amax: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone, [-1], True)
        sub_1: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone, amax);  clone = None
        exp: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        unsqueeze_8: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, 2);  arg2_1 = None
        unsqueeze_9: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 3);  unsqueeze_8 = None
        full_7: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_9, full_7, div);  div = None
        convert_element_type_6: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(where_6, torch.bfloat16);  where_6 = None
        inductor_seeds: "i64[36]" = torch.ops.prims.inductor_seeds.default(36, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0)
        inductor_random: "f32[8, 1024, 12, 513]" = torch.ops.prims.inductor_random.default(_shape_param_29, inductor_lookup_seed, 'rand');  _shape_param_29 = inductor_lookup_seed = None
        convert_element_type_7: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[8, 1024, 12, 513]" = torch.ops.aten.gt.Scalar(convert_element_type_7, 0.1);  convert_element_type_7 = None
        mul_1: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(gt, convert_element_type_6);  convert_element_type_6 = None
        mul_2: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(mul_1, 1.1111111111111112);  mul_1 = None
        permute_18: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(mul_2, [0, 2, 1, 3]);  mul_2 = None
        clone_1: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_15: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_1, _shape_param_30);  clone_1 = _shape_param_30 = None
        constant_pad_nd_2: "bf16[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_15, _shape_param_31, 0.0);  view_15 = _shape_param_31 = None
        view_16: "bf16[96, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_2, _shape_param_32);  constant_pad_nd_2 = _shape_param_32 = None
        slice_39: "bf16[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_16, 2, 0, -256);  view_16 = None
        view_17: "bf16[96, 4, 256, 769]" = torch.ops.aten.view.default(slice_39, _shape_param_33);  slice_39 = _shape_param_33 = None
        slice_40: "bf16[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_17, 3, 0, -1);  view_17 = None
        unsqueeze_10: "bf16[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_40, 4);  slice_40 = None
        view_18: "bf16[384, 256, 768]" = torch.ops.aten.view.default(unsqueeze_10, _shape_param_34);  unsqueeze_10 = _shape_param_34 = None
        permute_19: "bf16[384, 768, 256]" = torch.ops.aten.permute.default(view_18, [0, 2, 1])
        return (full, slice_3, slice_4, unsqueeze_3, rev_1, permute_2, convert_element_type, convert_element_type_1, slice_scatter_10, slice_scatter_21, permute_15, amax, sum_1, unsqueeze_9, full_7, inductor_seeds, gt, view_18, permute_19)



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
