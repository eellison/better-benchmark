"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_infer
Pattern hash: 528a3c274a41
Shape hash: 79c25467
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
    def forward(self, arg0_1: "b8[8, 1024]", arg1_1: "bf16[288, 512, 512]", arg2_1: "bf16[8, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44):
        # No stacktrace found for following nodes
        unsqueeze: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(arg0_1, 2);  arg0_1 = None
        unsqueeze_1: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 3);  unsqueeze = None
        full: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_1: "bf16[96, 4, 256, 513]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        slice_1: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_1, 1, 0, -1)
        slice_2: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_1, 1, 0, -1)
        slice_3: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2, 3, 256, 9223372036854775807);  slice_2 = None
        view: "bf16[96, 3, 512, 1, 512]" = torch.ops.aten.view.default(arg1_1, _shape_param_1);  arg1_1 = _shape_param_1 = None
        permute: "bf16[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view, [0, 1, 2, 4, 3]);  view = None
        view_1: "bf16[96, 3, 512, 512]" = torch.ops.aten.view.default(permute, _shape_param_2);  permute = _shape_param_2 = None
        constant_pad_nd: "bf16[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_1, [0, 0, 0, 1], 0.0);  view_1 = None
        view_2: "bf16[96, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd, _shape_param_3);  constant_pad_nd = _shape_param_3 = None
        slice_4: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_2, 2, 0, 256)
        slice_5: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_4, 3, 0, 257);  slice_4 = None
        copy: "bf16[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_3, slice_5);  slice_3 = slice_5 = None
        slice_scatter: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1, copy, 3, 256, 9223372036854775807);  slice_1 = copy = None
        slice_scatter_1: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_1, slice_scatter, 1, 0, -1);  full_1 = slice_scatter = None
        select: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_1, 1, -1)
        select_1: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_1, 1, -1)
        slice_6: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_1, 2, 256, 9223372036854775807);  select_1 = None
        select_2: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_2, 1, -1)
        slice_7: "bf16[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_2, 1, 256, 9223372036854775807);  select_2 = None
        slice_8: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_7, 2, 0, 257);  slice_7 = None
        copy_1: "bf16[96, 256, 257]" = torch.ops.aten.copy.default(slice_6, slice_8);  slice_6 = slice_8 = None
        slice_scatter_2: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select, copy_1, 2, 256, 9223372036854775807);  select = copy_1 = None
        select_scatter: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_1, slice_scatter_2, 1, -1);  slice_scatter_1 = slice_scatter_2 = None
        slice_9: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter, 1, 1, 9223372036854775807)
        slice_10: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter, 1, 1, 9223372036854775807)
        slice_11: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_10, 3, 0, 256);  slice_10 = None
        slice_12: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_2, 2, -257, -1)
        slice_13: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_12, 3, 257, 9223372036854775807);  slice_12 = None
        copy_2: "bf16[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_11, slice_13);  slice_11 = slice_13 = None
        slice_scatter_3: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_9, copy_2, 3, 0, 256);  slice_9 = copy_2 = None
        slice_scatter_4: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter, slice_scatter_3, 1, 1, 9223372036854775807);  select_scatter = slice_scatter_3 = None
        select_3: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_4, 1, 0)
        slice_14: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_3, 1, 1, 256)
        select_4: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_4, 1, 0)
        slice_15: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_4, 1, 1, 256);  select_4 = None
        slice_16: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_15, 2, 1, 256);  slice_15 = None
        select_5: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_2, 1, 0);  view_2 = None
        slice_17: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_5, 1, 0, 255);  select_5 = None
        slice_18: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_17, 2, -255, 9223372036854775807);  slice_17 = None
        copy_3: "bf16[96, 255, 255]" = torch.ops.aten.copy.default(slice_16, slice_18);  slice_16 = slice_18 = None
        slice_scatter_5: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_14, copy_3, 2, 1, 256);  slice_14 = copy_3 = None
        slice_scatter_6: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_3, slice_scatter_5, 1, 1, 256);  select_3 = slice_scatter_5 = None
        select_scatter_1: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_4, slice_scatter_6, 1, 0);  slice_scatter_4 = slice_scatter_6 = None
        view_3: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_1, _shape_param_4);  _shape_param_4 = None
        permute_1: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        slice_19: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_1, 1, 0, 256)
        view_4: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_1, _shape_param_5);  _shape_param_5 = None
        permute_2: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        slice_20: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_2, 1, 0, 256);  permute_2 = None
        slice_21: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_20, 3, 0, 257);  slice_20 = None
        iota: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_2: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota, -2);  iota = None
        iota_1: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_3: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
        sub: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_2, unsqueeze_3);  unsqueeze_2 = unsqueeze_3 = None
        le: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub, 0);  sub = None
        full_2: "bf16[256, 257]" = torch.ops.aten.full.default(_shape_param_6, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_6 = None
        scalar_tensor: "bf16[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where: "bf16[256, 257]" = torch.ops.aten.where.self(le, full_2, scalar_tensor);  le = full_2 = scalar_tensor = None
        rev: "bf16[256, 257]" = torch.ops.prims.rev.default(where, [0]);  where = None
        unsqueeze_4: "bf16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev, 0);  rev = None
        unsqueeze_5: "bf16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None
        expand: "bf16[8, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_5, _shape_param_7);  _shape_param_7 = None
        convert_element_type: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand, torch.bool);  expand = None
        full_3: "bf16[8, 12, 256, 257]" = torch.ops.aten.full.default(_shape_param_8, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_8 = None
        permute_3: "bf16[8, 256, 12, 257]" = torch.ops.aten.permute.default(full_3, [0, 2, 1, 3]);  full_3 = None
        view_5: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_1, _shape_param_9);  select_scatter_1 = _shape_param_9 = None
        permute_4: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        slice_22: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_4, 1, 0, 256);  permute_4 = None
        slice_23: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_22, 3, 0, 257);  slice_22 = None
        where_1: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, permute_3, slice_23);  convert_element_type = permute_3 = slice_23 = None
        copy_4: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_21, where_1);  slice_21 = where_1 = None
        slice_scatter_7: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_19, copy_4, 3, 0, 257);  slice_19 = copy_4 = None
        slice_scatter_8: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_1, slice_scatter_7, 1, 0, 256);  permute_1 = slice_scatter_7 = None
        permute_5: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_8, [0, 2, 1, 3]);  slice_scatter_8 = None
        view_6: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_5, _shape_param_10);  permute_5 = _shape_param_10 = None
        view_7: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_6, _shape_param_11);  _shape_param_11 = None
        permute_6: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
        slice_24: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_6, 1, -256, 9223372036854775807)
        view_8: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_6, _shape_param_12);  _shape_param_12 = None
        permute_7: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        slice_25: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_7, 1, -256, 9223372036854775807);  permute_7 = None
        slice_26: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_25, 3, -257, 9223372036854775807);  slice_25 = None
        rev_1: "bf16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_5, [1, 3]);  unsqueeze_5 = None
        expand_1: "bf16[8, 256, 12, 257]" = torch.ops.aten.expand.default(rev_1, _shape_param_13);  rev_1 = _shape_param_13 = None
        convert_element_type_1: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_1, torch.bool);  expand_1 = None
        full_4: "bf16[8, 12, 256, 257]" = torch.ops.aten.full.default(_shape_param_14, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_14 = None
        permute_8: "bf16[8, 256, 12, 257]" = torch.ops.aten.permute.default(full_4, [0, 2, 1, 3]);  full_4 = None
        view_9: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_6, _shape_param_15);  view_6 = _shape_param_15 = None
        permute_9: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
        slice_27: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_9, 1, -256, 9223372036854775807);  permute_9 = None
        slice_28: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_27, 3, -257, 9223372036854775807);  slice_27 = None
        where_2: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, permute_8, slice_28);  convert_element_type_1 = permute_8 = slice_28 = None
        copy_5: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_26, where_2);  slice_26 = where_2 = None
        slice_scatter_9: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_24, copy_5, 3, -257, 9223372036854775807);  slice_24 = copy_5 = None
        slice_scatter_10: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_6, slice_scatter_9, 1, -256, 9223372036854775807);  permute_6 = slice_scatter_9 = None
        permute_10: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_10, [0, 2, 1, 3]);  slice_scatter_10 = None
        permute_11: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_10, [0, 2, 1, 3]);  permute_10 = None
        full_5: "bf16[8, 4, 256, 513]" = torch.ops.aten.full.default(_shape_param_16, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_16 = None
        slice_29: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_5, 1, 0, -1)
        slice_30: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_5, 1, 0, -1)
        slice_31: "bf16[8, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_30, 3, 256, 9223372036854775807);  slice_30 = None
        full_6: "bf16[8, 1024, 1, 1]" = torch.ops.aten.full.default(_shape_param_17, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_17 = None
        permute_12: "bf16[8, 1, 1024, 1]" = torch.ops.aten.permute.default(full_6, [0, 2, 1, 3]);  full_6 = None
        view_10: "bf16[8, 1024, 1]" = torch.ops.aten.view.default(permute_12, _shape_param_18);  permute_12 = _shape_param_18 = None
        view_11: "bf16[8, 2, 512, 1]" = torch.ops.aten.view.default(view_10, _shape_param_19);  view_10 = _shape_param_19 = None
        as_strided: "bf16[8, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_11, _shape_param_20, _shape_param_21);  view_11 = _shape_param_20 = _shape_param_21 = None
        unsqueeze_6: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided, 4);  as_strided = None
        permute_13: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_6, [0, 1, 2, 4, 3]);  unsqueeze_6 = None
        full_7: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        ne: "b8[8, 1024]" = torch.ops.aten.ne.Scalar(arg2_1, 0);  arg2_1 = None
        unsqueeze_7: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(ne, 2);  ne = None
        unsqueeze_8: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
        convert_element_type_2: "bf16[8, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_8, torch.bfloat16)
        where_3: "bf16[8, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_8, full_7, convert_element_type_2);  unsqueeze_8 = full_7 = convert_element_type_2 = None
        permute_14: "bf16[8, 1, 1024, 1]" = torch.ops.aten.permute.default(where_3, [0, 2, 1, 3]);  where_3 = None
        view_12: "bf16[8, 1024, 1]" = torch.ops.aten.view.default(permute_14, _shape_param_22);  permute_14 = _shape_param_22 = None
        view_13: "bf16[8, 2, 512, 1]" = torch.ops.aten.view.default(view_12, _shape_param_23);  view_12 = _shape_param_23 = None
        as_strided_1: "bf16[8, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_13, _shape_param_24, _shape_param_25);  view_13 = _shape_param_24 = _shape_param_25 = None
        unsqueeze_9: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_1, 4);  as_strided_1 = None
        permute_15: "bf16[8, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_9, [0, 1, 4, 2, 3]);  unsqueeze_9 = None
        mul: "bf16[8, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_13, permute_15);  permute_13 = permute_15 = None
        view_14: "bf16[8, 3, 512, 512]" = torch.ops.aten.view.default(mul, _shape_param_26);  mul = _shape_param_26 = None
        constant_pad_nd_1: "bf16[8, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_14, [0, 0, 0, 1], 0.0);  view_14 = None
        view_15: "bf16[8, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_1, _shape_param_27);  constant_pad_nd_1 = _shape_param_27 = None
        slice_32: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_15, 2, 0, 256)
        slice_33: "bf16[8, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_32, 3, 0, 257);  slice_32 = None
        copy_6: "bf16[8, 3, 256, 257]" = torch.ops.aten.copy.default(slice_31, slice_33);  slice_31 = slice_33 = None
        slice_scatter_11: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_29, copy_6, 3, 256, 9223372036854775807);  slice_29 = copy_6 = None
        slice_scatter_12: "bf16[8, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_5, slice_scatter_11, 1, 0, -1);  full_5 = slice_scatter_11 = None
        select_6: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_12, 1, -1)
        select_7: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_12, 1, -1)
        slice_34: "bf16[8, 256, 257]" = torch.ops.aten.slice.Tensor(select_7, 2, 256, 9223372036854775807);  select_7 = None
        select_8: "bf16[8, 512, 513]" = torch.ops.aten.select.int(view_15, 1, -1)
        slice_35: "bf16[8, 256, 513]" = torch.ops.aten.slice.Tensor(select_8, 1, 256, 9223372036854775807);  select_8 = None
        slice_36: "bf16[8, 256, 257]" = torch.ops.aten.slice.Tensor(slice_35, 2, 0, 257);  slice_35 = None
        copy_7: "bf16[8, 256, 257]" = torch.ops.aten.copy.default(slice_34, slice_36);  slice_34 = slice_36 = None
        slice_scatter_13: "bf16[8, 256, 513]" = torch.ops.aten.slice_scatter.default(select_6, copy_7, 2, 256, 9223372036854775807);  select_6 = copy_7 = None
        select_scatter_2: "bf16[8, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_12, slice_scatter_13, 1, -1);  slice_scatter_12 = slice_scatter_13 = None
        slice_37: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_2, 1, 1, 9223372036854775807)
        slice_38: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_2, 1, 1, 9223372036854775807)
        slice_39: "bf16[8, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_38, 3, 0, 256);  slice_38 = None
        slice_40: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_15, 2, -257, -1)
        slice_41: "bf16[8, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_40, 3, 257, 9223372036854775807);  slice_40 = None
        copy_8: "bf16[8, 3, 256, 256]" = torch.ops.aten.copy.default(slice_39, slice_41);  slice_39 = slice_41 = None
        slice_scatter_14: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_37, copy_8, 3, 0, 256);  slice_37 = copy_8 = None
        slice_scatter_15: "bf16[8, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_2, slice_scatter_14, 1, 1, 9223372036854775807);  select_scatter_2 = slice_scatter_14 = None
        select_9: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_15, 1, 0)
        slice_42: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_9, 1, 1, 256)
        select_10: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_15, 1, 0)
        slice_43: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_10, 1, 1, 256);  select_10 = None
        slice_44: "bf16[8, 255, 255]" = torch.ops.aten.slice.Tensor(slice_43, 2, 1, 256);  slice_43 = None
        select_11: "bf16[8, 512, 513]" = torch.ops.aten.select.int(view_15, 1, 0);  view_15 = None
        slice_45: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_11, 1, 0, 255);  select_11 = None
        slice_46: "bf16[8, 255, 255]" = torch.ops.aten.slice.Tensor(slice_45, 2, -255, 9223372036854775807);  slice_45 = None
        copy_9: "bf16[8, 255, 255]" = torch.ops.aten.copy.default(slice_44, slice_46);  slice_44 = slice_46 = None
        slice_scatter_16: "bf16[8, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_42, copy_9, 2, 1, 256);  slice_42 = copy_9 = None
        slice_scatter_17: "bf16[8, 256, 513]" = torch.ops.aten.slice_scatter.default(select_9, slice_scatter_16, 1, 1, 256);  select_9 = slice_scatter_16 = None
        select_scatter_3: "bf16[8, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_15, slice_scatter_17, 1, 0);  slice_scatter_15 = slice_scatter_17 = None
        view_16: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_3, _shape_param_28);  _shape_param_28 = None
        permute_16: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        slice_47: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_16, 1, 0, 256)
        view_17: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_3, _shape_param_29);  _shape_param_29 = None
        permute_17: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None
        slice_48: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_17, 1, 0, 256);  permute_17 = None
        slice_49: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_48, 3, 0, 257);  slice_48 = None
        iota_2: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_10: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_2, -2);  iota_2 = None
        iota_3: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_11: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_3, -1);  iota_3 = None
        sub_1: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_10, unsqueeze_11);  unsqueeze_10 = unsqueeze_11 = None
        le_1: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_1, 0);  sub_1 = None
        full_8: "bf16[256, 257]" = torch.ops.aten.full.default(_shape_param_30, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_30 = None
        scalar_tensor_1: "bf16[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_4: "bf16[256, 257]" = torch.ops.aten.where.self(le_1, full_8, scalar_tensor_1);  le_1 = full_8 = scalar_tensor_1 = None
        rev_2: "bf16[256, 257]" = torch.ops.prims.rev.default(where_4, [0]);  where_4 = None
        unsqueeze_12: "bf16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_2, 0);  rev_2 = None
        unsqueeze_13: "bf16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, 2);  unsqueeze_12 = None
        expand_2: "bf16[8, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_13, _shape_param_31);  _shape_param_31 = None
        convert_element_type_3: "b8[8, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_2, torch.bool);  expand_2 = None
        full_9: "bf16[8, 256, 1, 257]" = torch.ops.aten.full.default(_shape_param_32, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_32 = None
        view_18: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_3, _shape_param_33);  select_scatter_3 = _shape_param_33 = None
        permute_18: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        slice_50: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_18, 1, 0, 256);  permute_18 = None
        slice_51: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_50, 3, 0, 257);  slice_50 = None
        where_5: "bf16[8, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_3, full_9, slice_51);  convert_element_type_3 = full_9 = slice_51 = None
        copy_10: "bf16[8, 256, 1, 257]" = torch.ops.aten.copy.default(slice_49, where_5);  slice_49 = where_5 = None
        slice_scatter_18: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_47, copy_10, 3, 0, 257);  slice_47 = copy_10 = None
        slice_scatter_19: "bf16[8, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_16, slice_scatter_18, 1, 0, 256);  permute_16 = slice_scatter_18 = None
        permute_19: "bf16[8, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_19, [0, 2, 1, 3]);  slice_scatter_19 = None
        view_19: "bf16[8, 4, 256, 513]" = torch.ops.aten.view.default(permute_19, _shape_param_34);  permute_19 = _shape_param_34 = None
        view_20: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_19, _shape_param_35);  _shape_param_35 = None
        permute_20: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_20, [0, 2, 1, 3]);  view_20 = None
        slice_52: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_20, 1, -256, 9223372036854775807)
        view_21: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_19, _shape_param_36);  _shape_param_36 = None
        permute_21: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        slice_53: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_21, 1, -256, 9223372036854775807);  permute_21 = None
        slice_54: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_53, 3, -257, 9223372036854775807);  slice_53 = None
        rev_3: "bf16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_13, [1, 3]);  unsqueeze_13 = None
        expand_3: "bf16[8, 256, 1, 257]" = torch.ops.aten.expand.default(rev_3, _shape_param_37);  rev_3 = _shape_param_37 = None
        convert_element_type_4: "b8[8, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_3, torch.bool);  expand_3 = None
        full_10: "bf16[8, 256, 1, 257]" = torch.ops.aten.full.default(_shape_param_38, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_38 = None
        view_22: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_19, _shape_param_39);  view_19 = _shape_param_39 = None
        permute_22: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
        slice_55: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_22, 1, -256, 9223372036854775807);  permute_22 = None
        slice_56: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_55, 3, -257, 9223372036854775807);  slice_55 = None
        where_6: "bf16[8, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_4, full_10, slice_56);  convert_element_type_4 = full_10 = slice_56 = None
        copy_11: "bf16[8, 256, 1, 257]" = torch.ops.aten.copy.default(slice_54, where_6);  slice_54 = where_6 = None
        slice_scatter_20: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_52, copy_11, 3, -257, 9223372036854775807);  slice_52 = copy_11 = None
        slice_scatter_21: "bf16[8, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_20, slice_scatter_20, 1, -256, 9223372036854775807);  permute_20 = slice_scatter_20 = None
        permute_23: "bf16[8, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_21, [0, 2, 1, 3]);  slice_scatter_21 = None
        permute_24: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(permute_23, [0, 2, 1, 3]);  permute_23 = None
        add: "bf16[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_11, permute_24);  permute_11 = permute_24 = None
        permute_25: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add, [0, 2, 1, 3]);  add = None
        permute_26: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_25, [0, 2, 1, 3]);  permute_25 = None
        convert_element_type_5: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(permute_26, torch.float32);  permute_26 = None
        clone: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type_5, memory_format = torch.contiguous_format);  convert_element_type_5 = None
        amax: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone, [-1], True)
        sub_2: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone, amax);  clone = amax = None
        exp: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_1: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        where_7: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_1, full, div);  unsqueeze_1 = full = div = None
        convert_element_type_6: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(where_7, torch.bfloat16);  where_7 = None
        permute_27: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(convert_element_type_6, [0, 2, 1, 3]);  convert_element_type_6 = None
        clone_1: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
        view_23: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_1, _shape_param_40);  clone_1 = _shape_param_40 = None
        constant_pad_nd_2: "bf16[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_23, _shape_param_41, 0.0);  view_23 = _shape_param_41 = None
        view_24: "bf16[96, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_2, _shape_param_42);  constant_pad_nd_2 = _shape_param_42 = None
        slice_57: "bf16[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_24, 2, 0, -256);  view_24 = None
        view_25: "bf16[96, 4, 256, 769]" = torch.ops.aten.view.default(slice_57, _shape_param_43);  slice_57 = _shape_param_43 = None
        slice_58: "bf16[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_25, 3, 0, -1);  view_25 = None
        unsqueeze_14: "bf16[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_58, 4);  slice_58 = None
        view_26: "bf16[384, 256, 768]" = torch.ops.aten.view.default(unsqueeze_14, _shape_param_44);  unsqueeze_14 = _shape_param_44 = None
        return view_26



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
