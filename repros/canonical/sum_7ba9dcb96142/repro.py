"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_train
Pattern hash: 7ba9dcb96142
Shape hash: 39dafa96
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
    def forward(self, arg0_1: "bf16[384, 256, 768]", arg1_1: "b8[8, 1024, 12, 513]", arg2_1: "b8[8, 1024]", arg3_1: "bf16[8, 1024, 12, 513]", arg4_1: "f32[8, 1024, 12, 1]", arg5_1: "f32[8, 1024, 12, 1]", arg6_1: "bf16[1, 256, 1, 257]", arg7_1: "bf16[1, 256, 1, 257]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31):
        # No stacktrace found for following nodes
        view: "bf16[96, 4, 256, 768, 1]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        squeeze: "bf16[96, 4, 256, 768]" = torch.ops.aten.squeeze.dim(view, 4);  view = None
        full: "bf16[96, 4, 256, 769]" = torch.ops.aten.full.default(_shape_param_1, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        slice_scatter: "bf16[96, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full, squeeze, 3, 0, -1);  squeeze = None
        view_1: "bf16[96, 4, 196864]" = torch.ops.aten.view.default(slice_scatter, _shape_param_2);  slice_scatter = _shape_param_2 = None
        full_1: "bf16[96, 4, 197120]" = torch.ops.aten.full.default(_shape_param_3, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_3 = None
        slice_scatter_1: "bf16[96, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_1, view_1, 2, 0, -256);  view_1 = None
        view_2: "bf16[96, 4, 256, 770]" = torch.ops.aten.view.default(slice_scatter_1, _shape_param_4);  slice_scatter_1 = _shape_param_4 = None
        constant_pad_nd: "bf16[96, 4, 256, 513]" = torch.ops.aten.constant_pad_nd.default(view_2, _shape_param_5);  view_2 = _shape_param_5 = None
        view_3: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(constant_pad_nd, _shape_param_6);  constant_pad_nd = _shape_param_6 = None
        permute: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        convert_element_type: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        mul: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
        mul_1: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(permute, mul);  permute = mul = None
        clone: "bf16[8, 1024, 12, 513]" = torch.ops.aten.clone.default(mul_1, memory_format = torch.contiguous_format);  mul_1 = None
        convert_element_type_1: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(clone, torch.float32);  clone = None
        unsqueeze: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, 2);  arg2_1 = None
        unsqueeze_1: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 3);  unsqueeze = None
        full_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_1, full_2, convert_element_type_1);  convert_element_type_1 = None
        convert_element_type_2: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        clone_1: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type_2, memory_format = torch.contiguous_format);  convert_element_type_2 = None
        sub: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_1, arg4_1);  clone_1 = arg4_1 = None
        exp: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub);  sub = None
        div: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp, arg5_1);  exp = arg5_1 = None
        mul_2: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(where, div);  where = None
        sum_1: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [-1], True)
        neg: "f32[8, 1024, 12, 513]" = torch.ops.aten.neg.default(div);  div = None
        fma: "f32[8, 1024, 12, 513]" = torch.ops.prims.fma.default(neg, sum_1, mul_2);  neg = sum_1 = mul_2 = None
        convert_element_type_3: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        permute_1: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(convert_element_type_3, [0, 2, 1, 3]);  convert_element_type_3 = None
        clone_2: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_4: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_2, _shape_param_7);  clone_2 = _shape_param_7 = None
        view_5: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_4, _shape_param_8);  view_4 = _shape_param_8 = None
        permute_2: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        clone_3: "bf16[8, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format)
        copy: "bf16[8, 1024, 12, 513]" = torch.ops.aten.copy.default(permute_2, clone_3);  permute_2 = clone_3 = None
        permute_3: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(copy, [0, 2, 1, 3]);  copy = None
        view_6: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_3, _shape_param_9);  permute_3 = _shape_param_9 = None
        view_7: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_6, _shape_param_10);  view_6 = _shape_param_10 = None
        permute_4: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
        slice_1: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_4, 1, -256, 9223372036854775807)
        slice_2: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1, 3, -257, 9223372036854775807)
        clone_4: "bf16[8, 256, 12, 257]" = torch.ops.aten.clone.default(slice_2, memory_format = torch.contiguous_format)
        full_3: "bf16[8, 256, 12, 257]" = torch.ops.aten.full.default(_shape_param_11, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_11 = None
        copy_1: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_2, full_3);  slice_2 = None
        slice_scatter_2: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1, copy_1, 3, -257, 9223372036854775807);  slice_1 = copy_1 = None
        slice_scatter_3: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_4, slice_scatter_2, 1, -256, 9223372036854775807);  permute_4 = slice_scatter_2 = None
        permute_5: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_3, [0, 2, 1, 3]);  slice_scatter_3 = None
        view_8: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_5, _shape_param_12);  permute_5 = _shape_param_12 = None
        full_4: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand: "bf16[8, 256, 12, 257]" = torch.ops.aten.expand.default(arg6_1, _shape_param_13);  arg6_1 = _shape_param_13 = None
        convert_element_type_4: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand, torch.bool);  expand = None
        where_1: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_4, full_4, clone_4);  clone_4 = None
        full_5: "bf16[8, 256, 12, 513]" = torch.ops.aten.full.default(_shape_param_14, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_14 = None
        slice_scatter_4: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_5, where_1, 3, -257, 9223372036854775807);  where_1 = None
        full_6: "bf16[8, 1024, 12, 513]" = torch.ops.aten.full.default(_shape_param_15, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_15 = None
        slice_scatter_5: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_6, slice_scatter_4, 1, -256, 9223372036854775807);  slice_scatter_4 = None
        permute_6: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_5, [0, 2, 1, 3]);  slice_scatter_5 = None
        clone_5: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
        view_9: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_5, _shape_param_16);  clone_5 = _shape_param_16 = None
        add: "bf16[96, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_8, view_9);  view_8 = view_9 = None
        view_10: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(add, _shape_param_17);  add = _shape_param_17 = None
        permute_7: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        slice_3: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_7, 1, 0, 256)
        slice_4: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_3, 3, 0, 257)
        clone_6: "bf16[8, 256, 12, 257]" = torch.ops.aten.clone.default(slice_4, memory_format = torch.contiguous_format)
        copy_2: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_4, full_3);  slice_4 = None
        slice_scatter_6: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_3, copy_2, 3, 0, 257);  slice_3 = copy_2 = None
        slice_scatter_7: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_7, slice_scatter_6, 1, 0, 256);  permute_7 = slice_scatter_6 = None
        permute_8: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_7, [0, 2, 1, 3]);  slice_scatter_7 = None
        view_11: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_8, _shape_param_18);  permute_8 = _shape_param_18 = None
        expand_1: "bf16[8, 256, 12, 257]" = torch.ops.aten.expand.default(arg7_1, _shape_param_19);  arg7_1 = _shape_param_19 = None
        convert_element_type_5: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_1, torch.bool);  expand_1 = None
        where_2: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_5, full_4, clone_6);  clone_6 = None
        slice_scatter_8: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_5, where_2, 3, 0, 257);  where_2 = None
        slice_scatter_9: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_6, slice_scatter_8, 1, 0, 256);  slice_scatter_8 = None
        permute_9: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_9, [0, 2, 1, 3]);  slice_scatter_9 = None
        clone_7: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None
        view_12: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_7, _shape_param_20);  clone_7 = _shape_param_20 = None
        add_1: "bf16[96, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_11, view_12);  view_11 = view_12 = None
        select: "bf16[96, 256, 513]" = torch.ops.aten.select.int(add_1, 1, 0)
        slice_5: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select, 1, 1, 256)
        slice_6: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_5, 2, 1, 256)
        clone_8: "bf16[96, 255, 255]" = torch.ops.aten.clone.default(slice_6, memory_format = torch.contiguous_format)
        full_7: "bf16[96, 255, 255]" = torch.ops.aten.full.default(_shape_param_21, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_21 = None
        copy_3: "bf16[96, 255, 255]" = torch.ops.aten.copy.default(slice_6, full_7);  slice_6 = None
        slice_scatter_10: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_5, copy_3, 2, 1, 256);  slice_5 = copy_3 = None
        slice_scatter_11: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select, slice_scatter_10, 1, 1, 256);  select = slice_scatter_10 = None
        select_scatter: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(add_1, slice_scatter_11, 1, 0);  add_1 = slice_scatter_11 = None
        full_8: "bf16[96, 255, 513]" = torch.ops.aten.full.default(_shape_param_22, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_22 = None
        slice_scatter_12: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(full_8, clone_8, 2, -255, 9223372036854775807);  clone_8 = None
        full_9: "bf16[96, 512, 513]" = torch.ops.aten.full.default(_shape_param_23, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_23 = None
        slice_scatter_13: "bf16[96, 512, 513]" = torch.ops.aten.slice_scatter.default(full_9, slice_scatter_12, 1, 0, 255);  slice_scatter_12 = None
        full_10: "bf16[96, 3, 512, 513]" = torch.ops.aten.full.default(_shape_param_24, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_24 = None
        select_scatter_1: "bf16[96, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_10, slice_scatter_13, 1, 0);  slice_scatter_13 = None
        slice_7: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter, 1, 1, 9223372036854775807)
        slice_8: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_7, 3, 0, 256)
        clone_9: "bf16[96, 3, 256, 256]" = torch.ops.aten.clone.default(slice_8, memory_format = torch.contiguous_format)
        full_11: "bf16[96, 3, 256, 256]" = torch.ops.aten.full.default(_shape_param_25, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_25 = None
        copy_4: "bf16[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_8, full_11);  slice_8 = None
        slice_scatter_14: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_7, copy_4, 3, 0, 256);  slice_7 = copy_4 = None
        slice_scatter_15: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter, slice_scatter_14, 1, 1, 9223372036854775807);  select_scatter = slice_scatter_14 = None
        full_12: "bf16[96, 3, 256, 513]" = torch.ops.aten.full.default(_shape_param_26, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_26 = None
        slice_scatter_16: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_12, clone_9, 3, 257, 9223372036854775807);  clone_9 = None
        slice_scatter_17: "bf16[96, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_10, slice_scatter_16, 2, -257, -1);  slice_scatter_16 = None
        add_2: "bf16[96, 3, 512, 513]" = torch.ops.aten.add.Tensor(select_scatter_1, slice_scatter_17);  select_scatter_1 = slice_scatter_17 = None
        select_1: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_15, 1, -1)
        slice_9: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_1, 2, 256, 9223372036854775807)
        clone_10: "bf16[96, 256, 257]" = torch.ops.aten.clone.default(slice_9, memory_format = torch.contiguous_format)
        full_13: "bf16[96, 256, 257]" = torch.ops.aten.full.default(_shape_param_27, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_27 = None
        copy_5: "bf16[96, 256, 257]" = torch.ops.aten.copy.default(slice_9, full_13);  slice_9 = None
        slice_scatter_18: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_1, copy_5, 2, 256, 9223372036854775807);  select_1 = copy_5 = None
        select_scatter_2: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_15, slice_scatter_18, 1, -1);  slice_scatter_15 = slice_scatter_18 = None
        full_14: "bf16[96, 256, 513]" = torch.ops.aten.full.default(_shape_param_28, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_28 = None
        slice_scatter_19: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(full_14, clone_10, 2, 0, 257);  clone_10 = None
        slice_scatter_20: "bf16[96, 512, 513]" = torch.ops.aten.slice_scatter.default(full_9, slice_scatter_19, 1, 256, 9223372036854775807);  slice_scatter_19 = None
        select_scatter_3: "bf16[96, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_10, slice_scatter_20, 1, -1);  slice_scatter_20 = None
        add_3: "bf16[96, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_2, select_scatter_3);  add_2 = select_scatter_3 = None
        slice_10: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_2, 1, 0, -1);  select_scatter_2 = None
        slice_11: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_10, 3, 256, 9223372036854775807);  slice_10 = None
        clone_11: "bf16[96, 3, 256, 257]" = torch.ops.aten.clone.default(slice_11, memory_format = torch.contiguous_format);  slice_11 = None
        slice_scatter_21: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_12, clone_11, 3, 0, 257);  clone_11 = None
        slice_scatter_22: "bf16[96, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_10, slice_scatter_21, 2, 0, 256);  slice_scatter_21 = None
        add_4: "bf16[96, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_3, slice_scatter_22);  add_3 = slice_scatter_22 = None
        view_13: "bf16[96, 3, 513, 512]" = torch.ops.aten.view.default(add_4, _shape_param_29);  add_4 = _shape_param_29 = None
        constant_pad_nd_1: "bf16[96, 3, 512, 512]" = torch.ops.aten.constant_pad_nd.default(view_13, [0, 0, 0, -1]);  view_13 = None
        view_14: "bf16[96, 3, 512, 512, 1]" = torch.ops.aten.view.default(constant_pad_nd_1, _shape_param_30);  constant_pad_nd_1 = _shape_param_30 = None
        permute_10: "bf16[96, 3, 512, 1, 512]" = torch.ops.aten.permute.default(view_14, [0, 1, 2, 4, 3]);  view_14 = None
        view_15: "bf16[288, 512, 512]" = torch.ops.aten.view.default(permute_10, _shape_param_31);  permute_10 = _shape_param_31 = None
        return (full, full_1, unsqueeze_1, full_2, full_3, full_4, convert_element_type_4, full_5, full_6, convert_element_type_5, full_7, full_8, full_9, full_10, full_11, full_12, full_13, full_14, view_15)



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
