"""
Standalone repro captured via capture_hook.
Label: XLNetLMHeadModel
Pattern hash: 3c53183461ca
Shape hash: d066309e
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
    def forward(self, arg0_1: "bf16[256, 512, 512]", arg1_1: "bf16[256, 512, 64]", arg2_1: "bf16[16, 16, 512, 1023]", arg3_1: "bf16[256, 64, 512]", arg4_1: "bf16[256, 1024, 64]", arg5_1: "bf16[16, 16, 512, 512]", arg6_1: "bf16[256, 64, 512]", arg7_1: "bf16[256, 512, 64]", arg8_1: "bf16[1024, 16384]", arg9_1: "bf16[1024, 8192]", arg10_1: "bf16[1024, 1024]", arg11_1: "f32[512, 16, 1024]", arg12_1: "bf16[1024, 1024]", arg13_1: "bf16[1024, 1024]", arg14_1: "f32[1024]", arg15_1: "f32[512, 16, 1024]", arg16_1: "f32[512, 16, 1]", arg17_1: "b8[512, 16, 1024]", arg18_1: "bf16[1024, 4096]", arg19_1: "bf16[8192, 4096]", arg20_1: "b8[512, 16, 4096]", arg21_1: "bf16[8192, 4096]", arg22_1: "bf16[4096, 1024]", arg23_1: "bf16[8192, 1024]", arg24_1: "f32[1024]", arg25_1: "f32[512, 16, 1024]", arg26_1: "f32[512, 16, 1]", arg27_1: "b8[512, 16, 1024]", arg28_1: "bf16[1024, 8192]", arg29_1: "bf16[1024, 1024]", arg30_1: "bf16[256, 64, 512]", arg31_1: "b8[16, 16, 512, 512]", arg32_1: "bf16[16, 16, 512, 512]", arg33_1: "f32[16, 16, 512, 1]", arg34_1: "f32[16, 16, 512, 1]", arg35_1: "b8[16, 16, 512, 1]", arg36_1: "f32[16, 16, 512, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42):
        # No stacktrace found for following nodes
        bmm: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        permute: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(view, [4, 0, 1, 3, 2]);  view = None
        permute_1: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute, [4, 1, 2, 3, 0]);  permute = None
        squeeze: "bf16[512, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1, 4);  permute_1 = None
        view_1: "bf16[16, 16, 1023, 512]" = torch.ops.aten.view.default(arg2_1, _shape_param_1);  arg2_1 = _shape_param_1 = None
        full: "bf16[16, 16, 1024, 512]" = torch.ops.aten.full.default(_shape_param_2, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_2 = None
        slice_scatter: "bf16[16, 16, 1024, 512]" = torch.ops.aten.slice_scatter.default(full, view_1, 2, 1, 9223372036854775807);  view_1 = None
        view_2: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(slice_scatter, _shape_param_3);  slice_scatter = _shape_param_3 = None
        view_3: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.view.default(view_2, _shape_param_4);  view_2 = _shape_param_4 = None
        permute_2: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.permute.default(view_3, [0, 1, 2, 4, 3]);  view_3 = None
        view_4: "bf16[256, 512, 1024]" = torch.ops.aten.view.default(permute_2, _shape_param_5);  permute_2 = _shape_param_5 = None
        bmm_1: "bf16[256, 64, 1024]" = torch.ops.aten.bmm.default(arg3_1, view_4);  arg3_1 = None
        bmm_2: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_4, arg4_1);  view_4 = arg4_1 = None
        view_5: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.view.default(bmm_1, _shape_param_6);  bmm_1 = _shape_param_6 = None
        permute_3: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(view_5, [0, 1, 4, 3, 2]);  view_5 = None
        view_6: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.view.default(bmm_2, _shape_param_7);  bmm_2 = _shape_param_7 = None
        permute_4: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_6, [0, 1, 2, 4, 3]);  view_6 = None
        permute_5: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_3, [3, 0, 1, 4, 2]);  permute_3 = None
        squeeze_1: "bf16[1024, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_5, 4);  permute_5 = None
        permute_6: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_4, [2, 0, 1, 4, 3]);  permute_4 = None
        squeeze_2: "bf16[512, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_6, 4);  permute_6 = None
        convert_element_type: "f32[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(squeeze_2, torch.float32);  squeeze_2 = None
        convert_element_type_1: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.bfloat16)
        sum_1: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(convert_element_type, [0, 1], True, dtype = torch.float32);  convert_element_type = None
        view_7: "f32[16, 64]" = torch.ops.aten.view.default(sum_1, _shape_param_8);  sum_1 = _shape_param_8 = None
        view_8: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.view.default(arg5_1, _shape_param_9);  arg5_1 = _shape_param_9 = None
        permute_7: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.permute.default(view_8, [0, 1, 2, 4, 3]);  view_8 = None
        view_9: "bf16[256, 512, 512]" = torch.ops.aten.view.default(permute_7, _shape_param_10);  permute_7 = _shape_param_10 = None
        bmm_3: "bf16[256, 64, 512]" = torch.ops.aten.bmm.default(arg6_1, view_9);  arg6_1 = None
        bmm_4: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_9, arg7_1);  view_9 = arg7_1 = None
        view_10: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.view.default(bmm_3, _shape_param_11);  bmm_3 = _shape_param_11 = None
        permute_8: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(view_10, [0, 1, 4, 3, 2]);  view_10 = None
        view_11: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.view.default(bmm_4, _shape_param_12);  bmm_4 = _shape_param_12 = None
        permute_9: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_11, [0, 1, 2, 4, 3]);  view_11 = None
        permute_10: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_8, [3, 0, 1, 4, 2]);  permute_8 = None
        squeeze_3: "bf16[512, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_10, 4);  permute_10 = None
        permute_11: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_9, [2, 0, 1, 4, 3]);  permute_9 = None
        squeeze_4: "bf16[512, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_11, 4);  permute_11 = None
        convert_element_type_2: "f32[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(squeeze_4, torch.float32);  squeeze_4 = None
        convert_element_type_3: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.bfloat16)
        sum_2: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(convert_element_type_2, [0, 1], True, dtype = torch.float32);  convert_element_type_2 = None
        view_12: "f32[16, 64]" = torch.ops.aten.view.default(sum_2, _shape_param_13);  sum_2 = _shape_param_13 = None
        add: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(convert_element_type_1, convert_element_type_3);  convert_element_type_1 = convert_element_type_3 = None
        view_13: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.view.default(squeeze_1, _shape_param_14);  squeeze_1 = _shape_param_14 = None
        permute_12: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.permute.default(view_13, [0, 1, 4, 2, 3]);  view_13 = None
        clone: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None
        view_14: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(clone, _shape_param_15);  clone = _shape_param_15 = None
        squeeze_5: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_14, 0);  view_14 = None
        mm: "bf16[1024, 1024]" = torch.ops.aten.mm.default(arg8_1, squeeze_5);  arg8_1 = squeeze_5 = None
        unsqueeze: "bf16[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm, 0);  mm = None
        view_15: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.view.default(unsqueeze, _shape_param_16);  unsqueeze = _shape_param_16 = None
        squeeze_6: "bf16[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(view_15, 4);  view_15 = None
        squeeze_7: "bf16[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_6, 3);  squeeze_6 = None
        convert_element_type_4: "f32[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(squeeze_7, torch.float32);  squeeze_7 = None
        view_16: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.view.default(squeeze, _shape_param_17);  squeeze = _shape_param_17 = None
        permute_13: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.permute.default(view_16, [0, 1, 4, 2, 3]);  view_16 = None
        clone_1: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
        view_17: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_1, _shape_param_18);  clone_1 = _shape_param_18 = None
        squeeze_8: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_17, 0);  view_17 = None
        mm_1: "bf16[1024, 1024]" = torch.ops.aten.mm.default(arg9_1, squeeze_8)
        unsqueeze_1: "bf16[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_1, 0);  mm_1 = None
        mm_2: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_8, arg10_1);  squeeze_8 = arg10_1 = None
        unsqueeze_2: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_2, 0);  mm_2 = None
        view_18: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.view.default(unsqueeze_1, _shape_param_19);  unsqueeze_1 = _shape_param_19 = None
        view_19: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.view.default(unsqueeze_2, _shape_param_20);  unsqueeze_2 = _shape_param_20 = None
        squeeze_9: "bf16[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(view_18, 4);  view_18 = None
        squeeze_10: "bf16[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_9, 3);  squeeze_9 = None
        squeeze_11: "bf16[512, 16, 1024, 1]" = torch.ops.aten.squeeze.dim(view_19, 4);  view_19 = None
        squeeze_12: "bf16[512, 16, 1024]" = torch.ops.aten.squeeze.dim(squeeze_11, 3);  squeeze_11 = None
        convert_element_type_5: "f32[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(squeeze_10, torch.float32);  squeeze_10 = None
        convert_element_type_6: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(squeeze_12, torch.float32);  squeeze_12 = None
        add_1: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(arg11_1, convert_element_type_6);  arg11_1 = convert_element_type_6 = None
        view_20: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.view.default(squeeze_3, _shape_param_21);  squeeze_3 = _shape_param_21 = None
        permute_14: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.permute.default(view_20, [0, 1, 4, 2, 3]);  view_20 = None
        clone_2: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
        view_21: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_2, _shape_param_22);  clone_2 = _shape_param_22 = None
        squeeze_13: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_21, 0);  view_21 = None
        mm_3: "bf16[1024, 1024]" = torch.ops.aten.mm.default(arg9_1, squeeze_13)
        unsqueeze_3: "bf16[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_3, 0);  mm_3 = None
        mm_4: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_13, arg12_1);  squeeze_13 = arg12_1 = None
        unsqueeze_4: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_4, 0);  mm_4 = None
        view_22: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.view.default(unsqueeze_3, _shape_param_23);  unsqueeze_3 = _shape_param_23 = None
        view_23: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.view.default(unsqueeze_4, _shape_param_24);  unsqueeze_4 = _shape_param_24 = None
        squeeze_14: "bf16[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(view_22, 4);  view_22 = None
        squeeze_15: "bf16[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_14, 3);  squeeze_14 = None
        squeeze_16: "bf16[512, 16, 1024, 1]" = torch.ops.aten.squeeze.dim(view_23, 4);  view_23 = None
        squeeze_17: "bf16[512, 16, 1024]" = torch.ops.aten.squeeze.dim(squeeze_16, 3);  squeeze_16 = None
        convert_element_type_7: "f32[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(squeeze_15, torch.float32);  squeeze_15 = None
        convert_element_type_8: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(squeeze_17, torch.float32);  squeeze_17 = None
        add_2: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(add_1, convert_element_type_8);  add_1 = convert_element_type_8 = None
        view_24: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.view.default(add, _shape_param_25);  add = _shape_param_25 = None
        permute_15: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.permute.default(view_24, [0, 1, 4, 2, 3]);  view_24 = None
        clone_3: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
        view_25: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_3, _shape_param_26);  clone_3 = _shape_param_26 = None
        squeeze_18: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_25, 0);  view_25 = None
        mm_5: "bf16[1024, 1024]" = torch.ops.aten.mm.default(arg9_1, squeeze_18);  arg9_1 = None
        unsqueeze_5: "bf16[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_5, 0);  mm_5 = None
        mm_6: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_18, arg13_1);  squeeze_18 = arg13_1 = None
        unsqueeze_6: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_6, 0);  mm_6 = None
        view_26: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.view.default(unsqueeze_5, _shape_param_27);  unsqueeze_5 = _shape_param_27 = None
        view_27: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.view.default(unsqueeze_6, _shape_param_28);  unsqueeze_6 = _shape_param_28 = None
        squeeze_19: "bf16[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(view_26, 4);  view_26 = None
        squeeze_20: "bf16[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_19, 3);  squeeze_19 = None
        squeeze_21: "bf16[512, 16, 1024, 1]" = torch.ops.aten.squeeze.dim(view_27, 4);  view_27 = None
        squeeze_22: "bf16[512, 16, 1024]" = torch.ops.aten.squeeze.dim(squeeze_21, 3);  squeeze_21 = None
        convert_element_type_9: "f32[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(squeeze_20, torch.float32);  squeeze_20 = None
        convert_element_type_10: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(squeeze_22, torch.float32);  squeeze_22 = None
        add_3: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(add_2, convert_element_type_10);  add_2 = convert_element_type_10 = None
        mul: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_3, arg14_1);  arg14_1 = None
        mul_1: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul, 1024)
        sum_3: "f32[512, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        mul_2: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul, arg15_1);  mul = None
        sum_4: "f32[512, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True);  mul_2 = None
        mul_3: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(arg15_1, sum_4);  sum_4 = None
        sub: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(mul_1, sum_3);  mul_1 = sum_3 = None
        sub_1: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(sub, mul_3);  sub = mul_3 = None
        mul_4: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(arg16_1, sub_1);  arg16_1 = sub_1 = None
        mul_5: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_3, arg15_1);  arg15_1 = None
        sum_5: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_5, [0, 1]);  mul_5 = None
        sum_6: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_3, [0, 1]);  add_3 = None
        convert_element_type_11: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16)
        convert_element_type_12: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(arg17_1, torch.bfloat16);  arg17_1 = None
        mul_6: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.1111111111111112);  convert_element_type_12 = None
        mul_7: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_11, mul_6);  convert_element_type_11 = mul_6 = None
        view_28: "bf16[8192, 1024]" = torch.ops.aten.view.default(mul_7, _shape_param_29);  mul_7 = _shape_param_29 = None
        mm_7: "bf16[8192, 4096]" = torch.ops.aten.mm.default(view_28, arg18_1);  arg18_1 = None
        permute_16: "bf16[1024, 8192]" = torch.ops.aten.permute.default(view_28, [1, 0])
        mm_8: "bf16[1024, 4096]" = torch.ops.aten.mm.default(permute_16, arg19_1);  permute_16 = arg19_1 = None
        sum_7: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_28, [0], True, dtype = torch.float32);  view_28 = None
        view_29: "f32[1024]" = torch.ops.aten.view.default(sum_7, _shape_param_30);  sum_7 = _shape_param_30 = None
        convert_element_type_13: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_29, torch.bfloat16);  view_29 = None
        view_30: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(mm_7, _shape_param_31);  mm_7 = _shape_param_31 = None
        convert_element_type_14: "f32[1024, 4096]" = torch.ops.prims.convert_element_type.default(mm_8, torch.float32);  mm_8 = None
        convert_element_type_15: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_13, torch.float32);  convert_element_type_13 = None
        convert_element_type_16: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(arg20_1, torch.bfloat16);  arg20_1 = None
        mul_8: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
        mul_9: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(view_30, mul_8);  view_30 = mul_8 = None
        convert_element_type_17: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_9, torch.float32);  mul_9 = None
        view_31: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(arg21_1, _shape_param_32);  arg21_1 = _shape_param_32 = None
        convert_element_type_18: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_31, torch.float32);  view_31 = None
        mul_10: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 0.7071067811865476)
        erf: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_10);  mul_10 = None
        add_4: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_11: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(add_4, 0.5);  add_4 = None
        mul_12: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_18, convert_element_type_18)
        mul_13: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_12, -0.5);  mul_12 = None
        exp: "f32[512, 16, 4096]" = torch.ops.aten.exp.default(mul_13);  mul_13 = None
        mul_14: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_15: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_18, mul_14);  convert_element_type_18 = mul_14 = None
        add_5: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(mul_11, mul_15);  mul_11 = mul_15 = None
        mul_16: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_17, add_5);  convert_element_type_17 = add_5 = None
        convert_element_type_19: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_16, torch.bfloat16);  mul_16 = None
        view_32: "bf16[8192, 4096]" = torch.ops.aten.view.default(convert_element_type_19, _shape_param_33);  convert_element_type_19 = _shape_param_33 = None
        mm_9: "bf16[8192, 1024]" = torch.ops.aten.mm.default(view_32, arg22_1);  arg22_1 = None
        permute_17: "bf16[4096, 8192]" = torch.ops.aten.permute.default(view_32, [1, 0])
        mm_10: "bf16[4096, 1024]" = torch.ops.aten.mm.default(permute_17, arg23_1);  permute_17 = arg23_1 = None
        sum_8: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_32, [0], True, dtype = torch.float32);  view_32 = None
        view_33: "f32[4096]" = torch.ops.aten.view.default(sum_8, _shape_param_34);  sum_8 = _shape_param_34 = None
        convert_element_type_20: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_33, torch.bfloat16);  view_33 = None
        view_34: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(mm_9, _shape_param_35);  mm_9 = _shape_param_35 = None
        convert_element_type_21: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(view_34, torch.float32);  view_34 = None
        add_6: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_4, convert_element_type_21);  mul_4 = convert_element_type_21 = None
        convert_element_type_22: "f32[4096, 1024]" = torch.ops.prims.convert_element_type.default(mm_10, torch.float32);  mm_10 = None
        convert_element_type_23: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_20, torch.float32);  convert_element_type_20 = None
        mul_17: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_6, arg24_1);  arg24_1 = None
        mul_18: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_17, 1024)
        sum_9: "f32[512, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_17, [2], True)
        mul_19: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_17, arg25_1);  mul_17 = None
        sum_10: "f32[512, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_19, [2], True);  mul_19 = None
        mul_20: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(arg25_1, sum_10);  sum_10 = None
        sub_2: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(mul_18, sum_9);  mul_18 = sum_9 = None
        sub_3: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(sub_2, mul_20);  sub_2 = mul_20 = None
        mul_21: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(arg26_1, sub_3);  arg26_1 = sub_3 = None
        mul_22: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(add_6, arg25_1);  arg25_1 = None
        sum_11: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_22, [0, 1]);  mul_22 = None
        sum_12: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_6, [0, 1]);  add_6 = None
        convert_element_type_24: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(mul_21, torch.bfloat16)
        convert_element_type_25: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(arg27_1, torch.bfloat16);  arg27_1 = None
        mul_23: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
        mul_24: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_24, mul_23);  convert_element_type_24 = mul_23 = None
        view_35: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.view.default(mul_24, _shape_param_36);  mul_24 = _shape_param_36 = None
        permute_18: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.permute.default(view_35, [0, 1, 3, 4, 2]);  view_35 = None
        view_36: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(permute_18, _shape_param_37);  permute_18 = _shape_param_37 = None
        squeeze_23: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_36, 0);  view_36 = None
        mm_11: "bf16[1024, 1024]" = torch.ops.aten.mm.default(arg28_1, squeeze_23);  arg28_1 = None
        unsqueeze_7: "bf16[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_11, 0);  mm_11 = None
        mm_12: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_23, arg29_1);  squeeze_23 = arg29_1 = None
        unsqueeze_8: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_12, 0);  mm_12 = None
        view_37: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.view.default(unsqueeze_7, _shape_param_38);  unsqueeze_7 = _shape_param_38 = None
        permute_19: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(view_37, [3, 4, 2, 0, 1]);  view_37 = None
        view_38: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.view.default(unsqueeze_8, _shape_param_39);  unsqueeze_8 = _shape_param_39 = None
        permute_20: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(view_38, [0, 1, 4, 2, 3]);  view_38 = None
        permute_21: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.permute.default(permute_19, [2, 4, 3, 0, 1]);  permute_19 = None
        squeeze_24: "bf16[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(permute_21, 4);  permute_21 = None
        squeeze_25: "bf16[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_24, 3);  squeeze_24 = None
        permute_22: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_20, [0, 1, 4, 3, 2]);  permute_20 = None
        squeeze_26: "bf16[512, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_22, 4);  permute_22 = None
        convert_element_type_26: "f32[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(squeeze_25, torch.float32);  squeeze_25 = None
        view_39: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.view.default(squeeze_26, _shape_param_40);  squeeze_26 = _shape_param_40 = None
        permute_23: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_39, [1, 2, 0, 4, 3]);  view_39 = None
        clone_4: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
        view_40: "bf16[256, 512, 64]" = torch.ops.aten.view.default(clone_4, _shape_param_41);  clone_4 = _shape_param_41 = None
        bmm_5: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_40, arg30_1);  arg30_1 = None
        view_41: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.view.default(bmm_5, _shape_param_42);  bmm_5 = _shape_param_42 = None
        squeeze_27: "bf16[16, 16, 512, 512]" = torch.ops.aten.squeeze.dim(view_41, 4);  view_41 = None
        convert_element_type_27: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(squeeze_27, torch.float32);  squeeze_27 = None
        convert_element_type_28: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(arg31_1, torch.float32);  arg31_1 = None
        mul_25: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.1111111111111112);  convert_element_type_28 = None
        mul_26: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_27, mul_25);  convert_element_type_27 = mul_25 = None
        mul_27: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(arg32_1, 0.125)
        convert_element_type_29: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_27, torch.float32);  mul_27 = None
        convert_element_type_30: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(arg32_1, torch.float32);  arg32_1 = None
        mul_28: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 1);  convert_element_type_30 = None
        sub_4: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_28, arg33_1);  mul_28 = arg33_1 = None
        mul_29: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_4, 0.125);  sub_4 = None
        sub_5: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_29, arg34_1);  convert_element_type_29 = arg34_1 = None
        where: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(arg35_1, mul_29, sub_5);  arg35_1 = mul_29 = sub_5 = None
        exp_1: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where);  where = None
        div: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, arg36_1);  exp_1 = arg36_1 = None
        mul_30: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_26, div);  mul_26 = None
        sum_13: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_30, [3], True)
        neg: "f32[16, 16, 512, 512]" = torch.ops.aten.neg.default(div);  div = None
        fma: "f32[16, 16, 512, 512]" = torch.ops.prims.fma.default(neg, sum_13, mul_30);  neg = sum_13 = mul_30 = None
        convert_element_type_31: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        mul_31: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 0.125);  convert_element_type_31 = None
        return (full, view_7, view_12, convert_element_type_4, convert_element_type_5, convert_element_type_7, convert_element_type_9, sum_5, sum_6, convert_element_type_14, convert_element_type_15, convert_element_type_22, convert_element_type_23, mul_21, sum_11, sum_12, convert_element_type_26, view_40, mul_31)



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
