"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_train
Pattern hash: 57057d0973c9
Shape hash: 019a4c87
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
    def forward(self, arg0_1: "bf16[128, 12, 197, 197]", arg1_1: "i64[197, 197]", arg2_1: "bf16[128, 12, 197, 197]", arg3_1: "i64[197, 197]", arg4_1: "bf16[128, 12, 197, 197]", arg5_1: "i64[197, 197]", arg6_1: "bf16[128, 12, 197, 197]", arg7_1: "i64[197, 197]", arg8_1: "bf16[128, 12, 197, 197]", arg9_1: "i64[197, 197]", arg10_1: "bf16[128, 12, 197, 197]", arg11_1: "i64[197, 197]", arg12_1: "bf16[128, 12, 197, 197]", arg13_1: "i64[197, 197]", arg14_1: "bf16[128, 12, 197, 197]", arg15_1: "i64[197, 197]", arg16_1: "bf16[128, 12, 197, 197]", arg17_1: "i64[197, 197]", arg18_1: "bf16[128, 12, 197, 197]", arg19_1: "i64[197, 197]", arg20_1: "bf16[128, 12, 197, 197]", arg21_1: "i64[197, 197]", arg22_1: "bf16[128, 12, 197, 197]", arg23_1: "i64[197, 197]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25):
        # No stacktrace found for following nodes
        sum_1: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(arg0_1, [0], True, dtype = torch.float32);  arg0_1 = None
        convert_element_type: "bf16[1, 12, 197, 197]" = torch.ops.prims.convert_element_type.default(sum_1, torch.bfloat16);  sum_1 = None
        full: "bf16[1, 12, 197, 200]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        slice_scatter: "bf16[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full, convert_element_type, -1, 0, 197);  convert_element_type = None
        constant_pad_nd: "bf16[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter, _shape_param_1);  slice_scatter = _shape_param_1 = None
        convert_element_type_1: "f32[1, 12, 197, 197]" = torch.ops.prims.convert_element_type.default(constant_pad_nd, torch.float32);  constant_pad_nd = None
        squeeze: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(convert_element_type_1, 0);  convert_element_type_1 = None
        permute: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze, [1, 2, 0]);  squeeze = None
        view: "f32[38809, 12]" = torch.ops.aten.view.default(permute, _shape_param_2);  permute = _shape_param_2 = None
        full_1: "f32[732, 12]" = torch.ops.aten.full.default(_shape_param_3, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_3 = None
        view_1: "i64[38809]" = torch.ops.aten.view.default(arg1_1, [-1]);  arg1_1 = None
        index_put: "f32[732, 12]" = torch.ops.aten.index_put.default(full_1, [view_1], view, True);  view_1 = view = None
        sum_2: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(arg2_1, [0], True, dtype = torch.float32);  arg2_1 = None
        convert_element_type_2: "bf16[1, 12, 197, 197]" = torch.ops.prims.convert_element_type.default(sum_2, torch.bfloat16);  sum_2 = None
        slice_scatter_1: "bf16[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full, convert_element_type_2, -1, 0, 197);  convert_element_type_2 = None
        constant_pad_nd_1: "bf16[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_1, _shape_param_4);  slice_scatter_1 = _shape_param_4 = None
        convert_element_type_3: "f32[1, 12, 197, 197]" = torch.ops.prims.convert_element_type.default(constant_pad_nd_1, torch.float32);  constant_pad_nd_1 = None
        squeeze_1: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(convert_element_type_3, 0);  convert_element_type_3 = None
        permute_1: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_1, [1, 2, 0]);  squeeze_1 = None
        view_2: "f32[38809, 12]" = torch.ops.aten.view.default(permute_1, _shape_param_5);  permute_1 = _shape_param_5 = None
        view_3: "i64[38809]" = torch.ops.aten.view.default(arg3_1, [-1]);  arg3_1 = None
        index_put_1: "f32[732, 12]" = torch.ops.aten.index_put.default(full_1, [view_3], view_2, True);  view_3 = view_2 = None
        sum_3: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(arg4_1, [0], True, dtype = torch.float32);  arg4_1 = None
        convert_element_type_4: "bf16[1, 12, 197, 197]" = torch.ops.prims.convert_element_type.default(sum_3, torch.bfloat16);  sum_3 = None
        slice_scatter_2: "bf16[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full, convert_element_type_4, -1, 0, 197);  convert_element_type_4 = None
        constant_pad_nd_2: "bf16[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_2, _shape_param_6);  slice_scatter_2 = _shape_param_6 = None
        convert_element_type_5: "f32[1, 12, 197, 197]" = torch.ops.prims.convert_element_type.default(constant_pad_nd_2, torch.float32);  constant_pad_nd_2 = None
        squeeze_2: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(convert_element_type_5, 0);  convert_element_type_5 = None
        permute_2: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_2, [1, 2, 0]);  squeeze_2 = None
        view_4: "f32[38809, 12]" = torch.ops.aten.view.default(permute_2, _shape_param_7);  permute_2 = _shape_param_7 = None
        view_5: "i64[38809]" = torch.ops.aten.view.default(arg5_1, [-1]);  arg5_1 = None
        index_put_2: "f32[732, 12]" = torch.ops.aten.index_put.default(full_1, [view_5], view_4, True);  view_5 = view_4 = None
        sum_4: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(arg6_1, [0], True, dtype = torch.float32);  arg6_1 = None
        convert_element_type_6: "bf16[1, 12, 197, 197]" = torch.ops.prims.convert_element_type.default(sum_4, torch.bfloat16);  sum_4 = None
        slice_scatter_3: "bf16[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full, convert_element_type_6, -1, 0, 197);  convert_element_type_6 = None
        constant_pad_nd_3: "bf16[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_3, _shape_param_8);  slice_scatter_3 = _shape_param_8 = None
        convert_element_type_7: "f32[1, 12, 197, 197]" = torch.ops.prims.convert_element_type.default(constant_pad_nd_3, torch.float32);  constant_pad_nd_3 = None
        squeeze_3: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(convert_element_type_7, 0);  convert_element_type_7 = None
        permute_3: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_3, [1, 2, 0]);  squeeze_3 = None
        view_6: "f32[38809, 12]" = torch.ops.aten.view.default(permute_3, _shape_param_9);  permute_3 = _shape_param_9 = None
        view_7: "i64[38809]" = torch.ops.aten.view.default(arg7_1, [-1]);  arg7_1 = None
        index_put_3: "f32[732, 12]" = torch.ops.aten.index_put.default(full_1, [view_7], view_6, True);  view_7 = view_6 = None
        sum_5: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(arg8_1, [0], True, dtype = torch.float32);  arg8_1 = None
        convert_element_type_8: "bf16[1, 12, 197, 197]" = torch.ops.prims.convert_element_type.default(sum_5, torch.bfloat16);  sum_5 = None
        slice_scatter_4: "bf16[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full, convert_element_type_8, -1, 0, 197);  convert_element_type_8 = None
        constant_pad_nd_4: "bf16[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_4, _shape_param_10);  slice_scatter_4 = _shape_param_10 = None
        convert_element_type_9: "f32[1, 12, 197, 197]" = torch.ops.prims.convert_element_type.default(constant_pad_nd_4, torch.float32);  constant_pad_nd_4 = None
        squeeze_4: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(convert_element_type_9, 0);  convert_element_type_9 = None
        permute_4: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_4, [1, 2, 0]);  squeeze_4 = None
        view_8: "f32[38809, 12]" = torch.ops.aten.view.default(permute_4, _shape_param_11);  permute_4 = _shape_param_11 = None
        view_9: "i64[38809]" = torch.ops.aten.view.default(arg9_1, [-1]);  arg9_1 = None
        index_put_4: "f32[732, 12]" = torch.ops.aten.index_put.default(full_1, [view_9], view_8, True);  view_9 = view_8 = None
        sum_6: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(arg10_1, [0], True, dtype = torch.float32);  arg10_1 = None
        convert_element_type_10: "bf16[1, 12, 197, 197]" = torch.ops.prims.convert_element_type.default(sum_6, torch.bfloat16);  sum_6 = None
        slice_scatter_5: "bf16[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full, convert_element_type_10, -1, 0, 197);  convert_element_type_10 = None
        constant_pad_nd_5: "bf16[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_5, _shape_param_12);  slice_scatter_5 = _shape_param_12 = None
        convert_element_type_11: "f32[1, 12, 197, 197]" = torch.ops.prims.convert_element_type.default(constant_pad_nd_5, torch.float32);  constant_pad_nd_5 = None
        squeeze_5: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(convert_element_type_11, 0);  convert_element_type_11 = None
        permute_5: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_5, [1, 2, 0]);  squeeze_5 = None
        view_10: "f32[38809, 12]" = torch.ops.aten.view.default(permute_5, _shape_param_13);  permute_5 = _shape_param_13 = None
        view_11: "i64[38809]" = torch.ops.aten.view.default(arg11_1, [-1]);  arg11_1 = None
        index_put_5: "f32[732, 12]" = torch.ops.aten.index_put.default(full_1, [view_11], view_10, True);  view_11 = view_10 = None
        sum_7: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(arg12_1, [0], True, dtype = torch.float32);  arg12_1 = None
        convert_element_type_12: "bf16[1, 12, 197, 197]" = torch.ops.prims.convert_element_type.default(sum_7, torch.bfloat16);  sum_7 = None
        slice_scatter_6: "bf16[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full, convert_element_type_12, -1, 0, 197);  convert_element_type_12 = None
        constant_pad_nd_6: "bf16[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_6, _shape_param_14);  slice_scatter_6 = _shape_param_14 = None
        convert_element_type_13: "f32[1, 12, 197, 197]" = torch.ops.prims.convert_element_type.default(constant_pad_nd_6, torch.float32);  constant_pad_nd_6 = None
        squeeze_6: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(convert_element_type_13, 0);  convert_element_type_13 = None
        permute_6: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_6, [1, 2, 0]);  squeeze_6 = None
        view_12: "f32[38809, 12]" = torch.ops.aten.view.default(permute_6, _shape_param_15);  permute_6 = _shape_param_15 = None
        view_13: "i64[38809]" = torch.ops.aten.view.default(arg13_1, [-1]);  arg13_1 = None
        index_put_6: "f32[732, 12]" = torch.ops.aten.index_put.default(full_1, [view_13], view_12, True);  view_13 = view_12 = None
        sum_8: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(arg14_1, [0], True, dtype = torch.float32);  arg14_1 = None
        convert_element_type_14: "bf16[1, 12, 197, 197]" = torch.ops.prims.convert_element_type.default(sum_8, torch.bfloat16);  sum_8 = None
        slice_scatter_7: "bf16[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full, convert_element_type_14, -1, 0, 197);  convert_element_type_14 = None
        constant_pad_nd_7: "bf16[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_7, _shape_param_16);  slice_scatter_7 = _shape_param_16 = None
        convert_element_type_15: "f32[1, 12, 197, 197]" = torch.ops.prims.convert_element_type.default(constant_pad_nd_7, torch.float32);  constant_pad_nd_7 = None
        squeeze_7: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(convert_element_type_15, 0);  convert_element_type_15 = None
        permute_7: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_7, [1, 2, 0]);  squeeze_7 = None
        view_14: "f32[38809, 12]" = torch.ops.aten.view.default(permute_7, _shape_param_17);  permute_7 = _shape_param_17 = None
        view_15: "i64[38809]" = torch.ops.aten.view.default(arg15_1, [-1]);  arg15_1 = None
        index_put_7: "f32[732, 12]" = torch.ops.aten.index_put.default(full_1, [view_15], view_14, True);  view_15 = view_14 = None
        sum_9: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(arg16_1, [0], True, dtype = torch.float32);  arg16_1 = None
        convert_element_type_16: "bf16[1, 12, 197, 197]" = torch.ops.prims.convert_element_type.default(sum_9, torch.bfloat16);  sum_9 = None
        slice_scatter_8: "bf16[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full, convert_element_type_16, -1, 0, 197);  convert_element_type_16 = None
        constant_pad_nd_8: "bf16[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_8, _shape_param_18);  slice_scatter_8 = _shape_param_18 = None
        convert_element_type_17: "f32[1, 12, 197, 197]" = torch.ops.prims.convert_element_type.default(constant_pad_nd_8, torch.float32);  constant_pad_nd_8 = None
        squeeze_8: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(convert_element_type_17, 0);  convert_element_type_17 = None
        permute_8: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_8, [1, 2, 0]);  squeeze_8 = None
        view_16: "f32[38809, 12]" = torch.ops.aten.view.default(permute_8, _shape_param_19);  permute_8 = _shape_param_19 = None
        view_17: "i64[38809]" = torch.ops.aten.view.default(arg17_1, [-1]);  arg17_1 = None
        index_put_8: "f32[732, 12]" = torch.ops.aten.index_put.default(full_1, [view_17], view_16, True);  view_17 = view_16 = None
        sum_10: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(arg18_1, [0], True, dtype = torch.float32);  arg18_1 = None
        convert_element_type_18: "bf16[1, 12, 197, 197]" = torch.ops.prims.convert_element_type.default(sum_10, torch.bfloat16);  sum_10 = None
        slice_scatter_9: "bf16[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full, convert_element_type_18, -1, 0, 197);  convert_element_type_18 = None
        constant_pad_nd_9: "bf16[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_9, _shape_param_20);  slice_scatter_9 = _shape_param_20 = None
        convert_element_type_19: "f32[1, 12, 197, 197]" = torch.ops.prims.convert_element_type.default(constant_pad_nd_9, torch.float32);  constant_pad_nd_9 = None
        squeeze_9: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(convert_element_type_19, 0);  convert_element_type_19 = None
        permute_9: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_9, [1, 2, 0]);  squeeze_9 = None
        view_18: "f32[38809, 12]" = torch.ops.aten.view.default(permute_9, _shape_param_21);  permute_9 = _shape_param_21 = None
        view_19: "i64[38809]" = torch.ops.aten.view.default(arg19_1, [-1]);  arg19_1 = None
        index_put_9: "f32[732, 12]" = torch.ops.aten.index_put.default(full_1, [view_19], view_18, True);  view_19 = view_18 = None
        sum_11: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(arg20_1, [0], True, dtype = torch.float32);  arg20_1 = None
        convert_element_type_20: "bf16[1, 12, 197, 197]" = torch.ops.prims.convert_element_type.default(sum_11, torch.bfloat16);  sum_11 = None
        slice_scatter_10: "bf16[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full, convert_element_type_20, -1, 0, 197);  convert_element_type_20 = None
        constant_pad_nd_10: "bf16[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_10, _shape_param_22);  slice_scatter_10 = _shape_param_22 = None
        convert_element_type_21: "f32[1, 12, 197, 197]" = torch.ops.prims.convert_element_type.default(constant_pad_nd_10, torch.float32);  constant_pad_nd_10 = None
        squeeze_10: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(convert_element_type_21, 0);  convert_element_type_21 = None
        permute_10: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_10, [1, 2, 0]);  squeeze_10 = None
        view_20: "f32[38809, 12]" = torch.ops.aten.view.default(permute_10, _shape_param_23);  permute_10 = _shape_param_23 = None
        view_21: "i64[38809]" = torch.ops.aten.view.default(arg21_1, [-1]);  arg21_1 = None
        index_put_10: "f32[732, 12]" = torch.ops.aten.index_put.default(full_1, [view_21], view_20, True);  view_21 = view_20 = None
        sum_12: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(arg22_1, [0], True, dtype = torch.float32);  arg22_1 = None
        convert_element_type_22: "bf16[1, 12, 197, 197]" = torch.ops.prims.convert_element_type.default(sum_12, torch.bfloat16);  sum_12 = None
        slice_scatter_11: "bf16[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full, convert_element_type_22, -1, 0, 197);  full = convert_element_type_22 = None
        constant_pad_nd_11: "bf16[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_11, _shape_param_24);  slice_scatter_11 = _shape_param_24 = None
        convert_element_type_23: "f32[1, 12, 197, 197]" = torch.ops.prims.convert_element_type.default(constant_pad_nd_11, torch.float32);  constant_pad_nd_11 = None
        squeeze_11: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(convert_element_type_23, 0);  convert_element_type_23 = None
        permute_11: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_11, [1, 2, 0]);  squeeze_11 = None
        view_22: "f32[38809, 12]" = torch.ops.aten.view.default(permute_11, _shape_param_25);  permute_11 = _shape_param_25 = None
        view_23: "i64[38809]" = torch.ops.aten.view.default(arg23_1, [-1]);  arg23_1 = None
        index_put_11: "f32[732, 12]" = torch.ops.aten.index_put.default(full_1, [view_23], view_22, True);  full_1 = view_23 = view_22 = None
        return (index_put, index_put_1, index_put_2, index_put_3, index_put_4, index_put_5, index_put_6, index_put_7, index_put_8, index_put_9, index_put_10, index_put_11)



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
