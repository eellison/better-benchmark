"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_train_001
Pattern hash: 7ebfef460635
Shape hash: 485c0134
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), S([38809, 12]), S([38809, 12]), S([38809, 12]), S([38809, 12]), S([38809, 12]), S([38809, 12]), S([38809, 12]), S([38809, 12]), S([38809, 12]), S([38809, 12]), S([38809, 12]), S([38809, 12]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_3: "f32[128, 12, 197, 197]", arg104_1: "i64[197, 197]", getitem_7: "f32[128, 12, 197, 197]", arg95_1: "i64[197, 197]", getitem_11: "f32[128, 12, 197, 197]", arg86_1: "i64[197, 197]", getitem_15: "f32[128, 12, 197, 197]", arg77_1: "i64[197, 197]", getitem_19: "f32[128, 12, 197, 197]", arg68_1: "i64[197, 197]", getitem_23: "f32[128, 12, 197, 197]", arg59_1: "i64[197, 197]", getitem_27: "f32[128, 12, 197, 197]", arg50_1: "i64[197, 197]", getitem_31: "f32[128, 12, 197, 197]", arg41_1: "i64[197, 197]", getitem_35: "f32[128, 12, 197, 197]", arg32_1: "i64[197, 197]", getitem_39: "f32[128, 12, 197, 197]", arg23_1: "i64[197, 197]", getitem_43: "f32[128, 12, 197, 197]", arg14_1: "i64[197, 197]", getitem_47: "f32[128, 12, 197, 197]", arg5_1: "i64[197, 197]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11):
        # No stacktrace found for following nodes
        sum_dim_int_list: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_3, [0], True);  getitem_3 = None
        full_default: "f32[1, 12, 197, 200]" = torch.ops.aten.full.default([1, 12, 197, 200], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list, -1, 0, 197);  sum_dim_int_list = None
        constant_pad_nd_default: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default, [0, -3]);  slice_scatter_default = None
        squeeze_dim: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default, 0);  constant_pad_nd_default = None
        permute_default: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim, [1, 2, 0]);  squeeze_dim = None
        view_default: "f32[38809, 12]" = torch.ops.aten.view.default(permute_default, _shape_param_0);  permute_default = _shape_param_0 = None
        full_default_1: "f32[732, 12]" = torch.ops.aten.full.default([732, 12], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_default_1: "i64[38809]" = torch.ops.aten.view.default(arg104_1, [-1]);  arg104_1 = None
        index_put_default: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [view_default_1], view_default, True);  view_default_1 = view_default = None
        sum_dim_int_list_1: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_7, [0], True);  getitem_7 = None
        slice_scatter_default_1: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list_1, -1, 0, 197);  sum_dim_int_list_1 = None
        constant_pad_nd_default_1: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default_1, [0, -3]);  slice_scatter_default_1 = None
        squeeze_dim_1: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default_1, 0);  constant_pad_nd_default_1 = None
        permute_default_1: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim_1, [1, 2, 0]);  squeeze_dim_1 = None
        view_default_2: "f32[38809, 12]" = torch.ops.aten.view.default(permute_default_1, _shape_param_1);  permute_default_1 = _shape_param_1 = None
        view_default_3: "i64[38809]" = torch.ops.aten.view.default(arg95_1, [-1]);  arg95_1 = None
        index_put_default_1: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [view_default_3], view_default_2, True);  view_default_3 = view_default_2 = None
        sum_dim_int_list_2: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_11, [0], True);  getitem_11 = None
        slice_scatter_default_2: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list_2, -1, 0, 197);  sum_dim_int_list_2 = None
        constant_pad_nd_default_2: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default_2, [0, -3]);  slice_scatter_default_2 = None
        squeeze_dim_2: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default_2, 0);  constant_pad_nd_default_2 = None
        permute_default_2: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim_2, [1, 2, 0]);  squeeze_dim_2 = None
        view_default_4: "f32[38809, 12]" = torch.ops.aten.view.default(permute_default_2, _shape_param_2);  permute_default_2 = _shape_param_2 = None
        view_default_5: "i64[38809]" = torch.ops.aten.view.default(arg86_1, [-1]);  arg86_1 = None
        index_put_default_2: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [view_default_5], view_default_4, True);  view_default_5 = view_default_4 = None
        sum_dim_int_list_3: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_15, [0], True);  getitem_15 = None
        slice_scatter_default_3: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list_3, -1, 0, 197);  sum_dim_int_list_3 = None
        constant_pad_nd_default_3: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default_3, [0, -3]);  slice_scatter_default_3 = None
        squeeze_dim_3: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default_3, 0);  constant_pad_nd_default_3 = None
        permute_default_3: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim_3, [1, 2, 0]);  squeeze_dim_3 = None
        view_default_6: "f32[38809, 12]" = torch.ops.aten.view.default(permute_default_3, _shape_param_3);  permute_default_3 = _shape_param_3 = None
        view_default_7: "i64[38809]" = torch.ops.aten.view.default(arg77_1, [-1]);  arg77_1 = None
        index_put_default_3: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [view_default_7], view_default_6, True);  view_default_7 = view_default_6 = None
        sum_dim_int_list_4: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_19, [0], True);  getitem_19 = None
        slice_scatter_default_4: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list_4, -1, 0, 197);  sum_dim_int_list_4 = None
        constant_pad_nd_default_4: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default_4, [0, -3]);  slice_scatter_default_4 = None
        squeeze_dim_4: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default_4, 0);  constant_pad_nd_default_4 = None
        permute_default_4: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim_4, [1, 2, 0]);  squeeze_dim_4 = None
        view_default_8: "f32[38809, 12]" = torch.ops.aten.view.default(permute_default_4, _shape_param_4);  permute_default_4 = _shape_param_4 = None
        view_default_9: "i64[38809]" = torch.ops.aten.view.default(arg68_1, [-1]);  arg68_1 = None
        index_put_default_4: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [view_default_9], view_default_8, True);  view_default_9 = view_default_8 = None
        sum_dim_int_list_5: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_23, [0], True);  getitem_23 = None
        slice_scatter_default_5: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list_5, -1, 0, 197);  sum_dim_int_list_5 = None
        constant_pad_nd_default_5: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default_5, [0, -3]);  slice_scatter_default_5 = None
        squeeze_dim_5: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default_5, 0);  constant_pad_nd_default_5 = None
        permute_default_5: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim_5, [1, 2, 0]);  squeeze_dim_5 = None
        view_default_10: "f32[38809, 12]" = torch.ops.aten.view.default(permute_default_5, _shape_param_5);  permute_default_5 = _shape_param_5 = None
        view_default_11: "i64[38809]" = torch.ops.aten.view.default(arg59_1, [-1]);  arg59_1 = None
        index_put_default_5: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [view_default_11], view_default_10, True);  view_default_11 = view_default_10 = None
        sum_dim_int_list_6: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_27, [0], True);  getitem_27 = None
        slice_scatter_default_6: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list_6, -1, 0, 197);  sum_dim_int_list_6 = None
        constant_pad_nd_default_6: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default_6, [0, -3]);  slice_scatter_default_6 = None
        squeeze_dim_6: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default_6, 0);  constant_pad_nd_default_6 = None
        permute_default_6: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim_6, [1, 2, 0]);  squeeze_dim_6 = None
        view_default_12: "f32[38809, 12]" = torch.ops.aten.view.default(permute_default_6, _shape_param_6);  permute_default_6 = _shape_param_6 = None
        view_default_13: "i64[38809]" = torch.ops.aten.view.default(arg50_1, [-1]);  arg50_1 = None
        index_put_default_6: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [view_default_13], view_default_12, True);  view_default_13 = view_default_12 = None
        sum_dim_int_list_7: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_31, [0], True);  getitem_31 = None
        slice_scatter_default_7: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list_7, -1, 0, 197);  sum_dim_int_list_7 = None
        constant_pad_nd_default_7: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default_7, [0, -3]);  slice_scatter_default_7 = None
        squeeze_dim_7: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default_7, 0);  constant_pad_nd_default_7 = None
        permute_default_7: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim_7, [1, 2, 0]);  squeeze_dim_7 = None
        view_default_14: "f32[38809, 12]" = torch.ops.aten.view.default(permute_default_7, _shape_param_7);  permute_default_7 = _shape_param_7 = None
        view_default_15: "i64[38809]" = torch.ops.aten.view.default(arg41_1, [-1]);  arg41_1 = None
        index_put_default_7: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [view_default_15], view_default_14, True);  view_default_15 = view_default_14 = None
        sum_dim_int_list_8: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_35, [0], True);  getitem_35 = None
        slice_scatter_default_8: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list_8, -1, 0, 197);  sum_dim_int_list_8 = None
        constant_pad_nd_default_8: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default_8, [0, -3]);  slice_scatter_default_8 = None
        squeeze_dim_8: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default_8, 0);  constant_pad_nd_default_8 = None
        permute_default_8: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim_8, [1, 2, 0]);  squeeze_dim_8 = None
        view_default_16: "f32[38809, 12]" = torch.ops.aten.view.default(permute_default_8, _shape_param_8);  permute_default_8 = _shape_param_8 = None
        view_default_17: "i64[38809]" = torch.ops.aten.view.default(arg32_1, [-1]);  arg32_1 = None
        index_put_default_8: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [view_default_17], view_default_16, True);  view_default_17 = view_default_16 = None
        sum_dim_int_list_9: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_39, [0], True);  getitem_39 = None
        slice_scatter_default_9: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list_9, -1, 0, 197);  sum_dim_int_list_9 = None
        constant_pad_nd_default_9: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default_9, [0, -3]);  slice_scatter_default_9 = None
        squeeze_dim_9: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default_9, 0);  constant_pad_nd_default_9 = None
        permute_default_9: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim_9, [1, 2, 0]);  squeeze_dim_9 = None
        view_default_18: "f32[38809, 12]" = torch.ops.aten.view.default(permute_default_9, _shape_param_9);  permute_default_9 = _shape_param_9 = None
        view_default_19: "i64[38809]" = torch.ops.aten.view.default(arg23_1, [-1]);  arg23_1 = None
        index_put_default_9: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [view_default_19], view_default_18, True);  view_default_19 = view_default_18 = None
        sum_dim_int_list_10: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_43, [0], True);  getitem_43 = None
        slice_scatter_default_10: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list_10, -1, 0, 197);  sum_dim_int_list_10 = None
        constant_pad_nd_default_10: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default_10, [0, -3]);  slice_scatter_default_10 = None
        squeeze_dim_10: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default_10, 0);  constant_pad_nd_default_10 = None
        permute_default_10: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim_10, [1, 2, 0]);  squeeze_dim_10 = None
        view_default_20: "f32[38809, 12]" = torch.ops.aten.view.default(permute_default_10, _shape_param_10);  permute_default_10 = _shape_param_10 = None
        view_default_21: "i64[38809]" = torch.ops.aten.view.default(arg14_1, [-1]);  arg14_1 = None
        index_put_default_10: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [view_default_21], view_default_20, True);  view_default_21 = view_default_20 = None
        sum_dim_int_list_11: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_47, [0], True);  getitem_47 = None
        slice_scatter_default_11: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list_11, -1, 0, 197);  full_default = sum_dim_int_list_11 = None
        constant_pad_nd_default_11: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default_11, [0, -3]);  slice_scatter_default_11 = None
        squeeze_dim_11: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default_11, 0);  constant_pad_nd_default_11 = None
        permute_default_11: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim_11, [1, 2, 0]);  squeeze_dim_11 = None
        view_default_22: "f32[38809, 12]" = torch.ops.aten.view.default(permute_default_11, _shape_param_11);  permute_default_11 = _shape_param_11 = None
        view_default_23: "i64[38809]" = torch.ops.aten.view.default(arg5_1, [-1]);  arg5_1 = None
        index_put_default_11: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [view_default_23], view_default_22, True);  full_default_1 = view_default_23 = view_default_22 = None
        return (index_put_default, index_put_default_1, index_put_default_2, index_put_default_3, index_put_default_4, index_put_default_5, index_put_default_6, index_put_default_7, index_put_default_8, index_put_default_9, index_put_default_10, index_put_default_11)

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
