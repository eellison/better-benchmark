"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train_041
Pattern hash: c25aea0182b7
Shape hash: df042e9c
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
_shapes_config = "(T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(169)), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(169)), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(169)), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(169)), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(169)), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(169)), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(169)), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(169)), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(169)), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(169)), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(169)), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(169)), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(169)), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(169)), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(169)), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(169)), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(169)), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(169)), S([2401, 16]), S([2401, 16]), S([2401, 16]), S([2401, 16]), S([2401, 16]), S([2401, 16]), S([2401, 16]), S([2401, 16]), S([2401, 16]), S([2401, 16]), S([2401, 16]), S([2401, 16]), S([2401, 16]), S([2401, 16]), S([2401, 16]), S([2401, 16]), S([2401, 16]), S([2401, 16]))"

class Repro(torch.nn.Module):
    def forward(self, fma_2: "f32[512, 16, 49, 49]", arg157_1: "i64[49, 49]", fma_3: "f32[512, 16, 49, 49]", arg150_1: "i64[49, 49]", fma_4: "f32[512, 16, 49, 49]", arg143_1: "i64[49, 49]", fma_5: "f32[512, 16, 49, 49]", arg136_1: "i64[49, 49]", fma_6: "f32[512, 16, 49, 49]", arg129_1: "i64[49, 49]", fma_7: "f32[512, 16, 49, 49]", arg122_1: "i64[49, 49]", fma_8: "f32[512, 16, 49, 49]", arg115_1: "i64[49, 49]", fma_9: "f32[512, 16, 49, 49]", arg108_1: "i64[49, 49]", fma_10: "f32[512, 16, 49, 49]", arg101_1: "i64[49, 49]", fma_11: "f32[512, 16, 49, 49]", arg94_1: "i64[49, 49]", fma_12: "f32[512, 16, 49, 49]", arg87_1: "i64[49, 49]", fma_13: "f32[512, 16, 49, 49]", arg80_1: "i64[49, 49]", fma_14: "f32[512, 16, 49, 49]", arg73_1: "i64[49, 49]", fma_15: "f32[512, 16, 49, 49]", arg66_1: "i64[49, 49]", fma_16: "f32[512, 16, 49, 49]", arg59_1: "i64[49, 49]", fma_17: "f32[512, 16, 49, 49]", arg52_1: "i64[49, 49]", fma_18: "f32[512, 16, 49, 49]", arg45_1: "i64[49, 49]", fma_19: "f32[512, 16, 49, 49]", arg38_1: "i64[49, 49]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17):
        # No stacktrace found for following nodes
        sum_dim_int_list: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_2, [0], True);  fma_2 = None
        squeeze_dim: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list, 0);  sum_dim_int_list = None
        permute_default: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim, [1, 2, 0]);  squeeze_dim = None
        view_default: "f32[2401, 16]" = torch.ops.aten.view.default(permute_default, _shape_param_0);  permute_default = _shape_param_0 = None
        full_default: "f32[169, 16]" = torch.ops.aten.full.default([169, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_default_1: "i64[2401]" = torch.ops.aten.view.default(arg157_1, [-1]);  arg157_1 = None
        index_put_default: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default, [view_default_1], view_default, True);  view_default_1 = view_default = None
        sum_dim_int_list_1: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_3, [0], True);  fma_3 = None
        squeeze_dim_1: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_1, 0);  sum_dim_int_list_1 = None
        permute_default_1: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_1, [1, 2, 0]);  squeeze_dim_1 = None
        view_default_2: "f32[2401, 16]" = torch.ops.aten.view.default(permute_default_1, _shape_param_1);  permute_default_1 = _shape_param_1 = None
        view_default_3: "i64[2401]" = torch.ops.aten.view.default(arg150_1, [-1]);  arg150_1 = None
        index_put_default_1: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default, [view_default_3], view_default_2, True);  view_default_3 = view_default_2 = None
        sum_dim_int_list_2: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_4, [0], True);  fma_4 = None
        squeeze_dim_2: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_2, 0);  sum_dim_int_list_2 = None
        permute_default_2: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_2, [1, 2, 0]);  squeeze_dim_2 = None
        view_default_4: "f32[2401, 16]" = torch.ops.aten.view.default(permute_default_2, _shape_param_2);  permute_default_2 = _shape_param_2 = None
        view_default_5: "i64[2401]" = torch.ops.aten.view.default(arg143_1, [-1]);  arg143_1 = None
        index_put_default_2: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default, [view_default_5], view_default_4, True);  view_default_5 = view_default_4 = None
        sum_dim_int_list_3: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_5, [0], True);  fma_5 = None
        squeeze_dim_3: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_3, 0);  sum_dim_int_list_3 = None
        permute_default_3: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_3, [1, 2, 0]);  squeeze_dim_3 = None
        view_default_6: "f32[2401, 16]" = torch.ops.aten.view.default(permute_default_3, _shape_param_3);  permute_default_3 = _shape_param_3 = None
        view_default_7: "i64[2401]" = torch.ops.aten.view.default(arg136_1, [-1]);  arg136_1 = None
        index_put_default_3: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default, [view_default_7], view_default_6, True);  view_default_7 = view_default_6 = None
        sum_dim_int_list_4: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_6, [0], True);  fma_6 = None
        squeeze_dim_4: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_4, 0);  sum_dim_int_list_4 = None
        permute_default_4: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_4, [1, 2, 0]);  squeeze_dim_4 = None
        view_default_8: "f32[2401, 16]" = torch.ops.aten.view.default(permute_default_4, _shape_param_4);  permute_default_4 = _shape_param_4 = None
        view_default_9: "i64[2401]" = torch.ops.aten.view.default(arg129_1, [-1]);  arg129_1 = None
        index_put_default_4: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default, [view_default_9], view_default_8, True);  view_default_9 = view_default_8 = None
        sum_dim_int_list_5: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_7, [0], True);  fma_7 = None
        squeeze_dim_5: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_5, 0);  sum_dim_int_list_5 = None
        permute_default_5: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_5, [1, 2, 0]);  squeeze_dim_5 = None
        view_default_10: "f32[2401, 16]" = torch.ops.aten.view.default(permute_default_5, _shape_param_5);  permute_default_5 = _shape_param_5 = None
        view_default_11: "i64[2401]" = torch.ops.aten.view.default(arg122_1, [-1]);  arg122_1 = None
        index_put_default_5: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default, [view_default_11], view_default_10, True);  view_default_11 = view_default_10 = None
        sum_dim_int_list_6: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_8, [0], True);  fma_8 = None
        squeeze_dim_6: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_6, 0);  sum_dim_int_list_6 = None
        permute_default_6: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_6, [1, 2, 0]);  squeeze_dim_6 = None
        view_default_12: "f32[2401, 16]" = torch.ops.aten.view.default(permute_default_6, _shape_param_6);  permute_default_6 = _shape_param_6 = None
        view_default_13: "i64[2401]" = torch.ops.aten.view.default(arg115_1, [-1]);  arg115_1 = None
        index_put_default_6: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default, [view_default_13], view_default_12, True);  view_default_13 = view_default_12 = None
        sum_dim_int_list_7: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_9, [0], True);  fma_9 = None
        squeeze_dim_7: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_7, 0);  sum_dim_int_list_7 = None
        permute_default_7: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_7, [1, 2, 0]);  squeeze_dim_7 = None
        view_default_14: "f32[2401, 16]" = torch.ops.aten.view.default(permute_default_7, _shape_param_7);  permute_default_7 = _shape_param_7 = None
        view_default_15: "i64[2401]" = torch.ops.aten.view.default(arg108_1, [-1]);  arg108_1 = None
        index_put_default_7: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default, [view_default_15], view_default_14, True);  view_default_15 = view_default_14 = None
        sum_dim_int_list_8: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_10, [0], True);  fma_10 = None
        squeeze_dim_8: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_8, 0);  sum_dim_int_list_8 = None
        permute_default_8: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_8, [1, 2, 0]);  squeeze_dim_8 = None
        view_default_16: "f32[2401, 16]" = torch.ops.aten.view.default(permute_default_8, _shape_param_8);  permute_default_8 = _shape_param_8 = None
        view_default_17: "i64[2401]" = torch.ops.aten.view.default(arg101_1, [-1]);  arg101_1 = None
        index_put_default_8: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default, [view_default_17], view_default_16, True);  view_default_17 = view_default_16 = None
        sum_dim_int_list_9: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_11, [0], True);  fma_11 = None
        squeeze_dim_9: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_9, 0);  sum_dim_int_list_9 = None
        permute_default_9: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_9, [1, 2, 0]);  squeeze_dim_9 = None
        view_default_18: "f32[2401, 16]" = torch.ops.aten.view.default(permute_default_9, _shape_param_9);  permute_default_9 = _shape_param_9 = None
        view_default_19: "i64[2401]" = torch.ops.aten.view.default(arg94_1, [-1]);  arg94_1 = None
        index_put_default_9: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default, [view_default_19], view_default_18, True);  view_default_19 = view_default_18 = None
        sum_dim_int_list_10: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_12, [0], True);  fma_12 = None
        squeeze_dim_10: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_10, 0);  sum_dim_int_list_10 = None
        permute_default_10: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_10, [1, 2, 0]);  squeeze_dim_10 = None
        view_default_20: "f32[2401, 16]" = torch.ops.aten.view.default(permute_default_10, _shape_param_10);  permute_default_10 = _shape_param_10 = None
        view_default_21: "i64[2401]" = torch.ops.aten.view.default(arg87_1, [-1]);  arg87_1 = None
        index_put_default_10: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default, [view_default_21], view_default_20, True);  view_default_21 = view_default_20 = None
        sum_dim_int_list_11: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_13, [0], True);  fma_13 = None
        squeeze_dim_11: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_11, 0);  sum_dim_int_list_11 = None
        permute_default_11: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_11, [1, 2, 0]);  squeeze_dim_11 = None
        view_default_22: "f32[2401, 16]" = torch.ops.aten.view.default(permute_default_11, _shape_param_11);  permute_default_11 = _shape_param_11 = None
        view_default_23: "i64[2401]" = torch.ops.aten.view.default(arg80_1, [-1]);  arg80_1 = None
        index_put_default_11: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default, [view_default_23], view_default_22, True);  view_default_23 = view_default_22 = None
        sum_dim_int_list_12: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_14, [0], True);  fma_14 = None
        squeeze_dim_12: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_12, 0);  sum_dim_int_list_12 = None
        permute_default_12: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_12, [1, 2, 0]);  squeeze_dim_12 = None
        view_default_24: "f32[2401, 16]" = torch.ops.aten.view.default(permute_default_12, _shape_param_12);  permute_default_12 = _shape_param_12 = None
        view_default_25: "i64[2401]" = torch.ops.aten.view.default(arg73_1, [-1]);  arg73_1 = None
        index_put_default_12: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default, [view_default_25], view_default_24, True);  view_default_25 = view_default_24 = None
        sum_dim_int_list_13: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_15, [0], True);  fma_15 = None
        squeeze_dim_13: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_13, 0);  sum_dim_int_list_13 = None
        permute_default_13: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_13, [1, 2, 0]);  squeeze_dim_13 = None
        view_default_26: "f32[2401, 16]" = torch.ops.aten.view.default(permute_default_13, _shape_param_13);  permute_default_13 = _shape_param_13 = None
        view_default_27: "i64[2401]" = torch.ops.aten.view.default(arg66_1, [-1]);  arg66_1 = None
        index_put_default_13: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default, [view_default_27], view_default_26, True);  view_default_27 = view_default_26 = None
        sum_dim_int_list_14: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_16, [0], True);  fma_16 = None
        squeeze_dim_14: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_14, 0);  sum_dim_int_list_14 = None
        permute_default_14: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_14, [1, 2, 0]);  squeeze_dim_14 = None
        view_default_28: "f32[2401, 16]" = torch.ops.aten.view.default(permute_default_14, _shape_param_14);  permute_default_14 = _shape_param_14 = None
        view_default_29: "i64[2401]" = torch.ops.aten.view.default(arg59_1, [-1]);  arg59_1 = None
        index_put_default_14: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default, [view_default_29], view_default_28, True);  view_default_29 = view_default_28 = None
        sum_dim_int_list_15: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_17, [0], True);  fma_17 = None
        squeeze_dim_15: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_15, 0);  sum_dim_int_list_15 = None
        permute_default_15: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_15, [1, 2, 0]);  squeeze_dim_15 = None
        view_default_30: "f32[2401, 16]" = torch.ops.aten.view.default(permute_default_15, _shape_param_15);  permute_default_15 = _shape_param_15 = None
        view_default_31: "i64[2401]" = torch.ops.aten.view.default(arg52_1, [-1]);  arg52_1 = None
        index_put_default_15: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default, [view_default_31], view_default_30, True);  view_default_31 = view_default_30 = None
        sum_dim_int_list_16: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_18, [0], True);  fma_18 = None
        squeeze_dim_16: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_16, 0);  sum_dim_int_list_16 = None
        permute_default_16: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_16, [1, 2, 0]);  squeeze_dim_16 = None
        view_default_32: "f32[2401, 16]" = torch.ops.aten.view.default(permute_default_16, _shape_param_16);  permute_default_16 = _shape_param_16 = None
        view_default_33: "i64[2401]" = torch.ops.aten.view.default(arg45_1, [-1]);  arg45_1 = None
        index_put_default_16: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default, [view_default_33], view_default_32, True);  view_default_33 = view_default_32 = None
        sum_dim_int_list_17: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_19, [0], True);  fma_19 = None
        squeeze_dim_17: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_17, 0);  sum_dim_int_list_17 = None
        permute_default_17: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_17, [1, 2, 0]);  squeeze_dim_17 = None
        view_default_34: "f32[2401, 16]" = torch.ops.aten.view.default(permute_default_17, _shape_param_17);  permute_default_17 = _shape_param_17 = None
        view_default_35: "i64[2401]" = torch.ops.aten.view.default(arg38_1, [-1]);  arg38_1 = None
        index_put_default_17: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default, [view_default_35], view_default_34, True);  full_default = view_default_35 = view_default_34 = None
        return (index_put_default, index_put_default_1, index_put_default_10, index_put_default_3, index_put_default_12, index_put_default_2, index_put_default_14, index_put_default_4, index_put_default_13, index_put_default_16, index_put_default_6, index_put_default_15, index_put_default_5, index_put_default_8, index_put_default_17, index_put_default_7, index_put_default_9, index_put_default_11)



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
