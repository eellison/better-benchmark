"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_train_001
Pattern hash: d06bf12e10d0
Shape hash: dc4391c3
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096], f32), T([8, 512, 4096], f32), T([8, 512, 1], f32), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([8, 512, 4096]), S([8, 512, 4096]), S([8, 512, 4096]), S([4096, 4096]), S([4096]))"

class Repro(torch.nn.Module):
    def forward(self, view_10: "f32[8, 512, 4096]", arg114_1: "f32[8, 512, 4096]", view_11: "f32[4096, 4096]", add_12: "f32[8, 512, 4096]", arg106_1: "f32[8, 512, 4096]", view_41: "f32[4096, 4096]", add_36: "f32[8, 512, 4096]", arg98_1: "f32[8, 512, 4096]", view_71: "f32[4096, 4096]", add_60: "f32[8, 512, 4096]", arg90_1: "f32[8, 512, 4096]", view_101: "f32[4096, 4096]", add_84: "f32[8, 512, 4096]", arg82_1: "f32[8, 512, 4096]", view_131: "f32[4096, 4096]", add_108: "f32[8, 512, 4096]", arg74_1: "f32[8, 512, 4096]", view_161: "f32[4096, 4096]", add_132: "f32[8, 512, 4096]", arg66_1: "f32[8, 512, 4096]", view_191: "f32[4096, 4096]", add_156: "f32[8, 512, 4096]", arg58_1: "f32[8, 512, 4096]", view_221: "f32[4096, 4096]", add_180: "f32[8, 512, 4096]", arg50_1: "f32[8, 512, 4096]", view_251: "f32[4096, 4096]", add_204: "f32[8, 512, 4096]", arg42_1: "f32[8, 512, 4096]", view_281: "f32[4096, 4096]", add_228: "f32[8, 512, 4096]", arg34_1: "f32[8, 512, 4096]", view_311: "f32[4096, 4096]", mm_130: "f32[4096, 4096]", mul_314: "f32[8, 512, 4096]", mm_132: "f32[4096, 4096]", mm_134: "f32[4096, 4096]", arg12_1: "f32[4096]", arg26_1: "f32[8, 512, 4096]", arg179_1: "f32[8, 512, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15):
        # No stacktrace found for following nodes
        mul_tensor: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(view_10, arg114_1);  arg114_1 = None
        sum_dim_int_list: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_1: "f32[4096]" = torch.ops.aten.sum.dim_IntList(view_10, [0, 1]);  view_10 = None
        sum_dim_int_list_2: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_11, [0], True);  view_11 = None
        view_default: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_2, _shape_param_0);  sum_dim_int_list_2 = _shape_param_0 = None
        mul_tensor_1: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_12, arg106_1);  arg106_1 = None
        sum_dim_int_list_3: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_4: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_12, [0, 1]);  add_12 = None
        add_tensor: "f32[4096]" = torch.ops.aten.add.Tensor(sum_dim_int_list, sum_dim_int_list_3);  sum_dim_int_list = sum_dim_int_list_3 = None
        add_tensor_1: "f32[4096]" = torch.ops.aten.add.Tensor(sum_dim_int_list_1, sum_dim_int_list_4);  sum_dim_int_list_1 = sum_dim_int_list_4 = None
        sum_dim_int_list_5: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_41, [0], True);  view_41 = None
        view_default_1: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_5, _shape_param_1);  sum_dim_int_list_5 = _shape_param_1 = None
        add_tensor_2: "f32[4096]" = torch.ops.aten.add.Tensor(view_default, view_default_1);  view_default = view_default_1 = None
        mul_tensor_2: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_36, arg98_1);  arg98_1 = None
        sum_dim_int_list_6: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1]);  mul_tensor_2 = None
        sum_dim_int_list_7: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_36, [0, 1]);  add_36 = None
        add_tensor_3: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor, sum_dim_int_list_6);  add_tensor = sum_dim_int_list_6 = None
        add_tensor_4: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_1, sum_dim_int_list_7);  add_tensor_1 = sum_dim_int_list_7 = None
        sum_dim_int_list_8: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_71, [0], True);  view_71 = None
        view_default_2: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_8, _shape_param_2);  sum_dim_int_list_8 = _shape_param_2 = None
        add_tensor_5: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_2, view_default_2);  add_tensor_2 = view_default_2 = None
        mul_tensor_3: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_60, arg90_1);  arg90_1 = None
        sum_dim_int_list_9: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1]);  mul_tensor_3 = None
        sum_dim_int_list_10: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_60, [0, 1]);  add_60 = None
        add_tensor_6: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_3, sum_dim_int_list_9);  add_tensor_3 = sum_dim_int_list_9 = None
        add_tensor_7: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_4, sum_dim_int_list_10);  add_tensor_4 = sum_dim_int_list_10 = None
        sum_dim_int_list_11: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_101, [0], True);  view_101 = None
        view_default_3: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_11, _shape_param_3);  sum_dim_int_list_11 = _shape_param_3 = None
        add_tensor_8: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_5, view_default_3);  add_tensor_5 = view_default_3 = None
        mul_tensor_4: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_84, arg82_1);  arg82_1 = None
        sum_dim_int_list_12: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1]);  mul_tensor_4 = None
        sum_dim_int_list_13: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_84, [0, 1]);  add_84 = None
        add_tensor_9: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_6, sum_dim_int_list_12);  add_tensor_6 = sum_dim_int_list_12 = None
        add_tensor_10: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_7, sum_dim_int_list_13);  add_tensor_7 = sum_dim_int_list_13 = None
        sum_dim_int_list_14: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_131, [0], True);  view_131 = None
        view_default_4: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_14, _shape_param_4);  sum_dim_int_list_14 = _shape_param_4 = None
        add_tensor_11: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_8, view_default_4);  add_tensor_8 = view_default_4 = None
        mul_tensor_5: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_108, arg74_1);  arg74_1 = None
        sum_dim_int_list_15: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_16: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_108, [0, 1]);  add_108 = None
        add_tensor_12: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_9, sum_dim_int_list_15);  add_tensor_9 = sum_dim_int_list_15 = None
        add_tensor_13: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_10, sum_dim_int_list_16);  add_tensor_10 = sum_dim_int_list_16 = None
        sum_dim_int_list_17: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_161, [0], True);  view_161 = None
        view_default_5: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_17, _shape_param_5);  sum_dim_int_list_17 = _shape_param_5 = None
        add_tensor_14: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_11, view_default_5);  add_tensor_11 = view_default_5 = None
        mul_tensor_6: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_132, arg66_1);  arg66_1 = None
        sum_dim_int_list_18: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_19: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_132, [0, 1]);  add_132 = None
        add_tensor_15: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_12, sum_dim_int_list_18);  add_tensor_12 = sum_dim_int_list_18 = None
        add_tensor_16: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_13, sum_dim_int_list_19);  add_tensor_13 = sum_dim_int_list_19 = None
        sum_dim_int_list_20: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_191, [0], True);  view_191 = None
        view_default_6: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_20, _shape_param_6);  sum_dim_int_list_20 = _shape_param_6 = None
        add_tensor_17: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_14, view_default_6);  add_tensor_14 = view_default_6 = None
        mul_tensor_7: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_156, arg58_1);  arg58_1 = None
        sum_dim_int_list_21: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_22: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_156, [0, 1]);  add_156 = None
        add_tensor_18: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_15, sum_dim_int_list_21);  add_tensor_15 = sum_dim_int_list_21 = None
        add_tensor_19: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_16, sum_dim_int_list_22);  add_tensor_16 = sum_dim_int_list_22 = None
        sum_dim_int_list_23: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_221, [0], True);  view_221 = None
        view_default_7: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_23, _shape_param_7);  sum_dim_int_list_23 = _shape_param_7 = None
        add_tensor_20: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_17, view_default_7);  add_tensor_17 = view_default_7 = None
        mul_tensor_8: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_180, arg50_1);  arg50_1 = None
        sum_dim_int_list_24: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_25: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_180, [0, 1]);  add_180 = None
        add_tensor_21: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_18, sum_dim_int_list_24);  add_tensor_18 = sum_dim_int_list_24 = None
        add_tensor_22: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_19, sum_dim_int_list_25);  add_tensor_19 = sum_dim_int_list_25 = None
        sum_dim_int_list_26: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_251, [0], True);  view_251 = None
        view_default_8: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_26, _shape_param_8);  sum_dim_int_list_26 = _shape_param_8 = None
        add_tensor_23: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_20, view_default_8);  add_tensor_20 = view_default_8 = None
        mul_tensor_9: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_204, arg42_1);  arg42_1 = None
        sum_dim_int_list_27: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1]);  mul_tensor_9 = None
        sum_dim_int_list_28: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_204, [0, 1]);  add_204 = None
        add_tensor_24: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_21, sum_dim_int_list_27);  add_tensor_21 = sum_dim_int_list_27 = None
        add_tensor_25: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_22, sum_dim_int_list_28);  add_tensor_22 = sum_dim_int_list_28 = None
        sum_dim_int_list_29: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_281, [0], True);  view_281 = None
        view_default_9: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_29, _shape_param_9);  sum_dim_int_list_29 = _shape_param_9 = None
        add_tensor_26: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_23, view_default_9);  add_tensor_23 = view_default_9 = None
        mul_tensor_10: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_228, arg34_1);  arg34_1 = None
        sum_dim_int_list_30: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        sum_dim_int_list_31: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_228, [0, 1]);  add_228 = None
        add_tensor_27: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_24, sum_dim_int_list_30);  add_tensor_24 = sum_dim_int_list_30 = None
        add_tensor_28: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_25, sum_dim_int_list_31);  add_tensor_25 = sum_dim_int_list_31 = None
        sum_dim_int_list_32: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_311, [0], True);  view_311 = None
        view_default_10: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_32, _shape_param_10);  sum_dim_int_list_32 = _shape_param_10 = None
        add_tensor_29: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_26, view_default_10);  add_tensor_26 = view_default_10 = None
        view_default_11: "f32[8, 512, 4096]" = torch.ops.aten.view.default(mm_130, _shape_param_11);  mm_130 = _shape_param_11 = None
        add_tensor_30: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_314, view_default_11);  mul_314 = view_default_11 = None
        view_default_12: "f32[8, 512, 4096]" = torch.ops.aten.view.default(mm_132, _shape_param_12);  mm_132 = _shape_param_12 = None
        add_tensor_31: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_tensor_30, view_default_12);  add_tensor_30 = view_default_12 = None
        view_default_13: "f32[8, 512, 4096]" = torch.ops.aten.view.default(mm_134, _shape_param_13);  mm_134 = _shape_param_13 = None
        add_tensor_32: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_tensor_31, view_default_13);  add_tensor_31 = view_default_13 = None
        mul_tensor_11: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_tensor_32, arg12_1);  arg12_1 = None
        mul_tensor_12: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor_11, 4096)
        sum_dim_int_list_33: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [2], True)
        mul_tensor_13: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor_11, arg26_1);  mul_tensor_11 = None
        sum_dim_int_list_34: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [2], True);  mul_tensor_13 = None
        mul_tensor_14: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg26_1, sum_dim_int_list_34);  sum_dim_int_list_34 = None
        sub_tensor: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_tensor_12, sum_dim_int_list_33);  mul_tensor_12 = sum_dim_int_list_33 = None
        sub_tensor_1: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_14);  sub_tensor = mul_tensor_14 = None
        mul_tensor_15: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg179_1, sub_tensor_1);  arg179_1 = sub_tensor_1 = None
        mul_tensor_16: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_tensor_32, arg26_1);  arg26_1 = None
        sum_dim_int_list_35: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1]);  mul_tensor_16 = None
        sum_dim_int_list_36: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_tensor_32, [0, 1]);  add_tensor_32 = None
        add_tensor_33: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_27, sum_dim_int_list_35);  add_tensor_27 = sum_dim_int_list_35 = None
        add_tensor_34: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_28, sum_dim_int_list_36);  add_tensor_28 = sum_dim_int_list_36 = None
        view_default_14: "f32[4096, 4096]" = torch.ops.aten.view.default(mul_tensor_15, _shape_param_14);  mul_tensor_15 = _shape_param_14 = None
        permute_default: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_default_14, [1, 0])
        sum_dim_int_list_37: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_default_14, [0], True);  view_default_14 = None
        view_default_15: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_37, _shape_param_15);  sum_dim_int_list_37 = _shape_param_15 = None
        add_tensor_35: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_29, view_default_15);  add_tensor_29 = view_default_15 = None
        return (add_tensor_33, add_tensor_34, permute_default, add_tensor_35)

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
