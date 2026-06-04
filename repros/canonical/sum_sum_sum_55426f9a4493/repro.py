"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_train_001
Pattern hash: 55426f9a4493
Shape hash: d58b10fb
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([8, 512, 4096], f32), T([4096], f32), T([8, 512, 4096], f32), T([8, 512, 1], f32), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([8, 512, 4096]), S([4096, 4096]), S([4096]))"

class Repro(torch.nn.Module):
    def forward(self, add_9: "f32[8, 512, 4096]", arg110_1: "f32[8, 512, 4096]", view_18: "f32[4096, 4096]", add_23: "f32[8, 512, 4096]", arg102_1: "f32[8, 512, 4096]", view_48: "f32[4096, 4096]", add_47: "f32[8, 512, 4096]", arg94_1: "f32[8, 512, 4096]", view_78: "f32[4096, 4096]", add_71: "f32[8, 512, 4096]", arg86_1: "f32[8, 512, 4096]", view_108: "f32[4096, 4096]", add_95: "f32[8, 512, 4096]", arg78_1: "f32[8, 512, 4096]", view_138: "f32[4096, 4096]", add_119: "f32[8, 512, 4096]", arg70_1: "f32[8, 512, 4096]", view_168: "f32[4096, 4096]", add_143: "f32[8, 512, 4096]", arg62_1: "f32[8, 512, 4096]", view_198: "f32[4096, 4096]", add_167: "f32[8, 512, 4096]", arg54_1: "f32[8, 512, 4096]", view_228: "f32[4096, 4096]", add_191: "f32[8, 512, 4096]", arg46_1: "f32[8, 512, 4096]", view_258: "f32[4096, 4096]", add_215: "f32[8, 512, 4096]", arg38_1: "f32[8, 512, 4096]", view_288: "f32[4096, 4096]", add_239: "f32[8, 512, 4096]", arg30_1: "f32[8, 512, 4096]", view_318: "f32[4096, 4096]", mm_138: "f32[4096, 4096]", mul_323: "f32[8, 512, 4096]", arg9_1: "f32[4096]", arg22_1: "f32[8, 512, 4096]", arg180_1: "f32[8, 512, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13):
        # No stacktrace found for following nodes
        mul_tensor: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_9, arg110_1);  arg110_1 = None
        sum_dim_int_list: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_1: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_9, [0, 1]);  add_9 = None
        sum_dim_int_list_2: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_18, [0], True);  view_18 = None
        view_default: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_2, _shape_param_0);  sum_dim_int_list_2 = _shape_param_0 = None
        mul_tensor_1: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_23, arg102_1);  arg102_1 = None
        sum_dim_int_list_3: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_4: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_23, [0, 1]);  add_23 = None
        add_tensor: "f32[4096]" = torch.ops.aten.add.Tensor(sum_dim_int_list, sum_dim_int_list_3);  sum_dim_int_list = sum_dim_int_list_3 = None
        add_tensor_1: "f32[4096]" = torch.ops.aten.add.Tensor(sum_dim_int_list_1, sum_dim_int_list_4);  sum_dim_int_list_1 = sum_dim_int_list_4 = None
        sum_dim_int_list_5: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_48, [0], True);  view_48 = None
        view_default_1: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_5, _shape_param_1);  sum_dim_int_list_5 = _shape_param_1 = None
        add_tensor_2: "f32[4096]" = torch.ops.aten.add.Tensor(view_default, view_default_1);  view_default = view_default_1 = None
        mul_tensor_2: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_47, arg94_1);  arg94_1 = None
        sum_dim_int_list_6: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1]);  mul_tensor_2 = None
        sum_dim_int_list_7: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_47, [0, 1]);  add_47 = None
        add_tensor_3: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor, sum_dim_int_list_6);  add_tensor = sum_dim_int_list_6 = None
        add_tensor_4: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_1, sum_dim_int_list_7);  add_tensor_1 = sum_dim_int_list_7 = None
        sum_dim_int_list_8: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_78, [0], True);  view_78 = None
        view_default_2: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_8, _shape_param_2);  sum_dim_int_list_8 = _shape_param_2 = None
        add_tensor_5: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_2, view_default_2);  add_tensor_2 = view_default_2 = None
        mul_tensor_3: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_71, arg86_1);  arg86_1 = None
        sum_dim_int_list_9: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1]);  mul_tensor_3 = None
        sum_dim_int_list_10: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_71, [0, 1]);  add_71 = None
        add_tensor_6: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_3, sum_dim_int_list_9);  add_tensor_3 = sum_dim_int_list_9 = None
        add_tensor_7: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_4, sum_dim_int_list_10);  add_tensor_4 = sum_dim_int_list_10 = None
        sum_dim_int_list_11: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_108, [0], True);  view_108 = None
        view_default_3: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_11, _shape_param_3);  sum_dim_int_list_11 = _shape_param_3 = None
        add_tensor_8: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_5, view_default_3);  add_tensor_5 = view_default_3 = None
        mul_tensor_4: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_95, arg78_1);  arg78_1 = None
        sum_dim_int_list_12: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1]);  mul_tensor_4 = None
        sum_dim_int_list_13: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_95, [0, 1]);  add_95 = None
        add_tensor_9: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_6, sum_dim_int_list_12);  add_tensor_6 = sum_dim_int_list_12 = None
        add_tensor_10: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_7, sum_dim_int_list_13);  add_tensor_7 = sum_dim_int_list_13 = None
        sum_dim_int_list_14: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_138, [0], True);  view_138 = None
        view_default_4: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_14, _shape_param_4);  sum_dim_int_list_14 = _shape_param_4 = None
        add_tensor_11: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_8, view_default_4);  add_tensor_8 = view_default_4 = None
        mul_tensor_5: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_119, arg70_1);  arg70_1 = None
        sum_dim_int_list_15: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_16: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_119, [0, 1]);  add_119 = None
        add_tensor_12: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_9, sum_dim_int_list_15);  add_tensor_9 = sum_dim_int_list_15 = None
        add_tensor_13: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_10, sum_dim_int_list_16);  add_tensor_10 = sum_dim_int_list_16 = None
        sum_dim_int_list_17: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_168, [0], True);  view_168 = None
        view_default_5: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_17, _shape_param_5);  sum_dim_int_list_17 = _shape_param_5 = None
        add_tensor_14: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_11, view_default_5);  add_tensor_11 = view_default_5 = None
        mul_tensor_6: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_143, arg62_1);  arg62_1 = None
        sum_dim_int_list_18: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_19: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_143, [0, 1]);  add_143 = None
        add_tensor_15: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_12, sum_dim_int_list_18);  add_tensor_12 = sum_dim_int_list_18 = None
        add_tensor_16: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_13, sum_dim_int_list_19);  add_tensor_13 = sum_dim_int_list_19 = None
        sum_dim_int_list_20: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_198, [0], True);  view_198 = None
        view_default_6: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_20, _shape_param_6);  sum_dim_int_list_20 = _shape_param_6 = None
        add_tensor_17: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_14, view_default_6);  add_tensor_14 = view_default_6 = None
        mul_tensor_7: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_167, arg54_1);  arg54_1 = None
        sum_dim_int_list_21: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_22: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_167, [0, 1]);  add_167 = None
        add_tensor_18: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_15, sum_dim_int_list_21);  add_tensor_15 = sum_dim_int_list_21 = None
        add_tensor_19: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_16, sum_dim_int_list_22);  add_tensor_16 = sum_dim_int_list_22 = None
        sum_dim_int_list_23: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_228, [0], True);  view_228 = None
        view_default_7: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_23, _shape_param_7);  sum_dim_int_list_23 = _shape_param_7 = None
        add_tensor_20: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_17, view_default_7);  add_tensor_17 = view_default_7 = None
        mul_tensor_8: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_191, arg46_1);  arg46_1 = None
        sum_dim_int_list_24: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_25: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_191, [0, 1]);  add_191 = None
        add_tensor_21: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_18, sum_dim_int_list_24);  add_tensor_18 = sum_dim_int_list_24 = None
        add_tensor_22: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_19, sum_dim_int_list_25);  add_tensor_19 = sum_dim_int_list_25 = None
        sum_dim_int_list_26: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_258, [0], True);  view_258 = None
        view_default_8: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_26, _shape_param_8);  sum_dim_int_list_26 = _shape_param_8 = None
        add_tensor_23: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_20, view_default_8);  add_tensor_20 = view_default_8 = None
        mul_tensor_9: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_215, arg38_1);  arg38_1 = None
        sum_dim_int_list_27: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1]);  mul_tensor_9 = None
        sum_dim_int_list_28: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_215, [0, 1]);  add_215 = None
        add_tensor_24: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_21, sum_dim_int_list_27);  add_tensor_21 = sum_dim_int_list_27 = None
        add_tensor_25: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_22, sum_dim_int_list_28);  add_tensor_22 = sum_dim_int_list_28 = None
        sum_dim_int_list_29: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_288, [0], True);  view_288 = None
        view_default_9: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_29, _shape_param_9);  sum_dim_int_list_29 = _shape_param_9 = None
        add_tensor_26: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_23, view_default_9);  add_tensor_23 = view_default_9 = None
        mul_tensor_10: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_239, arg30_1);  arg30_1 = None
        sum_dim_int_list_30: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        sum_dim_int_list_31: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_239, [0, 1]);  add_239 = None
        add_tensor_27: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_24, sum_dim_int_list_30);  add_tensor_24 = sum_dim_int_list_30 = None
        add_tensor_28: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_25, sum_dim_int_list_31);  add_tensor_25 = sum_dim_int_list_31 = None
        sum_dim_int_list_32: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_318, [0], True);  view_318 = None
        view_default_10: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_32, _shape_param_10);  sum_dim_int_list_32 = _shape_param_10 = None
        add_tensor_29: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_26, view_default_10);  add_tensor_26 = view_default_10 = None
        view_default_11: "f32[8, 512, 4096]" = torch.ops.aten.view.default(mm_138, _shape_param_11);  mm_138 = _shape_param_11 = None
        add_tensor_30: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_323, view_default_11);  mul_323 = view_default_11 = None
        mul_tensor_11: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_tensor_30, arg9_1);  arg9_1 = None
        mul_tensor_12: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor_11, 4096)
        sum_dim_int_list_33: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [2], True)
        mul_tensor_13: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor_11, arg22_1);  mul_tensor_11 = None
        sum_dim_int_list_34: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [2], True);  mul_tensor_13 = None
        mul_tensor_14: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg22_1, sum_dim_int_list_34);  sum_dim_int_list_34 = None
        sub_tensor: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_tensor_12, sum_dim_int_list_33);  mul_tensor_12 = sum_dim_int_list_33 = None
        sub_tensor_1: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_14);  sub_tensor = mul_tensor_14 = None
        mul_tensor_15: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg180_1, sub_tensor_1);  arg180_1 = sub_tensor_1 = None
        mul_tensor_16: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_tensor_30, arg22_1);  arg22_1 = None
        sum_dim_int_list_35: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1]);  mul_tensor_16 = None
        sum_dim_int_list_36: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_tensor_30, [0, 1]);  add_tensor_30 = None
        add_tensor_31: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_27, sum_dim_int_list_35);  add_tensor_27 = sum_dim_int_list_35 = None
        add_tensor_32: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_28, sum_dim_int_list_36);  add_tensor_28 = sum_dim_int_list_36 = None
        view_default_12: "f32[4096, 4096]" = torch.ops.aten.view.default(mul_tensor_15, _shape_param_12);  mul_tensor_15 = _shape_param_12 = None
        permute_default: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_default_12, [1, 0])
        sum_dim_int_list_37: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_default_12, [0], True);  view_default_12 = None
        view_default_13: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_37, _shape_param_13);  sum_dim_int_list_37 = _shape_param_13 = None
        add_tensor_33: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_29, view_default_13);  add_tensor_29 = view_default_13 = None
        return (add_tensor_31, add_tensor_32, permute_default, add_tensor_33)

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
