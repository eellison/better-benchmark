"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_train_001
Pattern hash: 692fd552937e
Shape hash: 932b9c42
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
_shapes_config = "(T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32), T([8, 512, 4096], f32))"

class Repro(torch.nn.Module):
    def forward(self, view_10: "f32[8, 512, 4096]", arg114_1: "f32[8, 512, 4096]", add_12: "f32[8, 512, 4096]", arg106_1: "f32[8, 512, 4096]", add_36: "f32[8, 512, 4096]", arg98_1: "f32[8, 512, 4096]", add_60: "f32[8, 512, 4096]", arg90_1: "f32[8, 512, 4096]", add_84: "f32[8, 512, 4096]", arg82_1: "f32[8, 512, 4096]", add_108: "f32[8, 512, 4096]", arg74_1: "f32[8, 512, 4096]", add_132: "f32[8, 512, 4096]", arg66_1: "f32[8, 512, 4096]", add_156: "f32[8, 512, 4096]", arg58_1: "f32[8, 512, 4096]", add_180: "f32[8, 512, 4096]", arg50_1: "f32[8, 512, 4096]", add_204: "f32[8, 512, 4096]", arg42_1: "f32[8, 512, 4096]", add_228: "f32[8, 512, 4096]", arg34_1: "f32[8, 512, 4096]", add_252: "f32[8, 512, 4096]", arg26_1: "f32[8, 512, 4096]"):
        # No stacktrace found for following nodes
        mul_tensor: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(view_10, arg114_1);  view_10 = arg114_1 = None
        sum_dim_int_list: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        mul_tensor_1: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_12, arg106_1);  add_12 = arg106_1 = None
        sum_dim_int_list_1: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        add_tensor: "f32[4096]" = torch.ops.aten.add.Tensor(sum_dim_int_list, sum_dim_int_list_1);  sum_dim_int_list = sum_dim_int_list_1 = None
        mul_tensor_2: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_36, arg98_1);  add_36 = arg98_1 = None
        sum_dim_int_list_2: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1]);  mul_tensor_2 = None
        add_tensor_1: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor, sum_dim_int_list_2);  add_tensor = sum_dim_int_list_2 = None
        mul_tensor_3: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_60, arg90_1);  add_60 = arg90_1 = None
        sum_dim_int_list_3: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1]);  mul_tensor_3 = None
        add_tensor_2: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_1, sum_dim_int_list_3);  add_tensor_1 = sum_dim_int_list_3 = None
        mul_tensor_4: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_84, arg82_1);  add_84 = arg82_1 = None
        sum_dim_int_list_4: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1]);  mul_tensor_4 = None
        add_tensor_3: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_2, sum_dim_int_list_4);  add_tensor_2 = sum_dim_int_list_4 = None
        mul_tensor_5: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_108, arg74_1);  add_108 = arg74_1 = None
        sum_dim_int_list_5: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        add_tensor_4: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_3, sum_dim_int_list_5);  add_tensor_3 = sum_dim_int_list_5 = None
        mul_tensor_6: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_132, arg66_1);  add_132 = arg66_1 = None
        sum_dim_int_list_6: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        add_tensor_5: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_4, sum_dim_int_list_6);  add_tensor_4 = sum_dim_int_list_6 = None
        mul_tensor_7: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_156, arg58_1);  add_156 = arg58_1 = None
        sum_dim_int_list_7: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        add_tensor_6: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_5, sum_dim_int_list_7);  add_tensor_5 = sum_dim_int_list_7 = None
        mul_tensor_8: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_180, arg50_1);  add_180 = arg50_1 = None
        sum_dim_int_list_8: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        add_tensor_7: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_6, sum_dim_int_list_8);  add_tensor_6 = sum_dim_int_list_8 = None
        mul_tensor_9: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_204, arg42_1);  add_204 = arg42_1 = None
        sum_dim_int_list_9: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1]);  mul_tensor_9 = None
        add_tensor_8: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_7, sum_dim_int_list_9);  add_tensor_7 = sum_dim_int_list_9 = None
        mul_tensor_10: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_228, arg34_1);  add_228 = arg34_1 = None
        sum_dim_int_list_10: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        add_tensor_9: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_8, sum_dim_int_list_10);  add_tensor_8 = sum_dim_int_list_10 = None
        mul_tensor_11: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_252, arg26_1);  add_252 = arg26_1 = None
        sum_dim_int_list_11: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1]);  mul_tensor_11 = None
        add_tensor_10: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_9, sum_dim_int_list_11);  add_tensor_9 = sum_dim_int_list_11 = None
        return add_tensor_10



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
