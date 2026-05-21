"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_train_001
Pattern hash: c19430384454
Shape hash: 35949c2c
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
_shapes_config = "(T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]), S([4096]))"

class Repro(torch.nn.Module):
    def forward(self, view_11: "f32[4096, 4096]", view_41: "f32[4096, 4096]", view_71: "f32[4096, 4096]", view_101: "f32[4096, 4096]", view_131: "f32[4096, 4096]", view_161: "f32[4096, 4096]", view_191: "f32[4096, 4096]", view_221: "f32[4096, 4096]", view_251: "f32[4096, 4096]", view_281: "f32[4096, 4096]", view_311: "f32[4096, 4096]", view_341: "f32[4096, 4096]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11):
        # No stacktrace found for following nodes
        sum_dim_int_list: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_11, [0], True);  view_11 = None
        view_default: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None
        sum_dim_int_list_1: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_41, [0], True);  view_41 = None
        view_default_1: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_1, _shape_param_1);  sum_dim_int_list_1 = _shape_param_1 = None
        add_tensor: "f32[4096]" = torch.ops.aten.add.Tensor(view_default, view_default_1);  view_default = view_default_1 = None
        sum_dim_int_list_2: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_71, [0], True);  view_71 = None
        view_default_2: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_2, _shape_param_2);  sum_dim_int_list_2 = _shape_param_2 = None
        add_tensor_1: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor, view_default_2);  add_tensor = view_default_2 = None
        sum_dim_int_list_3: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_101, [0], True);  view_101 = None
        view_default_3: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_3, _shape_param_3);  sum_dim_int_list_3 = _shape_param_3 = None
        add_tensor_2: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_1, view_default_3);  add_tensor_1 = view_default_3 = None
        sum_dim_int_list_4: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_131, [0], True);  view_131 = None
        view_default_4: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_4, _shape_param_4);  sum_dim_int_list_4 = _shape_param_4 = None
        add_tensor_3: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_2, view_default_4);  add_tensor_2 = view_default_4 = None
        sum_dim_int_list_5: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_161, [0], True);  view_161 = None
        view_default_5: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_5, _shape_param_5);  sum_dim_int_list_5 = _shape_param_5 = None
        add_tensor_4: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_3, view_default_5);  add_tensor_3 = view_default_5 = None
        sum_dim_int_list_6: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_191, [0], True);  view_191 = None
        view_default_6: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_6, _shape_param_6);  sum_dim_int_list_6 = _shape_param_6 = None
        add_tensor_5: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_4, view_default_6);  add_tensor_4 = view_default_6 = None
        sum_dim_int_list_7: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_221, [0], True);  view_221 = None
        view_default_7: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_7, _shape_param_7);  sum_dim_int_list_7 = _shape_param_7 = None
        add_tensor_6: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_5, view_default_7);  add_tensor_5 = view_default_7 = None
        sum_dim_int_list_8: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_251, [0], True);  view_251 = None
        view_default_8: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_8, _shape_param_8);  sum_dim_int_list_8 = _shape_param_8 = None
        add_tensor_7: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_6, view_default_8);  add_tensor_6 = view_default_8 = None
        sum_dim_int_list_9: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_281, [0], True);  view_281 = None
        view_default_9: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_9, _shape_param_9);  sum_dim_int_list_9 = _shape_param_9 = None
        add_tensor_8: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_7, view_default_9);  add_tensor_7 = view_default_9 = None
        sum_dim_int_list_10: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_311, [0], True);  view_311 = None
        view_default_10: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_10, _shape_param_10);  sum_dim_int_list_10 = _shape_param_10 = None
        add_tensor_9: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_8, view_default_10);  add_tensor_8 = view_default_10 = None
        sum_dim_int_list_11: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_341, [0], True);  view_341 = None
        view_default_11: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_11, _shape_param_11);  sum_dim_int_list_11 = _shape_param_11 = None
        add_tensor_10: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_9, view_default_11);  add_tensor_9 = view_default_11 = None
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
