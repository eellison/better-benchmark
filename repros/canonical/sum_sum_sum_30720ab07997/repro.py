"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Albert_train
Pattern hash: 30720ab07997
Shape hash: 77744a6d
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
_shapes_config = "(T([4096, 768], f32), T([4096, 768], f32), T([4096, 768], f32), T([4096, 768], f32), T([4096, 768], f32), T([4096, 768], f32), T([4096, 768], f32), T([4096, 768], f32), T([4096, 768], f32), T([4096, 768], f32), T([4096, 768], f32), T([4096, 768], f32), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]))"

class Repro(torch.nn.Module):
    def forward(self, view_279: "f32[4096, 768]", view_307: "f32[4096, 768]", view_335: "f32[4096, 768]", view_363: "f32[4096, 768]", view_391: "f32[4096, 768]", view_419: "f32[4096, 768]", view_447: "f32[4096, 768]", view_475: "f32[4096, 768]", view_503: "f32[4096, 768]", view_531: "f32[4096, 768]", view_559: "f32[4096, 768]", view_587: "f32[4096, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        sum_dim_int_list: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_279, [0], True);  view_279 = None
        reshape_default: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None
        sum_dim_int_list_1: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_307, [0], True);  view_307 = None
        reshape_default_1: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, _shape_param_1);  sum_dim_int_list_1 = _shape_param_1 = None
        add_tensor: "f32[768]" = torch.ops.aten.add.Tensor(reshape_default, reshape_default_1);  reshape_default = reshape_default_1 = None
        sum_dim_int_list_2: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_335, [0], True);  view_335 = None
        reshape_default_2: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, _shape_param_2);  sum_dim_int_list_2 = _shape_param_2 = None
        add_tensor_1: "f32[768]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_2);  add_tensor = reshape_default_2 = None
        sum_dim_int_list_3: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_363, [0], True);  view_363 = None
        reshape_default_3: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_3);  sum_dim_int_list_3 = _shape_param_3 = None
        add_tensor_2: "f32[768]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_3);  add_tensor_1 = reshape_default_3 = None
        sum_dim_int_list_4: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_391, [0], True);  view_391 = None
        reshape_default_4: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, _shape_param_4);  sum_dim_int_list_4 = _shape_param_4 = None
        add_tensor_3: "f32[768]" = torch.ops.aten.add.Tensor(add_tensor_2, reshape_default_4);  add_tensor_2 = reshape_default_4 = None
        sum_dim_int_list_5: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_419, [0], True);  view_419 = None
        reshape_default_5: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_5, _shape_param_5);  sum_dim_int_list_5 = _shape_param_5 = None
        add_tensor_4: "f32[768]" = torch.ops.aten.add.Tensor(add_tensor_3, reshape_default_5);  add_tensor_3 = reshape_default_5 = None
        sum_dim_int_list_6: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_447, [0], True);  view_447 = None
        reshape_default_6: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, _shape_param_6);  sum_dim_int_list_6 = _shape_param_6 = None
        add_tensor_5: "f32[768]" = torch.ops.aten.add.Tensor(add_tensor_4, reshape_default_6);  add_tensor_4 = reshape_default_6 = None
        sum_dim_int_list_7: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_475, [0], True);  view_475 = None
        reshape_default_7: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_7);  sum_dim_int_list_7 = _shape_param_7 = None
        add_tensor_6: "f32[768]" = torch.ops.aten.add.Tensor(add_tensor_5, reshape_default_7);  add_tensor_5 = reshape_default_7 = None
        sum_dim_int_list_8: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_503, [0], True);  view_503 = None
        reshape_default_8: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_8, _shape_param_8);  sum_dim_int_list_8 = _shape_param_8 = None
        add_tensor_7: "f32[768]" = torch.ops.aten.add.Tensor(add_tensor_6, reshape_default_8);  add_tensor_6 = reshape_default_8 = None
        sum_dim_int_list_9: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_531, [0], True);  view_531 = None
        reshape_default_9: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_9, _shape_param_9);  sum_dim_int_list_9 = _shape_param_9 = None
        add_tensor_8: "f32[768]" = torch.ops.aten.add.Tensor(add_tensor_7, reshape_default_9);  add_tensor_7 = reshape_default_9 = None
        sum_dim_int_list_10: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_559, [0], True);  view_559 = None
        reshape_default_10: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_10, _shape_param_10);  sum_dim_int_list_10 = _shape_param_10 = None
        add_tensor_9: "f32[768]" = torch.ops.aten.add.Tensor(add_tensor_8, reshape_default_10);  add_tensor_8 = reshape_default_10 = None
        sum_dim_int_list_11: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_587, [0], True);  view_587 = None
        reshape_default_11: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, _shape_param_11);  sum_dim_int_list_11 = _shape_param_11 = None
        add_tensor_10: "f32[768]" = torch.ops.aten.add.Tensor(add_tensor_9, reshape_default_11);  add_tensor_9 = reshape_default_11 = None
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
