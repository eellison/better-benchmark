"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Whisper_train_003
Pattern hash: 335abef71e93
Shape hash: dde5ac79
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
_shapes_config = "(T([12000, 384], f32), T([12000, 384], f32), T([8, 1500, 384], f32), T([8, 1500, 1], f32), T([8, 1500, 1], f32), T([384], f32), T([8, 1500, 384], f32), S([8, 1500, 384]), S([8, 1500, 384]), S([12000, 384]), S([384]))"

class Repro(torch.nn.Module):
    def forward(self, mm_2: "f32[12000, 384]", arg19_1: "f32[12000, 384]", arg1_1: "f32[8, 1500, 384]", arg20_1: "f32[8, 1500, 1]", arg21_1: "f32[8, 1500, 1]", arg6_1: "f32[384]", arg25_1: "f32[8, 1500, 384]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "f32[8, 1500, 384]" = torch.ops.aten.view.default(mm_2, _shape_param_0);  mm_2 = _shape_param_0 = None
        view_default_1: "f32[8, 1500, 384]" = torch.ops.aten.view.default(arg19_1, _shape_param_1);  arg19_1 = _shape_param_1 = None
        add_tensor: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(arg1_1, view_default_1);  arg1_1 = view_default_1 = None
        sub_tensor: "f32[8, 1500, 384]" = torch.ops.aten.sub.Tensor(add_tensor, arg20_1);  add_tensor = arg20_1 = None
        mul_tensor: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(sub_tensor, arg21_1);  sub_tensor = None
        mul_tensor_1: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(view_default, arg6_1);  arg6_1 = None
        mul_tensor_2: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_tensor_1, 384)
        sum_dim_int_list: "f32[8, 1500, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True)
        mul_tensor_3: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_tensor_1, mul_tensor);  mul_tensor_1 = None
        sum_dim_int_list_1: "f32[8, 1500, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        mul_tensor_4: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_tensor, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_1: "f32[8, 1500, 384]" = torch.ops.aten.sub.Tensor(mul_tensor_2, sum_dim_int_list);  mul_tensor_2 = sum_dim_int_list = None
        sub_tensor_2: "f32[8, 1500, 384]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor: "f32[8, 1500, 1]" = torch.ops.aten.div.Tensor(arg21_1, 384);  arg21_1 = None
        mul_tensor_5: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_6: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(view_default, mul_tensor);  mul_tensor = None
        sum_dim_int_list_2: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_3: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_default, [0, 1]);  view_default = None
        add_tensor_1: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(arg25_1, mul_tensor_5);  arg25_1 = mul_tensor_5 = None
        clone_default: "f32[8, 1500, 384]" = torch.ops.aten.clone.default(add_tensor_1, memory_format = torch.contiguous_format);  add_tensor_1 = None
        view_default_2: "f32[12000, 384]" = torch.ops.aten.view.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None
        permute_default: "f32[384, 12000]" = torch.ops.aten.permute.default(view_default_2, [1, 0])
        sum_dim_int_list_4: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_default_2, [0], True);  view_default_2 = None
        view_default_3: "f32[384]" = torch.ops.aten.view.default(sum_dim_int_list_4, _shape_param_3);  sum_dim_int_list_4 = _shape_param_3 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, permute_default, view_default_3)



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
