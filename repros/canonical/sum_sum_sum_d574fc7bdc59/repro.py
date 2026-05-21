"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Whisper_train_003
Pattern hash: d574fc7bdc59
Shape hash: 353519f1
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
_shapes_config = "(T([12000, 384], f32), T([12000, 384], f32), T([12000, 384], f32), T([8, 1500, 384], f32), T([8, 1500, 1], f32), T([8, 1500, 1], f32), T([384], f32), T([8, 1500, 384], f32), S([8, 1500, 384]), S([8, 1500, 384]), S([8, 1500, 384]))"

class Repro(torch.nn.Module):
    def forward(self, mm_6: "f32[12000, 384]", mm_9: "f32[12000, 384]", mm_10: "f32[12000, 384]", arg1_1: "f32[8, 1500, 384]", arg9_1: "f32[8, 1500, 1]", arg10_1: "f32[8, 1500, 1]", arg0_1: "f32[384]", add_3: "f32[8, 1500, 384]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[8, 1500, 384]" = torch.ops.aten.view.default(mm_6, _shape_param_0);  mm_6 = _shape_param_0 = None
        view_default_1: "f32[8, 1500, 384]" = torch.ops.aten.view.default(mm_9, _shape_param_1);  mm_9 = _shape_param_1 = None
        add_tensor: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(view_default, view_default_1);  view_default = view_default_1 = None
        view_default_2: "f32[8, 1500, 384]" = torch.ops.aten.view.default(mm_10, _shape_param_2);  mm_10 = _shape_param_2 = None
        add_tensor_1: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(add_tensor, view_default_2);  add_tensor = view_default_2 = None
        sub_tensor: "f32[8, 1500, 384]" = torch.ops.aten.sub.Tensor(arg1_1, arg9_1);  arg1_1 = arg9_1 = None
        mul_tensor: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(sub_tensor, arg10_1);  sub_tensor = None
        mul_tensor_1: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(add_tensor_1, arg0_1);  arg0_1 = None
        mul_tensor_2: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_tensor_1, 384)
        sum_dim_int_list: "f32[8, 1500, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True)
        mul_tensor_3: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_tensor_1, mul_tensor);  mul_tensor_1 = None
        sum_dim_int_list_1: "f32[8, 1500, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        mul_tensor_4: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_tensor, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_1: "f32[8, 1500, 384]" = torch.ops.aten.sub.Tensor(mul_tensor_2, sum_dim_int_list);  mul_tensor_2 = sum_dim_int_list = None
        sub_tensor_2: "f32[8, 1500, 384]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor: "f32[8, 1500, 1]" = torch.ops.aten.div.Tensor(arg10_1, 384);  arg10_1 = None
        mul_tensor_5: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_6: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(add_tensor_1, mul_tensor);  mul_tensor = None
        sum_dim_int_list_2: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_3: "f32[384]" = torch.ops.aten.sum.dim_IntList(add_tensor_1, [0, 1]);  add_tensor_1 = None
        add_tensor_2: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(add_3, mul_tensor_5);  add_3 = mul_tensor_5 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, add_tensor_2)



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
