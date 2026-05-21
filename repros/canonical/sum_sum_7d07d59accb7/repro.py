"""
Standalone repro captured via capture_hook.
Label: hf_GPTNeoForCausalLM_train_001
Pattern hash: 7d07d59accb7
Shape hash: efa5c410
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
_shapes_config = "(T([4096, 2048], f32), T([4096, 2048], f32), T([4096, 2048], f32), T([2048], f32), T([32, 128, 2048], f32), T([32, 128, 1], f32), T([32, 128, 2048], f32), S([32, 128, 2048]), S([32, 128, 2048]), S([32, 128, 2048]), S([4096, 2048]))"

class Repro(torch.nn.Module):
    def forward(self, mm_273: "f32[4096, 2048]", mm_275: "f32[4096, 2048]", mm_277: "f32[4096, 2048]", arg11_1: "f32[2048]", arg231_1: "f32[32, 128, 2048]", arg537_1: "f32[32, 128, 1]", add_181: "f32[32, 128, 2048]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "f32[32, 128, 2048]" = torch.ops.aten.view.default(mm_273, _shape_param_0);  mm_273 = _shape_param_0 = None
        view_default_1: "f32[32, 128, 2048]" = torch.ops.aten.view.default(mm_275, _shape_param_1);  mm_275 = _shape_param_1 = None
        add_tensor: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_default, view_default_1);  view_default = view_default_1 = None
        view_default_2: "f32[32, 128, 2048]" = torch.ops.aten.view.default(mm_277, _shape_param_2);  mm_277 = _shape_param_2 = None
        add_tensor_1: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_tensor, view_default_2);  add_tensor = view_default_2 = None
        mul_tensor: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_tensor_1, arg11_1);  add_tensor_1 = arg11_1 = None
        mul_tensor_1: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_tensor, 2048)
        sum_dim_int_list: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_tensor, arg231_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg231_1, sum_dim_int_list_1);  arg231_1 = sum_dim_int_list_1 = None
        sub_tensor: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg537_1, sub_tensor_1);  arg537_1 = sub_tensor_1 = None
        add_tensor_2: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_181, mul_tensor_4);  add_181 = mul_tensor_4 = None
        view_default_3: "f32[4096, 2048]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_3);  add_tensor_2 = _shape_param_3 = None
        return view_default_3



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
