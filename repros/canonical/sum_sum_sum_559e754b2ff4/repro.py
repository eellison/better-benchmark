"""
Standalone repro captured via capture_hook.
Label: hf_GPTJForCausalLM_train_001
Pattern hash: 559e754b2ff4
Shape hash: d4bb7bad
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
_shapes_config = "(T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([4096], f32), T([1, 128, 4096], f32), T([1, 128, 1], f32), T([1, 128, 4096], f32), S([1, 128, 4096]), S([1, 128, 4096]), S([1, 128, 4096]), S([1, 128, 4096]), S([128, 4096]), S([4096]))"

class Repro(torch.nn.Module):
    def forward(self, mm_316: "f32[128, 4096]", mm_321: "f32[128, 4096]", mm_323: "f32[128, 4096]", mm_325: "f32[128, 4096]", arg8_1: "f32[4096]", arg216_1: "f32[1, 128, 4096]", arg628_1: "f32[1, 128, 1]", add_364: "f32[1, 128, 4096]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        view_default: "f32[1, 128, 4096]" = torch.ops.aten.view.default(mm_316, _shape_param_0);  mm_316 = _shape_param_0 = None
        view_default_1: "f32[1, 128, 4096]" = torch.ops.aten.view.default(mm_321, _shape_param_1);  mm_321 = _shape_param_1 = None
        add_tensor: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_default, view_default_1);  view_default = view_default_1 = None
        view_default_2: "f32[1, 128, 4096]" = torch.ops.aten.view.default(mm_323, _shape_param_2);  mm_323 = _shape_param_2 = None
        add_tensor_1: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_tensor, view_default_2);  add_tensor = view_default_2 = None
        view_default_3: "f32[1, 128, 4096]" = torch.ops.aten.view.default(mm_325, _shape_param_3);  mm_325 = _shape_param_3 = None
        add_tensor_2: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_tensor_1, view_default_3);  add_tensor_1 = view_default_3 = None
        mul_tensor: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_tensor_2, arg8_1);  arg8_1 = None
        mul_tensor_1: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor, 4096)
        sum_dim_int_list: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor, arg216_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg216_1, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg628_1, sub_tensor_1);  arg628_1 = sub_tensor_1 = None
        mul_tensor_5: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_tensor_2, arg216_1);  arg216_1 = None
        sum_dim_int_list_2: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_3: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_tensor_2, [0, 1]);  add_tensor_2 = None
        add_tensor_3: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_364, mul_tensor_4);  add_364 = mul_tensor_4 = None
        view_default_4: "f32[128, 4096]" = torch.ops.aten.view.default(add_tensor_3, _shape_param_4);  add_tensor_3 = _shape_param_4 = None
        permute_default: "f32[4096, 128]" = torch.ops.aten.permute.default(view_default_4, [1, 0])
        sum_dim_int_list_4: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_default_4, [0], True);  view_default_4 = None
        view_default_5: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list_4, _shape_param_5);  sum_dim_int_list_4 = _shape_param_5 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, permute_default, view_default_5)



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
