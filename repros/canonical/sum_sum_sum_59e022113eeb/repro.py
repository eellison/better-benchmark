"""
Standalone repro captured via capture_hook.
Label: hf_DistilBertForMaskedLM_train_001
Pattern hash: 59e022113eeb
Shape hash: 2aec6584
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([32768, 768], f32), T([256, 128, 768], f32), T([768], f32), T([256, 128, 768], f32), T([256, 128, 1], f32), S([256, 128, 768]), S([32768, 768]), S([768]))"

class Repro(torch.nn.Module):
    def forward(self, mm_66: "f32[32768, 768]", mul_152: "f32[256, 128, 768]", arg8_1: "f32[768]", arg68_1: "f32[256, 128, 768]", arg164_1: "f32[256, 128, 1]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[256, 128, 768]" = torch.ops.aten.view.default(mm_66, _shape_param_0);  mm_66 = _shape_param_0 = None
        add_tensor: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_152, view_default);  mul_152 = view_default = None
        mul_tensor: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_tensor, arg8_1);  arg8_1 = None
        mul_tensor_1: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg68_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(arg68_1, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(arg164_1, sub_tensor_1);  arg164_1 = sub_tensor_1 = None
        mul_tensor_5: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_tensor, arg68_1);  arg68_1 = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_tensor, [0, 1]);  add_tensor = None
        view_default_1: "f32[32768, 768]" = torch.ops.aten.view.default(mul_tensor_4, _shape_param_1);  mul_tensor_4 = _shape_param_1 = None
        permute_default: "f32[768, 32768]" = torch.ops.aten.permute.default(view_default_1, [1, 0])
        sum_dim_int_list_4: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_default_1, [0], True);  view_default_1 = None
        view_default_2: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list_4, _shape_param_2);  sum_dim_int_list_4 = _shape_param_2 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, permute_default, view_default_2)

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
