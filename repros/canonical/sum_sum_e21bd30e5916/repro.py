"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train_014
Pattern hash: e21bd30e5916
Shape hash: 16ad3db5
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
_shapes_config = "(T([32768, 128], f32), T([128], f32), T([128], f32), T([32768, 128], f32), T([256, 128, 128], f32), T([256, 128, 128], f32), S([256, 128, 128]), S([256, 128, 128]), S([128]), S([128]))"

class Repro(torch.nn.Module):
    def forward(self, arg1097_1: "f32[32768, 128]", arg557_1: "f32[128]", arg558_1: "f32[128]", arg1103_1: "f32[32768, 128]", add_4: "f32[256, 128, 128]", mul_21: "f32[256, 128, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "f32[256, 128, 128]" = torch.ops.aten.view.default(arg1097_1, _shape_param_0);  arg1097_1 = _shape_param_0 = None
        mul_tensor: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_default, arg557_1);  arg557_1 = None
        add_tensor: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_tensor, arg558_1);  mul_tensor = arg558_1 = None
        view_default_1: "f32[256, 128, 128]" = torch.ops.aten.view.default(arg1103_1, _shape_param_1);  arg1103_1 = _shape_param_1 = None
        add_tensor_1: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_default_1, add_tensor);  view_default_1 = add_tensor = None
        mul_tensor_1: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_4, add_tensor_1);  add_4 = add_tensor_1 = None
        sum_dim_int_list: "f32[1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1], True);  mul_tensor_1 = None
        view_default_2: "f32[128]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_2);  sum_dim_int_list = _shape_param_2 = None
        mul_tensor_2: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(mul_21, view_default);  mul_21 = view_default = None
        sum_dim_int_list_1: "f32[1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1], True);  mul_tensor_2 = None
        view_default_3: "f32[128]" = torch.ops.aten.view.default(sum_dim_int_list_1, _shape_param_3);  sum_dim_int_list_1 = _shape_param_3 = None
        return (view_default_3, view_default_2)



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
