"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train_001
Pattern hash: 8f7c059eb6a4
Shape hash: 66cec495
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 2], f32), T([128, 2], f32), S([2]))"

class Repro(torch.nn.Module):
    def forward(self, arg285_1: "f32[128, 2]", arg336_1: "f32[128, 2]", _shape_param_0):
        # No stacktrace found for following nodes
        exp_default: "f32[128, 2]" = torch.ops.aten.exp.default(arg285_1);  arg285_1 = None
        sum_dim_int_list: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(arg336_1, [-1], True)
        mul_tensor: "f32[128, 2]" = torch.ops.aten.mul.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        sub_tensor: "f32[128, 2]" = torch.ops.aten.sub.Tensor(arg336_1, mul_tensor);  arg336_1 = mul_tensor = None
        permute_default: "f32[2, 128]" = torch.ops.aten.permute.default(sub_tensor, [1, 0])
        sum_dim_int_list_1: "f32[1, 2]" = torch.ops.aten.sum.dim_IntList(sub_tensor, [0], True);  sub_tensor = None
        view_default: "f32[2]" = torch.ops.aten.view.default(sum_dim_int_list_1, _shape_param_0);  sum_dim_int_list_1 = _shape_param_0 = None
        return (permute_default, view_default)

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
