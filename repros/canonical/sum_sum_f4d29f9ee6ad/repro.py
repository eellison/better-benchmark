"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train_001
Pattern hash: f4d29f9ee6ad
Shape hash: 9f5857a1
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
_shapes_config = "(T([128, 128, 20005], f32), T([128, 128, 20005], f32), S([16384, 20005]), S([20005]))"

class Repro(torch.nn.Module):
    def forward(self, arg337_1: "f32[128, 128, 20005]", arg287_1: "f32[128, 128, 20005]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        sum_dim_int_list: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(arg337_1, [-1], True)
        exp_default: "f32[128, 128, 20005]" = torch.ops.aten.exp.default(arg287_1);  arg287_1 = None
        mul_tensor: "f32[128, 128, 20005]" = torch.ops.aten.mul.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        sub_tensor: "f32[128, 128, 20005]" = torch.ops.aten.sub.Tensor(arg337_1, mul_tensor);  arg337_1 = mul_tensor = None
        view_default: "f32[16384, 20005]" = torch.ops.aten.view.default(sub_tensor, _shape_param_0);  sub_tensor = _shape_param_0 = None
        permute_default: "f32[20005, 16384]" = torch.ops.aten.permute.default(view_default, [1, 0])
        sum_dim_int_list_1: "f32[1, 20005]" = torch.ops.aten.sum.dim_IntList(view_default, [0], True);  view_default = None
        view_default_1: "f32[20005]" = torch.ops.aten.view.default(sum_dim_int_list_1, _shape_param_1);  sum_dim_int_list_1 = _shape_param_1 = None
        return (permute_default, view_default_1)



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
