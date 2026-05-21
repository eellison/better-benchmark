"""
Standalone repro captured via capture_hook.
Label: torchbench_timm_resnest_train_000
Pattern hash: b942094d64da
Shape hash: d0180a7f
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
_shapes_config = "(T([32, 1024, 1, 1], f32), T([32, 2, 512, 14, 14], f32), S([32, 1, 2, -1]), S([32, -1]), S([32, -1, 1, 1]), S([32, 2, 512, 1, 1]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_24: "f32[32, 1024, 1, 1]", view_15: "f32[32, 2, 512, 14, 14]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "f32[32, 1, 2, 512]" = torch.ops.aten.view.default(convolution_24, _shape_param_0);  convolution_24 = _shape_param_0 = None
        permute_default: "f32[32, 2, 1, 512]" = torch.ops.aten.permute.default(view_default, [0, 2, 1, 3]);  view_default = None
        amax_default: "f32[32, 1, 1, 512]" = torch.ops.aten.amax.default(permute_default, [1], True)
        sub_tensor: "f32[32, 2, 1, 512]" = torch.ops.aten.sub.Tensor(permute_default, amax_default);  permute_default = amax_default = None
        exp_default: "f32[32, 2, 1, 512]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[32, 1, 1, 512]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True)
        div_tensor: "f32[32, 2, 1, 512]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        view_default_1: "f32[32, 1024]" = torch.ops.aten.view.default(div_tensor, _shape_param_1);  div_tensor = _shape_param_1 = None
        view_default_2: "f32[32, 1024, 1, 1]" = torch.ops.aten.view.default(view_default_1, _shape_param_2);  view_default_1 = _shape_param_2 = None
        view_default_3: "f32[32, 2, 512, 1, 1]" = torch.ops.aten.view.default(view_default_2, _shape_param_3);  view_default_2 = _shape_param_3 = None
        mul_tensor: "f32[32, 2, 512, 14, 14]" = torch.ops.aten.mul.Tensor(view_15, view_default_3);  view_15 = view_default_3 = None
        sum_dim_int_list_1: "f32[32, 512, 14, 14]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [1]);  mul_tensor = None
        avg_pool2d_default: "f32[32, 512, 7, 7]" = torch.ops.aten.avg_pool2d.default(sum_dim_int_list_1, [3, 3], [2, 2], [1, 1]);  sum_dim_int_list_1 = None
        return avg_pool2d_default



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
