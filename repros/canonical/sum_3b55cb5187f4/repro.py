"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train_014
Pattern hash: 3b55cb5187f4
Shape hash: de2bbfba
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
_shapes_config = "(T([32768, 128], f32), T([256, 128, 128], f32), S([256, 128, 128]), S([128]))"

class Repro(torch.nn.Module):
    def forward(self, arg1098_1: "f32[32768, 128]", add_8: "f32[256, 128, 128]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[256, 128, 128]" = torch.ops.aten.view.default(arg1098_1, _shape_param_0);  arg1098_1 = _shape_param_0 = None
        mul_tensor: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_8, view_default);  add_8 = view_default = None
        sum_dim_int_list: "f32[1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1], True);  mul_tensor = None
        view_default_1: "f32[128]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_1);  sum_dim_int_list = _shape_param_1 = None
        return view_default_1



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
