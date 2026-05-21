"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train_014
Pattern hash: 81185df244d2
Shape hash: 56704436
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
_shapes_config = "(T([256, 128, 512], f32), T([256, 128, 512], f32), S([512]))"

class Repro(torch.nn.Module):
    def forward(self, view_9: "f32[256, 128, 512]", arg1117_1: "f32[256, 128, 512]", _shape_param_0):
        # No stacktrace found for following nodes
        mul_tensor: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(view_9, arg1117_1);  view_9 = arg1117_1 = None
        sum_dim_int_list: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1], True);  mul_tensor = None
        view_default: "f32[512]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None
        return view_default



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
