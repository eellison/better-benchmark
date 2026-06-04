"""
Standalone repro captured via capture_hook.
Label: timm_adv_inception_v3_train_001
Pattern hash: 22b5cd24890b
Shape hash: 6f5549c8
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 1000], f32), S([1000]))"

class Repro(torch.nn.Module):
    def forward(self, arg592_1: "f32[128, 1000]", _shape_param_0):
        # No stacktrace found for following nodes
        sum_dim_int_list: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(arg592_1, [0], True);  arg592_1 = None
        view_default: "f32[1000]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None
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
