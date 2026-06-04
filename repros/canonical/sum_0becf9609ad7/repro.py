"""
Standalone repro captured via capture_hook.
Label: torchbench_squeezenet1_1_train_001
Pattern hash: 0becf9609ad7
Shape hash: e99dcbfa
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([512, 1000], f32), T([512, 1000, 13, 13], b8), S([512, 1000, 1, 1]), S([512, 1000, 13, 13]))"

class Repro(torch.nn.Module):
    def forward(self, arg66_1: "f32[512, 1000]", arg48_1: "b8[512, 1000, 13, 13]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[512, 1000, 1, 1]" = torch.ops.aten.view.default(arg66_1, _shape_param_0);  arg66_1 = _shape_param_0 = None
        expand_default: "f32[512, 1000, 13, 13]" = torch.ops.aten.expand.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        div_scalar: "f32[512, 1000, 13, 13]" = torch.ops.aten.div.Scalar(expand_default, 169);  expand_default = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[512, 1000, 13, 13]" = torch.ops.aten.where.self(arg48_1, full_default, div_scalar);  arg48_1 = full_default = div_scalar = None
        sum_dim_int_list: "f32[1000]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3]);  where_self = None
        return sum_dim_int_list

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
