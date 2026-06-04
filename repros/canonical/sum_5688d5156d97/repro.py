"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_train_001
Pattern hash: 5688d5156d97
Shape hash: 0f41d4d2
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 256, 48, 48], f32), T([128, 256, 1, 1], f32), T([128, 256, 1, 1], f32), S([128, 256, 48, 48]))"

class Repro(torch.nn.Module):
    def forward(self, mul_980: "f32[128, 256, 48, 48]", sigmoid_11: "f32[128, 256, 1, 1]", getitem_213: "f32[128, 256, 1, 1]", _shape_param_0):
        # No stacktrace found for following nodes
        mul_tensor: "f32[128, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_980, sigmoid_11);  mul_980 = sigmoid_11 = None
        expand_default: "f32[128, 256, 48, 48]" = torch.ops.aten.expand.default(getitem_213, _shape_param_0);  getitem_213 = _shape_param_0 = None
        div_scalar: "f32[128, 256, 48, 48]" = torch.ops.aten.div.Scalar(expand_default, 2304);  expand_default = None
        add_tensor: "f32[128, 256, 48, 48]" = torch.ops.aten.add.Tensor(mul_tensor, div_scalar);  mul_tensor = div_scalar = None
        sum_dim_int_list: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_tensor, [0, 2, 3]);  add_tensor = None
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
