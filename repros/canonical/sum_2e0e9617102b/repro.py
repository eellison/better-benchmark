"""
Standalone repro captured via capture_hook.
Label: timm_convnextv2_nano.fcmae_ft_in22k_in1k_train_001
Pattern hash: 2e0e9617102b
Shape hash: 5416df5b
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 80, 56, 56], f32), T([128, 80, 56, 56], f32))"

class Repro(torch.nn.Module):
    def forward(self, permute_46: "f32[128, 80, 56, 56]", getitem_123: "f32[128, 80, 56, 56]"):
        # No stacktrace found for following nodes
        add_tensor: "f32[128, 80, 56, 56]" = torch.ops.aten.add.Tensor(permute_46, getitem_123);  permute_46 = getitem_123 = None
        sum_dim_int_list: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_tensor, [0, 2, 3]);  add_tensor = None
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
