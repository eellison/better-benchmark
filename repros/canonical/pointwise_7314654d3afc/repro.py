"""
Standalone repro captured via capture_hook.
Label: timm_deit_base_distilled_patch16_224_infer_000
Pattern hash: 7314654d3afc
Shape hash: e14495d7
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 1000], f32), T([128, 1000], f32))"

class Repro(torch.nn.Module):
    def forward(self, addmm_48: "f32[128, 1000]", addmm_49: "f32[128, 1000]"):
        # No stacktrace found for following nodes
        add_tensor: "f32[128, 1000]" = torch.ops.aten.add.Tensor(addmm_48, addmm_49);  addmm_48 = addmm_49 = None
        div_tensor: "f32[128, 1000]" = torch.ops.aten.div.Tensor(add_tensor, 2);  add_tensor = None
        return div_tensor

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
