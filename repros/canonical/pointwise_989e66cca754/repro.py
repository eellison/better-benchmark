"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Whisper_train_000
Pattern hash: 989e66cca754
Shape hash: e34b5d25
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([12000, 384], f32), T([8, 1500, 384], f32), S([8, 1500, 384]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_4: "f32[12000, 384]", add_2: "f32[8, 1500, 384]", _shape_param_0):
        # No stacktrace found for following nodes
        view_default: "f32[8, 1500, 384]" = torch.ops.aten.view.default(addmm_4, _shape_param_0);  addmm_4 = _shape_param_0 = None
        add_tensor: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(add_2, view_default);  add_2 = view_default = None
        return add_tensor

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
