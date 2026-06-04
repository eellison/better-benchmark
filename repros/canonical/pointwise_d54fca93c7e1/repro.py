"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Longformer_infer_000
Pattern hash: d54fca93c7e1
Shape hash: d7517139
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "()"

class Repro(torch.nn.Module):
    def forward(self):
        # No stacktrace found for following nodes
        full_default: "f32[1, 4096]" = torch.ops.aten.full.default([1, 4096], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        unsqueeze_default: "f32[1, 1, 4096]" = torch.ops.aten.unsqueeze.default(full_default, 1);  full_default = None
        unsqueeze_default_1: "f32[1, 1, 1, 4096]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        convert_element_type_default: "f16[1, 1, 1, 4096]" = torch.ops.prims.convert_element_type.default(unsqueeze_default_1, torch.float16);  unsqueeze_default_1 = None
        sub_tensor: "f16[1, 1, 1, 4096]" = torch.ops.aten.sub.Tensor(1.0, convert_element_type_default);  convert_element_type_default = None
        return sub_tensor

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
