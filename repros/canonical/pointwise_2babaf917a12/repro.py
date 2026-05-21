"""
Standalone repro captured via capture_hook.
Label: hf_LayoutLMForMaskedLM_infer_000
Pattern hash: 2babaf917a12
Shape hash: e045a23d
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
_shapes_config = "(T([16384, 768], f32), S([32, 512, 768]), S([32, 512, -1, 64]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_66: "f32[16384, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[32, 512, 768]" = torch.ops.aten.view.default(addmm_66, _shape_param_0);  addmm_66 = _shape_param_0 = None
        view_default_1: "f32[32, 512, 12, 64]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        permute_default: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_default_1, [0, 2, 1, 3]);  view_default_1 = None
        clone_default: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        return clone_default



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
