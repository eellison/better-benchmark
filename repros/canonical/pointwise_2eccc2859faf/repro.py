"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_infer_002
Pattern hash: 2eccc2859faf
Shape hash: 40f7eb36
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
_shapes_config = "(T([8, 1024, 768], f32), S([8192, 768]), S([8192, 768]))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[8, 1024, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        permute_default: "f32[1024, 8, 768]" = torch.ops.aten.permute.default(arg0_1, [1, 0, 2]);  arg0_1 = None
        clone_default: "f32[1024, 8, 768]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format)
        view_default: "f32[8192, 768]" = torch.ops.aten.view.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None
        clone_default_1: "f32[1024, 8, 768]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_1: "f32[8192, 768]" = torch.ops.aten.view.default(clone_default_1, _shape_param_1);  clone_default_1 = _shape_param_1 = None
        return (view_default, view_default_1)



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
