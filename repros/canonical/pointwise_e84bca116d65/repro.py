"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_infer_002
Pattern hash: e84bca116d65
Shape hash: 20b2616c
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
_shapes_config = "(T([1024, 8, 768], f32, stride=(768, 786432, 1)), S([8192, 768]))"

class Repro(torch.nn.Module):
    def forward(self, permute_517: "f32[1024, 8, 768]", _shape_param_0):
        # No stacktrace found for following nodes
        clone_default: "f32[1024, 8, 768]" = torch.ops.aten.clone.default(permute_517, memory_format = torch.contiguous_format);  permute_517 = None
        view_default: "f32[8192, 768]" = torch.ops.aten.view.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None
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
