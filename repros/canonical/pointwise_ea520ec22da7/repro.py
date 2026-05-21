"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_infer_000
Pattern hash: ea520ec22da7
Shape hash: 5a1fbf8b
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
_shapes_config = "(T([25216, 2304], f32), S([128, 197, 2304]), S([128, 197, 3, 12, -1]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_44: "f32[25216, 2304]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[128, 197, 2304]" = torch.ops.aten.view.default(addmm_44, _shape_param_0);  addmm_44 = _shape_param_0 = None
        view_default_1: "f32[128, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        permute_default: "f32[3, 128, 12, 197, 64]" = torch.ops.aten.permute.default(view_default_1, [2, 0, 3, 1, 4]);  view_default_1 = None
        unbind_int = torch.ops.aten.unbind.int(permute_default);  permute_default = None
        getitem: "f32[128, 12, 197, 64]" = unbind_int[0]
        getitem_1: "f32[128, 12, 197, 64]" = unbind_int[1]
        getitem_2: "f32[128, 12, 197, 64]" = unbind_int[2];  unbind_int = None
        return (getitem, getitem_1, getitem_2)



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
