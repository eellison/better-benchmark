"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_infer_000
Pattern hash: 52a30d4882cf
Shape hash: db5dcd9c
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
_shapes_config = "(T([128, 32, 49, 32], f32, stride=(150528, 32, 3072, 1)), S([128, 32, 49, 32]), S([4096, 49, 32]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_173: "f32[128, 32, 49, 32]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        expand_default: "f32[128, 32, 49, 32]" = torch.ops.aten.expand.default(getitem_173, _shape_param_0);  getitem_173 = _shape_param_0 = None
        clone_default: "f32[128, 32, 49, 32]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default: "f32[4096, 49, 32]" = torch.ops.aten.view.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None
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
