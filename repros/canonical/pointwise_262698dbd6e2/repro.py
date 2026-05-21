"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train_014
Pattern hash: 262698dbd6e2
Shape hash: c7318a28
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
_shapes_config = "(T([32768, 128], f32), T([32768, 128], f32), T([128], f32), S([256, 128, 128]), S([256, 128, 128]), S([32768, 128]))"

class Repro(torch.nn.Module):
    def forward(self, mm_716: "f32[32768, 128]", mm_718: "f32[32768, 128]", arg8_1: "f32[128]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[256, 128, 128]" = torch.ops.aten.view.default(mm_716, _shape_param_0);  mm_716 = _shape_param_0 = None
        view_default_1: "f32[256, 128, 128]" = torch.ops.aten.view.default(mm_718, _shape_param_1);  mm_718 = _shape_param_1 = None
        add_tensor: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_default, view_default_1);  view_default = view_default_1 = None
        mul_tensor: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_tensor, arg8_1);  add_tensor = arg8_1 = None
        view_default_2: "f32[32768, 128]" = torch.ops.aten.view.default(mul_tensor, _shape_param_2);  mul_tensor = _shape_param_2 = None
        return view_default_2



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
