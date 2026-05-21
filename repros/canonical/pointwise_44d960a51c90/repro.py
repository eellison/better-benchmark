"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_infer_000
Pattern hash: 44d960a51c90
Shape hash: 170aecd0
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
_shapes_config = "(T([32768, 128], f32), T([128], f32), T([128], f32), S([256, 128, 128]), S([32768, 128]), S([32768, 128]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_331: "f32[32768, 128]", arg1027_1: "f32[128]", arg1028_1: "f32[128]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[256, 128, 128]" = torch.ops.aten.view.default(addmm_331, _shape_param_0);  addmm_331 = _shape_param_0 = None
        mul_tensor: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_default, arg1027_1);  view_default = arg1027_1 = None
        add_tensor: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_tensor, arg1028_1);  mul_tensor = arg1028_1 = None
        view_default_1: "f32[32768, 128]" = torch.ops.aten.view.default(add_tensor, _shape_param_1);  _shape_param_1 = None
        view_default_2: "f32[32768, 128]" = torch.ops.aten.view.default(add_tensor, _shape_param_2);  add_tensor = _shape_param_2 = None
        return (view_default_2, view_default_1)



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
