"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_infer_000
Pattern hash: e0498dcba9fe
Shape hash: b827522f
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
_shapes_config = "(T([32768, 512], f32), T([256, 128, 512], f32), T([512], f32), T([512], f32), S([256, 128, 512]), S([32768, 512]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_360: "f32[32768, 512]", add_348: "f32[256, 128, 512]", arg1111_1: "f32[512]", arg1112_1: "f32[512]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[256, 128, 512]" = torch.ops.aten.view.default(addmm_360, _shape_param_0);  addmm_360 = _shape_param_0 = None
        add_tensor: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_default, add_348);  view_default = add_348 = None
        mul_tensor: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor, arg1111_1);  add_tensor = arg1111_1 = None
        add_tensor_1: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_tensor, arg1112_1);  mul_tensor = arg1112_1 = None
        view_default_1: "f32[32768, 512]" = torch.ops.aten.view.default(add_tensor_1, _shape_param_1);  add_tensor_1 = _shape_param_1 = None
        return view_default_1



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
