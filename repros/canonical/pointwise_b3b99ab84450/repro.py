"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train_014
Pattern hash: b3b99ab84450
Shape hash: b92348b8
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
_shapes_config = "(T([32768, 512], f32), T([256, 128, 512], f32), T([32768, 512], f32), T([32768, 512], f32), T([512], f32), S([256, 128, 512]), S([256, 128, 512]), S([256, 128, 512]), S([32768, 512]))"

class Repro(torch.nn.Module):
    def forward(self, mm_714: "f32[32768, 512]", mul_516: "f32[256, 128, 512]", mm_720: "f32[32768, 512]", mm_722: "f32[32768, 512]", arg3_1: "f32[512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "f32[256, 128, 512]" = torch.ops.aten.view.default(mm_714, _shape_param_0);  mm_714 = _shape_param_0 = None
        add_tensor: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_516, view_default);  mul_516 = view_default = None
        view_default_1: "f32[256, 128, 512]" = torch.ops.aten.view.default(mm_720, _shape_param_1);  mm_720 = _shape_param_1 = None
        add_tensor_1: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor, view_default_1);  add_tensor = view_default_1 = None
        view_default_2: "f32[256, 128, 512]" = torch.ops.aten.view.default(mm_722, _shape_param_2);  mm_722 = _shape_param_2 = None
        add_tensor_2: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_1, view_default_2);  add_tensor_1 = view_default_2 = None
        mul_tensor: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor_2, arg3_1);  add_tensor_2 = arg3_1 = None
        view_default_3: "f32[32768, 512]" = torch.ops.aten.view.default(mul_tensor, _shape_param_3);  mul_tensor = _shape_param_3 = None
        return view_default_3



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
