"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_train_005
Pattern hash: 9cc353ec2f96
Shape hash: 57b63c66
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
_shapes_config = "(T([8192, 768], f32), T([8192, 768], f32), T([8192, 768], f32), T([8, 1024, 768], f32), S([1024, 8, 768]), S([1024, 8, 768]), S([1024, 8, 768]))"

class Repro(torch.nn.Module):
    def forward(self, mm_139: "f32[8192, 768]", mm_141: "f32[8192, 768]", mm_143: "f32[8192, 768]", mul_305: "f32[8, 1024, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[1024, 8, 768]" = torch.ops.aten.view.default(mm_139, _shape_param_0);  mm_139 = _shape_param_0 = None
        view_default_1: "f32[1024, 8, 768]" = torch.ops.aten.view.default(mm_141, _shape_param_1);  mm_141 = _shape_param_1 = None
        add_tensor: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_default, view_default_1);  view_default = view_default_1 = None
        view_default_2: "f32[1024, 8, 768]" = torch.ops.aten.view.default(mm_143, _shape_param_2);  mm_143 = _shape_param_2 = None
        add_tensor_1: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(add_tensor, view_default_2);  add_tensor = view_default_2 = None
        permute_default: "f32[8, 1024, 768]" = torch.ops.aten.permute.default(add_tensor_1, [1, 0, 2]);  add_tensor_1 = None
        add_tensor_2: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_305, permute_default);  mul_305 = permute_default = None
        return add_tensor_2



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
