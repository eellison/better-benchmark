"""
Standalone repro captured via capture_hook.
Label: hf_BartForCausalLM_train_007
Pattern hash: 5466ad7742f1
Shape hash: 5699528d
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
_shapes_config = "(T([8192, 1024], f32), T([8, 1024, 1024], f32), T([8192, 1024], f32), T([8192, 1024], f32), S([8, 1024, 1024]), S([8, 1024, 1024]), S([8, 1024, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, mm_6: "f32[8192, 1024]", mul_22: "f32[8, 1024, 1024]", mm_8: "f32[8192, 1024]", mm_10: "f32[8192, 1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(mm_6, _shape_param_0);  mm_6 = _shape_param_0 = None
        add_tensor: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_22, view_default);  mul_22 = view_default = None
        view_default_1: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(mm_8, _shape_param_1);  mm_8 = _shape_param_1 = None
        add_tensor_1: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor, view_default_1);  add_tensor = view_default_1 = None
        view_default_2: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(mm_10, _shape_param_2);  mm_10 = _shape_param_2 = None
        add_tensor_2: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_1, view_default_2);  add_tensor_1 = view_default_2 = None
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
