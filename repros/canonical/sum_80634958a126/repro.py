"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_train_001
Pattern hash: 80634958a126
Shape hash: 840e3665
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([4096, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), S([8, 512, 4096]), S([8, 512, 4096]), S([8, 512, 4096]), S([4096, 4096]), S([4096]))"

class Repro(torch.nn.Module):
    def forward(self, mm_142: "f32[4096, 4096]", mul_341: "f32[8, 512, 4096]", mm_144: "f32[4096, 4096]", mm_146: "f32[4096, 4096]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view_default: "f32[8, 512, 4096]" = torch.ops.aten.view.default(mm_142, _shape_param_0);  mm_142 = _shape_param_0 = None
        add_tensor: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_341, view_default);  mul_341 = view_default = None
        view_default_1: "f32[8, 512, 4096]" = torch.ops.aten.view.default(mm_144, _shape_param_1);  mm_144 = _shape_param_1 = None
        add_tensor_1: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_tensor, view_default_1);  add_tensor = view_default_1 = None
        view_default_2: "f32[8, 512, 4096]" = torch.ops.aten.view.default(mm_146, _shape_param_2);  mm_146 = _shape_param_2 = None
        add_tensor_2: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_tensor_1, view_default_2);  add_tensor_1 = view_default_2 = None
        view_default_3: "f32[4096, 4096]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_3);  add_tensor_2 = _shape_param_3 = None
        permute_default: "f32[4096, 4096]" = torch.ops.aten.permute.default(view_default_3, [1, 0])
        sum_dim_int_list: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_default_3, [0], True);  view_default_3 = None
        view_default_4: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_4);  sum_dim_int_list = _shape_param_4 = None
        return (permute_default, view_default_4)

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
