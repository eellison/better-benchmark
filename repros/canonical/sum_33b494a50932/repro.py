"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_train_005
Pattern hash: 33b494a50932
Shape hash: 3ee03bf5
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
_shapes_config = "(T([8192, 3072], f32), T([8192, 3072], f32), S([8, 1024, 3072]), S([8, 1024, 3072]), S([8192, 3072]), S([3072]))"

class Repro(torch.nn.Module):
    def forward(self, mm_132: "f32[8192, 3072]", arg106_1: "f32[8192, 3072]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "f32[8, 1024, 3072]" = torch.ops.aten.view.default(mm_132, _shape_param_0);  mm_132 = _shape_param_0 = None
        view_default_1: "f32[8, 1024, 3072]" = torch.ops.aten.view.default(arg106_1, _shape_param_1);  arg106_1 = _shape_param_1 = None
        mul_tensor: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_default_1, 0.7071067811865476)
        erf_default: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_tensor);  mul_tensor = None
        add_tensor: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_1: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_tensor, 0.5);  add_tensor = None
        mul_tensor_2: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_default_1, view_default_1)
        mul_tensor_3: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_tensor_2, -0.5);  mul_tensor_2 = None
        exp_default: "f32[8, 1024, 3072]" = torch.ops.aten.exp.default(mul_tensor_3);  mul_tensor_3 = None
        mul_tensor_4: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_5: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_default_1, mul_tensor_4);  view_default_1 = mul_tensor_4 = None
        add_tensor_1: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_5);  mul_tensor_1 = mul_tensor_5 = None
        mul_tensor_6: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_default, add_tensor_1);  view_default = add_tensor_1 = None
        view_default_2: "f32[8192, 3072]" = torch.ops.aten.view.default(mul_tensor_6, _shape_param_2);  mul_tensor_6 = _shape_param_2 = None
        permute_default: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_default_2, [1, 0])
        sum_dim_int_list: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_default_2, [0], True);  view_default_2 = None
        view_default_3: "f32[3072]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_3);  sum_dim_int_list = _shape_param_3 = None
        return (permute_default, view_default_3)



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
