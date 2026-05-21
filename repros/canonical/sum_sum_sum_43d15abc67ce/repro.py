"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train_014
Pattern hash: 43d15abc67ce
Shape hash: 2113a6aa
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
_shapes_config = "(T([32768, 128], f32), T([256, 128, 128], f32), T([256, 128, 128], f32), T([128], f32), S([256, 128, 128]), S([128]), S([128]), S([32768, 128]), S([128]))"

class Repro(torch.nn.Module):
    def forward(self, mm_706: "f32[32768, 128]", mul_522: "f32[256, 128, 128]", arg600_1: "f32[256, 128, 128]", arg16_1: "f32[128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view_default: "f32[256, 128, 128]" = torch.ops.aten.view.default(mm_706, _shape_param_0);  mm_706 = _shape_param_0 = None
        add_tensor: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_522, view_default);  mul_522 = view_default = None
        sum_dim_int_list: "f32[1, 1, 128]" = torch.ops.aten.sum.dim_IntList(add_tensor, [0, 1], True)
        view_default_1: "f32[128]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_1);  sum_dim_int_list = _shape_param_1 = None
        mul_tensor: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_tensor, arg600_1);  arg600_1 = None
        mul_tensor_1: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_tensor, arg16_1);  add_tensor = arg16_1 = None
        sum_dim_int_list_1: "f32[1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1], True);  mul_tensor = None
        view_default_2: "f32[128]" = torch.ops.aten.view.default(sum_dim_int_list_1, _shape_param_2);  sum_dim_int_list_1 = _shape_param_2 = None
        view_default_3: "f32[32768, 128]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_3);  mul_tensor_1 = _shape_param_3 = None
        permute_default: "f32[128, 32768]" = torch.ops.aten.permute.default(view_default_3, [1, 0])
        sum_dim_int_list_2: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_default_3, [0], True);  view_default_3 = None
        view_default_4: "f32[128]" = torch.ops.aten.view.default(sum_dim_int_list_2, _shape_param_4);  sum_dim_int_list_2 = _shape_param_4 = None
        return (view_default_1, view_default_2, permute_default, view_default_4)



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
