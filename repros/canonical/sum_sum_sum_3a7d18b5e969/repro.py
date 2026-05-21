"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train_014
Pattern hash: 3a7d18b5e969
Shape hash: 2488abed
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
_shapes_config = "(T([32768, 512], f32), T([256, 128, 512], f32), T([32768, 512], f32), T([32768, 512], f32), T([256, 128, 512], f32), T([512], f32), S([256, 128, 512]), S([256, 128, 512]), S([256, 128, 512]), S([512]), S([512]), S([32768, 512]), S([512]))"

class Repro(torch.nn.Module):
    def forward(self, mm_684: "f32[32768, 512]", mul_494: "f32[256, 128, 512]", mm_690: "f32[32768, 512]", mm_692: "f32[32768, 512]", arg611_1: "f32[256, 128, 512]", arg27_1: "f32[512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        view_default: "f32[256, 128, 512]" = torch.ops.aten.view.default(mm_684, _shape_param_0);  mm_684 = _shape_param_0 = None
        add_tensor: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_494, view_default);  mul_494 = view_default = None
        view_default_1: "f32[256, 128, 512]" = torch.ops.aten.view.default(mm_690, _shape_param_1);  mm_690 = _shape_param_1 = None
        add_tensor_1: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor, view_default_1);  add_tensor = view_default_1 = None
        view_default_2: "f32[256, 128, 512]" = torch.ops.aten.view.default(mm_692, _shape_param_2);  mm_692 = _shape_param_2 = None
        add_tensor_2: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_1, view_default_2);  add_tensor_1 = view_default_2 = None
        sum_dim_int_list: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(add_tensor_2, [0, 1], True)
        view_default_3: "f32[512]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_3);  sum_dim_int_list = _shape_param_3 = None
        mul_tensor: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor_2, arg611_1);  arg611_1 = None
        mul_tensor_1: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor_2, arg27_1);  add_tensor_2 = arg27_1 = None
        sum_dim_int_list_1: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1], True);  mul_tensor = None
        view_default_4: "f32[512]" = torch.ops.aten.view.default(sum_dim_int_list_1, _shape_param_4);  sum_dim_int_list_1 = _shape_param_4 = None
        view_default_5: "f32[32768, 512]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_5);  mul_tensor_1 = _shape_param_5 = None
        permute_default: "f32[512, 32768]" = torch.ops.aten.permute.default(view_default_5, [1, 0])
        sum_dim_int_list_2: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_default_5, [0], True);  view_default_5 = None
        view_default_6: "f32[512]" = torch.ops.aten.view.default(sum_dim_int_list_2, _shape_param_6);  sum_dim_int_list_2 = _shape_param_6 = None
        return (view_default_3, view_default_4, permute_default, view_default_6)



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
