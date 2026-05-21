"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_train_001
Pattern hash: bee8f9028fe0
Shape hash: d56b0132
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
_shapes_config = "(T([4096, 4096], f32), T([4096], f32), T([8, 512, 4096], f32), T([8, 512, 1], f32), S([8, 512, 4096]), S([4096, 4096]))"

class Repro(torch.nn.Module):
    def forward(self, mm_2: "f32[4096, 4096]", arg12_1: "f32[4096]", arg114_1: "f32[8, 512, 4096]", arg124_1: "f32[8, 512, 1]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[8, 512, 4096]" = torch.ops.aten.view.default(mm_2, _shape_param_0);  mm_2 = _shape_param_0 = None
        mul_tensor: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(view_default, arg12_1);  view_default = arg12_1 = None
        mul_tensor_1: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor, 4096)
        sum_dim_int_list: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor, arg114_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg114_1, sum_dim_int_list_1);  arg114_1 = sum_dim_int_list_1 = None
        sub_tensor: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg124_1, sub_tensor_1);  arg124_1 = sub_tensor_1 = None
        view_default_1: "f32[4096, 4096]" = torch.ops.aten.view.default(mul_tensor_4, _shape_param_1);  mul_tensor_4 = _shape_param_1 = None
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
