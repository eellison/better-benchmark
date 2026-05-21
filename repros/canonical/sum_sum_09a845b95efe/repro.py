"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Reformer_train_014
Pattern hash: 09a845b95efe
Shape hash: e9703142
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
_shapes_config = "(T([8, 4096, 256], f32), T([8, 64, 1, 1], b8), S([8, 64, 64, 256]), S([1, 64, 192]), S([64, 1, 64]))"

class Repro(torch.nn.Module):
    def forward(self, arg3_1: "f32[8, 4096, 256]", arg2_1: "b8[8, 64, 1, 1]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[8, 64, 64, 256]" = torch.ops.aten.view.default(arg3_1, _shape_param_0);  arg3_1 = _shape_param_0 = None
        permute_default: "f32[8, 64, 64, 256]" = torch.ops.aten.permute.default(view_default, [0, 2, 1, 3]);  view_default = None
        convert_element_type_default: "f32[8, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        div_scalar: "f32[8, 64, 1, 1]" = torch.ops.aten.div.Scalar(convert_element_type_default, 0.95);  convert_element_type_default = None
        mul_tensor: "f32[8, 64, 64, 256]" = torch.ops.aten.mul.Tensor(permute_default, div_scalar);  permute_default = div_scalar = None
        permute_default_1: "f32[8, 64, 64, 256]" = torch.ops.aten.permute.default(mul_tensor, [0, 2, 1, 3]);  mul_tensor = None
        slice_tensor: "f32[8, 64, 64, 64]" = torch.ops.aten.slice.Tensor(permute_default_1, 3, 0, 64)
        slice_tensor_1: "f32[8, 64, 64, 192]" = torch.ops.aten.slice.Tensor(permute_default_1, 3, 64, 256);  permute_default_1 = None
        sum_dim_int_list: "f32[1, 1, 64, 192]" = torch.ops.aten.sum.dim_IntList(slice_tensor_1, [0, 1], True);  slice_tensor_1 = None
        view_default_1: "f32[1, 64, 192]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_1);  sum_dim_int_list = _shape_param_1 = None
        sum_dim_int_list_1: "f32[1, 64, 1, 64]" = torch.ops.aten.sum.dim_IntList(slice_tensor, [0, 2], True);  slice_tensor = None
        view_default_2: "f32[64, 1, 64]" = torch.ops.aten.view.default(sum_dim_int_list_1, _shape_param_2);  sum_dim_int_list_1 = _shape_param_2 = None
        return (view_default_1, view_default_2)



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
