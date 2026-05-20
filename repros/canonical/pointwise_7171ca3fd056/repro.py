"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-2-5-linux.aws.a100_graph5
Pattern hash: 7171ca3fd056
Shape hash: e15c8f8d
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([512, 32000], f32), T([1, 512, 768, 2], f32), T([1, 512, 768], f32), T([1, 512, 768], b8), T([768, 768], f32), S([512, 768]))"

class Repro(torch.nn.Module):
    def forward(self, sub_2: "f32[512, 32000]", view_as_real_11: "f32[1, 512, 768, 2]", mul_332: "f32[1, 512, 768]", arg59_1: "b8[1, 512, 768]", arg5_1: "f32[768, 768]", _shape_param_0):
        # No stacktrace found for following nodes
        permute_default: "f32[32000, 512]" = torch.ops.aten.permute.default(sub_2, [1, 0]);  sub_2 = None
        select_int: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_11, 3, 0);  view_as_real_11 = None
        add_tensor: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_332, select_int);  mul_332 = select_int = None
        convert_element_type_default: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(arg59_1, torch.float32);  arg59_1 = None
        mul_tensor: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_tensor, mul_tensor);  add_tensor = mul_tensor = None
        view_default: "f32[512, 768]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_0);  mul_tensor_1 = _shape_param_0 = None
        permute_default_1: "f32[768, 768]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        permute_default_2: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_1, [1, 0]);  permute_default_1 = None
        return (permute_default, view_default, permute_default_2)


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
