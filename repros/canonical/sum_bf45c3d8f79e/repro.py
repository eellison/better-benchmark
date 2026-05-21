"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train_001
Pattern hash: bf45c3d8f79e
Shape hash: ffc7b387
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
_shapes_config = "(T([512, 1280], b8), T([512, 1280], f32), T([512, 1280, 1, 1], b8), S([512, 1280, 1, 1]))"

class Repro(torch.nn.Module):
    def forward(self, arg476_1: "b8[512, 1280]", mm: "f32[512, 1280]", arg478_1: "b8[512, 1280, 1, 1]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[512, 1280]" = torch.ops.prims.convert_element_type.default(arg476_1, torch.float32);  arg476_1 = None
        mul_tensor: "f32[512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.25);  convert_element_type_default = None
        mul_tensor_1: "f32[512, 1280]" = torch.ops.aten.mul.Tensor(mm, mul_tensor);  mm = mul_tensor = None
        view_default: "f32[512, 1280, 1, 1]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_0);  mul_tensor_1 = _shape_param_0 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[512, 1280, 1, 1]" = torch.ops.aten.where.self(arg478_1, full_default, view_default);  arg478_1 = full_default = view_default = None
        sum_dim_int_list: "f32[1280]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3]);  where_self = None
        return sum_dim_int_list



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
