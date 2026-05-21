"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_train_001
Pattern hash: 6f80b8821398
Shape hash: 5f23fbcf
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
_shapes_config = "(T([32, 128, 512], f32), T([32, 128, 1], f32), T([32, 128, 512], f32), S([512]))"

class Repro(torch.nn.Module):
    def forward(self, arg496_1: "f32[32, 128, 512]", arg497_1: "f32[32, 128, 1]", mul_3: "f32[32, 128, 512]", _shape_param_0):
        # No stacktrace found for following nodes
        mul_tensor: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg496_1, arg497_1);  arg496_1 = arg497_1 = None
        mul_tensor_1: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_3, mul_tensor);  mul_3 = mul_tensor = None
        sum_dim_int_list: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1], True);  mul_tensor_1 = None
        view_default: "f32[512]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None
        return view_default



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
