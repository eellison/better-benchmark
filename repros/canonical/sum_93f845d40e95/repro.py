"""
Standalone repro captured via capture_hook.
Label: vllm_Qwen_Qwen3-0.6B_001
Pattern hash: 93f845d40e95
Shape hash: e197ac67
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
_shapes_config = "(T([4, 512, 8, 128], f32), T([4, 512, 8, 1], f32), T([4, 512, 8, 128], bf16, stride=(524288, 128, 65536, 1)), S([128]))"

class Repro(torch.nn.Module):
    def forward(self, convert_element_type_17: "f32[4, 512, 8, 128]", arg859_1: "f32[4, 512, 8, 1]", permute_22: "bf16[4, 512, 8, 128]", _shape_param_0):
        # No stacktrace found for following nodes
        mul_tensor: "f32[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_17, arg859_1);  convert_element_type_17 = arg859_1 = None
        convert_element_type_default: "bf16[4, 512, 8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        mul_tensor_1: "bf16[4, 512, 8, 128]" = torch.ops.aten.mul.Tensor(permute_22, convert_element_type_default);  permute_22 = convert_element_type_default = None
        sum_dim_int_list: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1, 2], True);  mul_tensor_1 = None
        view_default: "bf16[128]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None
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
