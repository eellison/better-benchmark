"""
Standalone repro captured via capture_hook.
Label: genai_softmax_fwd_32768x256
Pattern hash: 3ed297ef02cd
Shape hash: 44f3d1cd
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([32768, 256], bf16))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[32768, 256]"):
        # File: /tmp/scratch_space/better_benchmark/capture_genai_kernels.py:217 in sm_fwd, code: return F.softmax(x, dim=-1)
        convert_element_type_default: "f32[32768, 256]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        amax_default: "f32[32768, 1]" = torch.ops.aten.amax.default(convert_element_type_default, [-1], True)
        sub_tensor: "f32[32768, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_default, amax_default);  convert_element_type_default = amax_default = None
        exp_default: "f32[32768, 256]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[32768, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[32768, 256]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        convert_element_type_default_1: "bf16[32768, 256]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        return convert_element_type_default_1


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
