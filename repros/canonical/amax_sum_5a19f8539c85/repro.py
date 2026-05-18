"""
Standalone repro captured via capture_hook.
Label: tritonbench_softmax_2048x50257
Pattern hash: 5a19f8539c85
Shape hash: 44070737
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f16[2048, 50257]"):
        # File: /tmp/scratch_space/better_benchmark/capture_tritonbench_ops.py:54 in softmax_fn, code: return F.softmax(x, dim=-1)
        convert_element_type_default: "f32[2048, 50257]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        amax_default: "f32[2048, 1]" = torch.ops.aten.amax.default(convert_element_type_default, [-1], True)
        sub_tensor: "f32[2048, 50257]" = torch.ops.aten.sub.Tensor(convert_element_type_default, amax_default);  convert_element_type_default = amax_default = None
        exp_default: "f32[2048, 50257]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[2048, 50257]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        convert_element_type_default_1: "f16[2048, 50257]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.float16);  div_tensor = None
        return convert_element_type_default_1


def _default_make_inputs():
    return [
    torch.randn([2048, 50257], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
