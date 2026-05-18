"""
Standalone repro captured via capture_hook.
Label: tritonbench_fused_softmax_B2_S2048_H32
Pattern hash: e6697b63eeec
Shape hash: 5cd6ce18
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
    def forward(self, arg0_1: "bf16[2, 2048, 32, 2048]", arg1_1: "bf16[1, 1, 1, 2048]"):
        # File: /tmp/scratch_space/better_benchmark/capture_tritonbench_ops.py:68 in fused_softmax, code: return F.softmax(x.permute(0, 2, 1, 3) + bias, dim=-1)
        permute_default: "bf16[2, 32, 2048, 2048]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
        add_tensor: "bf16[2, 32, 2048, 2048]" = torch.ops.aten.add.Tensor(permute_default, arg1_1);  permute_default = arg1_1 = None
        clone_default: "bf16[2, 32, 2048, 2048]" = torch.ops.aten.clone.default(add_tensor, memory_format = torch.contiguous_format);  add_tensor = None
        convert_element_type_default: "f32[2, 32, 2048, 2048]" = torch.ops.prims.convert_element_type.default(clone_default, torch.float32);  clone_default = None
        amax_default: "f32[2, 32, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_default, [-1], True)
        sub_tensor: "f32[2, 32, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_default, amax_default);  convert_element_type_default = amax_default = None
        exp_default: "f32[2, 32, 2048, 2048]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[2, 32, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[2, 32, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        convert_element_type_default_1: "bf16[2, 32, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        return convert_element_type_default_1


def _default_make_inputs():
    return [
    torch.randn([2, 2048, 32, 2048], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 1, 1, 2048], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
