"""
Standalone repro captured via capture_hook.
Label: genai_gqa_decode
Pattern hash: 0e92e1c207b5
Shape hash: cc4e903d
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
    def forward(self, bmm: "bf16[1024, 1, 1]", _shape_param_0, _shape_param_1, _shape_param_2, arg1_1: "bf16[32, 8, 1, 128]", _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # File: /tmp/scratch_space/better_benchmark/capture_genai_varied_shapes.py:147 in gqa_pattern, code: scores = torch.matmul(Q, K_expanded.transpose(-2, -1)) / math.sqrt(head_dim)
        reshape_default: "bf16[32, 32, 1, 1]" = torch.ops.aten.reshape.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None

        # No stacktrace found for following nodes
        convert_element_type_default: "f32[32, 32, 1, 1]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        mul_tensor: "f32[32, 32, 1, 1]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1);  convert_element_type_default = None
        amax_default: "f32[32, 32, 1, 1]" = torch.ops.aten.amax.default(mul_tensor, [-1], True)
        sub_tensor: "f32[32, 32, 1, 1]" = torch.ops.aten.sub.Tensor(mul_tensor, amax_default);  mul_tensor = amax_default = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_varied_shapes.py:148 in gqa_pattern, code: scores = torch.softmax(scores, dim=-1)
        div_tensor: "f32[32, 32, 1, 1]" = torch.ops.aten.div.Tensor(sub_tensor, 11.313708498984761);  sub_tensor = None
        exp_default: "f32[32, 32, 1, 1]" = torch.ops.aten.exp.default(div_tensor);  div_tensor = None
        sum_dim_int_list: "f32[32, 32, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor_1: "f32[32, 32, 1, 1]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        convert_element_type_default_1: "bf16[32, 32, 1, 1]" = torch.ops.prims.convert_element_type.default(div_tensor_1, torch.bfloat16);  div_tensor_1 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_varied_shapes.py:149 in gqa_pattern, code: out = torch.matmul(scores, V_expanded)
        expand_default: "bf16[32, 32, 1, 1]" = torch.ops.aten.expand.default(convert_element_type_default_1, _shape_param_1);  convert_element_type_default_1 = _shape_param_1 = None
        reshape_default_1: "bf16[1024, 1, 1]" = torch.ops.aten.reshape.default(expand_default, _shape_param_2);  expand_default = _shape_param_2 = None
        convert_element_type_default_2: "f32[1024, 1, 1]" = torch.ops.prims.convert_element_type.default(reshape_default_1, torch.float32);  reshape_default_1 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_varied_shapes.py:144 in gqa_pattern, code: V_expanded = V[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, hd)
        unsqueeze_default: "bf16[32, 8, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(arg1_1, 2);  arg1_1 = None
        expand_default_1: "bf16[32, 8, 4, 1, 128]" = torch.ops.aten.expand.default(unsqueeze_default, _shape_param_3);  unsqueeze_default = _shape_param_3 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_varied_shapes.py:145 in gqa_pattern, code: V_expanded = V_expanded.reshape(batch, num_kv_heads * n_rep, slen, hd)
        clone_default: "bf16[32, 8, 4, 1, 128]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_2: "bf16[32, 32, 1, 128]" = torch.ops.aten.reshape.default(clone_default, _shape_param_4);  clone_default = _shape_param_4 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_varied_shapes.py:149 in gqa_pattern, code: out = torch.matmul(scores, V_expanded)
        expand_default_2: "bf16[32, 32, 1, 128]" = torch.ops.aten.expand.default(reshape_default_2, _shape_param_5);  reshape_default_2 = _shape_param_5 = None
        reshape_default_3: "bf16[1024, 1, 128]" = torch.ops.aten.reshape.default(expand_default_2, _shape_param_6);  expand_default_2 = _shape_param_6 = None
        convert_element_type_default_3: "f32[1024, 1, 128]" = torch.ops.prims.convert_element_type.default(reshape_default_3, torch.float32);  reshape_default_3 = None
        mul_tensor_1: "f32[1024, 1, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, convert_element_type_default_3);  convert_element_type_default_2 = convert_element_type_default_3 = None
        convert_element_type_default_4: "bf16[1024, 1, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None
        reshape_default_4: "bf16[32, 32, 1, 128]" = torch.ops.aten.reshape.default(convert_element_type_default_4, _shape_param_7);  convert_element_type_default_4 = _shape_param_7 = None
        return reshape_default_4


def _default_make_inputs():
    return [
    torch.randn([1024, 1, 1], dtype=torch.bfloat16, device='cuda'),
    [32, 32, 1, 1],  # _shape_param_0
    [32, 32, 1, 1],  # _shape_param_1
    [1024, 1, 1],  # _shape_param_2
    torch.randn([32, 8, 1, 128], dtype=torch.bfloat16, device='cuda'),
    [32, 8, 4, 1, 128],  # _shape_param_3
    [32, 32, 1, 128],  # _shape_param_4
    [32, 32, 1, 128],  # _shape_param_5
    [1024, 1, 128],  # _shape_param_6
    [32, 32, 1, 128],  # _shape_param_7
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
