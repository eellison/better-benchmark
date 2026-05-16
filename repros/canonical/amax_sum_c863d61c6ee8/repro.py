"""
Standalone repro captured via capture_hook.
Label: genai_causal_attn_decode
Pattern hash: c863d61c6ee8
Shape hash: 5514944f
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm: "bf16[1024, 1, 1]", _shape_param_0, _shape_param_1, _shape_param_2, arg2_1: "bf16[32, 32, 1, 128]", _shape_param_3, _shape_param_4, _shape_param_5):
        # File: /tmp/scratch_space/better_benchmark/capture_genai_varied_shapes.py:125 in causal_attention_pattern, code: causal_mask = torch.triu(
        iota_default: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_default, -2);  iota_default = None
        iota_default_1: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_1: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_default_1, -1);  iota_default_1 = None
        sub_tensor: "i64[1, 1]" = torch.ops.aten.sub.Tensor(unsqueeze_default, unsqueeze_default_1);  unsqueeze_default = unsqueeze_default_1 = None
        ge_scalar: "b8[1, 1]" = torch.ops.aten.ge.Scalar(sub_tensor, 1);  sub_tensor = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_varied_shapes.py:126 in causal_attention_pattern, code: torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool),
        full_default: "b8[1, 1]" = torch.ops.aten.full.default([1, 1], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/scratch_space/better_benchmark/capture_genai_varied_shapes.py:125 in causal_attention_pattern, code: causal_mask = torch.triu(
        logical_and_default: "b8[1, 1]" = torch.ops.aten.logical_and.default(ge_scalar, full_default);  ge_scalar = full_default = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_varied_shapes.py:129 in causal_attention_pattern, code: scores = scores.masked_fill(causal_mask, float("-inf"))
        full_default_1: "bf16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /tmp/scratch_space/better_benchmark/capture_genai_varied_shapes.py:123 in causal_attention_pattern, code: scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
        reshape_default: "bf16[32, 32, 1, 1]" = torch.ops.aten.reshape.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        div_tensor: "bf16[32, 32, 1, 1]" = torch.ops.aten.div.Tensor(reshape_default, 11.313708498984761);  reshape_default = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_varied_shapes.py:129 in causal_attention_pattern, code: scores = scores.masked_fill(causal_mask, float("-inf"))
        where_self: "bf16[32, 32, 1, 1]" = torch.ops.aten.where.self(logical_and_default, full_default_1, div_tensor);  logical_and_default = full_default_1 = div_tensor = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_varied_shapes.py:130 in causal_attention_pattern, code: scores = torch.softmax(scores, dim=-1)
        convert_element_type_default: "f32[32, 32, 1, 1]" = torch.ops.prims.convert_element_type.default(where_self, torch.float32);  where_self = None
        amax_default: "f32[32, 32, 1, 1]" = torch.ops.aten.amax.default(convert_element_type_default, [-1], True)
        sub_tensor_1: "f32[32, 32, 1, 1]" = torch.ops.aten.sub.Tensor(convert_element_type_default, amax_default);  convert_element_type_default = amax_default = None
        exp_default: "f32[32, 32, 1, 1]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        sum_dim_int_list: "f32[32, 32, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor_1: "f32[32, 32, 1, 1]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        convert_element_type_default_1: "bf16[32, 32, 1, 1]" = torch.ops.prims.convert_element_type.default(div_tensor_1, torch.bfloat16);  div_tensor_1 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_varied_shapes.py:131 in causal_attention_pattern, code: out = torch.matmul(scores, V)
        expand_default: "bf16[32, 32, 1, 1]" = torch.ops.aten.expand.default(convert_element_type_default_1, _shape_param_1);  convert_element_type_default_1 = _shape_param_1 = None
        reshape_default_1: "bf16[1024, 1, 1]" = torch.ops.aten.reshape.default(expand_default, _shape_param_2);  expand_default = _shape_param_2 = None
        convert_element_type_default_2: "f32[1024, 1, 1]" = torch.ops.prims.convert_element_type.default(reshape_default_1, torch.float32);  reshape_default_1 = None
        expand_default_1: "bf16[32, 32, 1, 128]" = torch.ops.aten.expand.default(arg2_1, _shape_param_3);  arg2_1 = _shape_param_3 = None
        reshape_default_2: "bf16[1024, 1, 128]" = torch.ops.aten.reshape.default(expand_default_1, _shape_param_4);  expand_default_1 = _shape_param_4 = None
        convert_element_type_default_3: "f32[1024, 1, 128]" = torch.ops.prims.convert_element_type.default(reshape_default_2, torch.float32);  reshape_default_2 = None
        mul_tensor: "f32[1024, 1, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, convert_element_type_default_3);  convert_element_type_default_2 = convert_element_type_default_3 = None
        convert_element_type_default_4: "bf16[1024, 1, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        reshape_default_3: "bf16[32, 32, 1, 128]" = torch.ops.aten.reshape.default(convert_element_type_default_4, _shape_param_5);  convert_element_type_default_4 = _shape_param_5 = None
        return reshape_default_3


def _default_make_inputs():
    return [
    torch.randn([1024, 1, 1], dtype=torch.bfloat16, device='cuda'),
    [32, 32, 1, 1],  # _shape_param_0
    [32, 32, 1, 1],  # _shape_param_1
    [1024, 1, 1],  # _shape_param_2
    torch.randn([32, 32, 1, 128], dtype=torch.bfloat16, device='cuda'),
    [32, 32, 1, 128],  # _shape_param_3
    [1024, 1, 128],  # _shape_param_4
    [32, 32, 1, 128],  # _shape_param_5
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
