"""
Standalone repro captured via capture_hook.
Label: hf_GPTNeoForCausalLM_training
Pattern hash: e8f6d1c3f9bf
Shape hash: 94041ed4
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, div: "f32[8, 16, 128, 128]", _shape_param_0, _shape_param_1, bmm_141: "f32[128, 128, 128]", _shape_param_2, primals_9: "b8[1, 1, 2048, 2048]", full_default_1: "f32[]", _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_default: "f32[8, 16, 128, 128]" = torch.ops.aten.expand.default(div, _shape_param_0);  _shape_param_0 = None
        reshape_default: "f32[128, 128, 128]" = torch.ops.aten.reshape.default(expand_default, _shape_param_1);  expand_default = _shape_param_1 = None
        permute_default: "f32[128, 128, 128]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1]);  reshape_default = None
        reshape_default_1: "f32[8, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_141, _shape_param_2);  bmm_141 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_tensor: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(reshape_default_1, div);  reshape_default_1 = None
        sum_dim_int_list: "f32[8, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [-1], True)
        neg_default: "f32[8, 16, 128, 128]" = torch.ops.aten.neg.default(div);  div = None
        fma_default: "f32[8, 16, 128, 128]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor);  neg_default = sum_dim_int_list = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_tensor: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_9, 2, 0, 128);  primals_9 = None
        slice_tensor_1: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_tensor, 3, 0, 128);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_self: "f32[8, 16, 128, 128]" = torch.ops.aten.where.self(slice_tensor_1, fma_default, full_default_1);  slice_tensor_1 = fma_default = full_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        reshape_default_2: "f32[128, 128, 128]" = torch.ops.aten.reshape.default(where_self, _shape_param_3);  where_self = _shape_param_3 = None
        return (permute_default, reshape_default_2)


def _default_make_inputs():
    return [
    torch.randn([8, 16, 128, 128], dtype=torch.float32, device='cuda'),
    [8, 16, 128, 128],  # _shape_param_0
    [128, 128, 128],  # _shape_param_1
    torch.randn([128, 128, 128], dtype=torch.float32, device='cuda'),
    [8, 16, 128, 128],  # _shape_param_2
    torch.randint(0, 2, [1, 1, 2048, 2048], dtype=torch.bool, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    [128, 128, 128],  # _shape_param_3
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
