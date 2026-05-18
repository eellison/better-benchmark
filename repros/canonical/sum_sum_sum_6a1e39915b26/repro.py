"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['1', '1', '512'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '512'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '128', '512'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '512'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_715: "f32[32768, 512]", mul_691: "f32[256, 128, 512]", mm_721: "f32[32768, 512]", mm_723: "f32[32768, 512]", add_1: "f32[256, 128, 512]", primals_8: "f32[512]", full_default_3: "f32[]", full_default: "i64[256, 128]", primals_3: "i64[1, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:255 in forward, code: self.value(value_tensor)
        reshape_default: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(mm_715, _shape_param_0);  mm_715 = _shape_param_0 = None
        add_tensor: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_691, reshape_default);  mul_691 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:409 in forward, code: layer_input = self.dense(hidden_states)
        reshape_default_1: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(mm_721, _shape_param_1);  mm_721 = _shape_param_1 = None
        add_tensor_1: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_1);  add_tensor = reshape_default_1 = None
        reshape_default_2: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(mm_723, _shape_param_2);  mm_723 = _shape_param_2 = None
        add_tensor_2: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_2);  add_tensor_1 = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:138 in forward, code: return input_tensor * self.weight + self.bias
        sum_dim_int_list: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(add_tensor_2, [0, 1], True)
        reshape_default_3: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_3);  sum_dim_int_list = _shape_param_3 = None
        mul_tensor: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor_2, add_1);  add_1 = None
        mul_tensor_1: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor_2, primals_8);  add_tensor_2 = primals_8 = None
        sum_dim_int_list_1: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1], True);  mul_tensor = None
        reshape_default_4: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, _shape_param_4);  sum_dim_int_list_1 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:214 in forward, code: embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        sum_dim_int_list_2: "f32[1, 128, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:213 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        full_default: "b8[256, 128, 1]" = torch.ops.aten.full.default([256, 128, 1], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[256, 128, 512]" = torch.ops.aten.where.self(full_default, full_default_3, mul_tensor_1);  full_default = None
        full_default_4: "f32[2, 512]" = torch.ops.aten.full.default([2, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_5 = full_default
        index_put_default: "f32[2, 512]" = torch.ops.aten.index_put.default(full_default_4, [full_default_5], where_self, True);  full_default_4 = full_default_5 = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:184 in forward, code: position_ids = self.position_ids[:, :seq_length]
        slice_tensor: "i64[1, 128]" = torch.ops.aten.slice.Tensor(primals_3, 1, 0, 128);  primals_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:212 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        eq_scalar: "b8[1, 128]" = torch.ops.aten.eq.Scalar(slice_tensor, -1)
        unsqueeze_default: "b8[1, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self_1: "f32[1, 128, 512]" = torch.ops.aten.where.self(unsqueeze_default, full_default_3, sum_dim_int_list_2);  unsqueeze_default = full_default_3 = sum_dim_int_list_2 = None
        full_default_6: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[512, 512]" = torch.ops.aten.index_put.default(full_default_6, [slice_tensor], where_self_1, True);  full_default_6 = slice_tensor = where_self_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:208 in forward, code: inputs_embeds = self.embedding_transformation(inputs_embeds)
        reshape_default_5: "f32[32768, 512]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_5);  mul_tensor_1 = _shape_param_5 = None
        permute_default: "f32[512, 32768]" = torch.ops.aten.permute.default(reshape_default_5, [1, 0])
        sum_dim_int_list_3: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(reshape_default_5, [0], True);  reshape_default_5 = None
        reshape_default_6: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_6);  sum_dim_int_list_3 = _shape_param_6 = None
        return (reshape_default_3, reshape_default_4, index_put_default, index_put_default_1, permute_default, reshape_default_6)


def _default_make_inputs():
    return [
    torch.randn([32768, 512], dtype=torch.float32, device='cuda'),
    torch.randn([256, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32768, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32768, 512], dtype=torch.float32, device='cuda'),
    torch.randn([256, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.tensor(1),  # full_default_5 (unknown shape)
    torch.randint(0, 512, [1, 512], dtype=torch.int64, device='cuda'),
    [256, 128, 512],  # _shape_param_0
    [256, 128, 512],  # _shape_param_1
    [256, 128, 512],  # _shape_param_2
    [512],  # _shape_param_3
    [512],  # _shape_param_4
    [32768, 512],  # _shape_param_5
    [512],  # _shape_param_6
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
