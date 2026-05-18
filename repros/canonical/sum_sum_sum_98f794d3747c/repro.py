"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['64', '128', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['64', '128', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_6: "f32[8192, 1024]", mm_8: "f32[8192, 1024]", mm_10: "f32[8192, 1024]", primals_1: "f32[1024]", primals_3: "f32[64, 128, 1024]", getitem_1: "f32[64, 128, 1]", rsqrt: "f32[64, 128, 1]", add_6: "f32[64, 128, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:304 in forward, code: value_states = self.v_proj(current_states)
        reshape_default: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(mm_6, [64, 128, 1024]);  mm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:303 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_1: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(mm_8, [64, 128, 1024]);  mm_8 = None
        add_tensor: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(reshape_default, reshape_default_1);  reshape_default = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:283 in forward, code: query_states = self.q_proj(hidden_states).view(*q_input_shape).transpose(1, 2)
        reshape_default_2: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(mm_10, [64, 128, 1024]);  mm_10 = None
        add_tensor_1: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_2);  add_tensor = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:380 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        mul_tensor: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(add_tensor_1, primals_1);  primals_1 = None
        mul_tensor_1: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, 1024)
        sum_dim_int_list: "f32[64, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        sub_tensor: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(primals_3, getitem_1);  primals_3 = getitem_1 = None
        mul_tensor_2: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt);  sub_tensor = None
        mul_tensor_3: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1: "f32[64, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        mul_tensor_4: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_1: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor: "f32[64, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None
        mul_tensor_5: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_6: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(add_tensor_1, mul_tensor_2);  mul_tensor_2 = None
        sum_dim_int_list_2: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_3: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_tensor_1, [0, 1]);  add_tensor_1 = None
        add_tensor_2: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(add_6, mul_tensor_5);  add_6 = mul_tensor_5 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, add_tensor_2)


def _default_make_inputs():
    return [
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 128, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64, 128, 1024], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
