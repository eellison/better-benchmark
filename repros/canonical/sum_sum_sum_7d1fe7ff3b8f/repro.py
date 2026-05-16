"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['8', '512', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['8', '512', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch
from math import inf
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[4096, 128]", primals_29: "f32[128]", addmm_73: "f32[4096, 128]", getitem_51: "f32[8, 512, 1]", rsqrt_25: "f32[8, 512, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:541 in forward, code: hidden_states = self.decoder(hidden_states)
        reshape_default: "f32[8, 512, 128]" = torch.ops.aten.reshape.default(mm, [8, 512, 128]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:540 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        mul_tensor: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(reshape_default, primals_29);  primals_29 = None
        mul_tensor_1: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, 128)
        sum_dim_int_list: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:538 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default_1: "f32[8, 512, 128]" = torch.ops.aten.reshape.default(addmm_73, [8, 512, 128]);  addmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_tensor_2: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(reshape_default_1, 0.5)
        pow_tensor_scalar: "f32[8, 512, 128]" = torch.ops.aten.pow.Tensor_Scalar(reshape_default_1, 3.0)
        mul_tensor_3: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(reshape_default_1, mul_tensor_3);  mul_tensor_3 = None
        mul_tensor_4: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(add_tensor, 0.7978845608028654);  add_tensor = None
        tanh_default: "f32[8, 512, 128]" = torch.ops.aten.tanh.default(mul_tensor_4);  mul_tensor_4 = None
        add_tensor_1: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(tanh_default, 1.0)
        mul_tensor_5: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_2, add_tensor_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:540 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        sub_tensor: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(mul_tensor_5, getitem_51);  mul_tensor_5 = getitem_51 = None
        mul_tensor_6: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_25);  sub_tensor = None
        mul_tensor_7: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_6);  mul_tensor = None
        sum_dim_int_list_1: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [2], True);  mul_tensor_7 = None
        mul_tensor_8: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_6, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_1: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_8);  sub_tensor_1 = mul_tensor_8 = None
        div_tensor: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 128);  rsqrt_25 = None
        mul_tensor_9: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_10: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor_6);  mul_tensor_6 = None
        sum_dim_int_list_2: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        sum_dim_int_list_3: "f32[128]" = torch.ops.aten.sum.dim_IntList(reshape_default, [0, 1]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_tensor_11: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_9, mul_tensor_2);  mul_tensor_2 = None
        mul_tensor_12: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_9, add_tensor_1);  mul_tensor_9 = add_tensor_1 = None
        mul_tensor_13: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(tanh_default, tanh_default);  tanh_default = None
        sub_tensor_3: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(1, mul_tensor_13);  mul_tensor_13 = None
        mul_tensor_14: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_11, sub_tensor_3);  mul_tensor_11 = sub_tensor_3 = None
        mul_tensor_15: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_14, 0.7978845608028654);  mul_tensor_14 = None
        mul_tensor_16: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_15, 0.044715)
        pow_tensor_scalar_1: "f32[8, 512, 128]" = torch.ops.aten.pow.Tensor_Scalar(reshape_default_1, 2.0);  reshape_default_1 = None
        mul_scalar: "f32[8, 512, 128]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 3.0);  pow_tensor_scalar_1 = None
        mul_tensor_17: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_16, mul_scalar);  mul_tensor_16 = mul_scalar = None
        add_tensor_2: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(mul_tensor_15, mul_tensor_17);  mul_tensor_15 = mul_tensor_17 = None
        mul_tensor_18: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_12, 0.5);  mul_tensor_12 = None
        add_tensor_3: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_18);  add_tensor_2 = mul_tensor_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:538 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default_2: "f32[4096, 128]" = torch.ops.aten.reshape.default(add_tensor_3, [4096, 128]);  add_tensor_3 = None
        permute_default: "f32[128, 4096]" = torch.ops.aten.permute.default(reshape_default_2, [1, 0])
        sum_dim_int_list_4: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(reshape_default_2, [0], True);  reshape_default_2 = None
        reshape_default_3: "f32[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, [128]);  sum_dim_int_list_4 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, permute_default, reshape_default_3)


def _default_make_inputs():
    return [
    torch.randn([4096, 128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
