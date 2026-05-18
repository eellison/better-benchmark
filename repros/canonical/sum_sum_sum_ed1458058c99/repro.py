"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['1', '16384'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '16384'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '16384'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '16384'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '16384'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '16384'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '16384'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '16384'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '16384'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '16384'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '16384'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '16384'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch
from math import inf
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, view_282: "f32[4096, 16384]", view_310: "f32[4096, 16384]", view_338: "f32[4096, 16384]", view_366: "f32[4096, 16384]", view_394: "f32[4096, 16384]", view_422: "f32[4096, 16384]", view_450: "f32[4096, 16384]", view_478: "f32[4096, 16384]", view_506: "f32[4096, 16384]", view_534: "f32[4096, 16384]", view_562: "f32[4096, 16384]", mm_136: "f32[4096, 16384]", addmm_5: "f32[4096, 16384]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        sum_dim_int_list: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_282, [0], True);  view_282 = None
        reshape_default: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list, [16384]);  sum_dim_int_list = None
        sum_dim_int_list_1: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_310, [0], True);  view_310 = None
        reshape_default_1: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, [16384]);  sum_dim_int_list_1 = None
        add_tensor: "f32[16384]" = torch.ops.aten.add.Tensor(reshape_default, reshape_default_1);  reshape_default = reshape_default_1 = None
        sum_dim_int_list_2: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_338, [0], True);  view_338 = None
        reshape_default_2: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, [16384]);  sum_dim_int_list_2 = None
        add_tensor_1: "f32[16384]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_2);  add_tensor = reshape_default_2 = None
        sum_dim_int_list_3: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_366, [0], True);  view_366 = None
        reshape_default_3: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, [16384]);  sum_dim_int_list_3 = None
        add_tensor_2: "f32[16384]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_3);  add_tensor_1 = reshape_default_3 = None
        sum_dim_int_list_4: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_394, [0], True);  view_394 = None
        reshape_default_4: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, [16384]);  sum_dim_int_list_4 = None
        add_tensor_3: "f32[16384]" = torch.ops.aten.add.Tensor(add_tensor_2, reshape_default_4);  add_tensor_2 = reshape_default_4 = None
        sum_dim_int_list_5: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_422, [0], True);  view_422 = None
        reshape_default_5: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_5, [16384]);  sum_dim_int_list_5 = None
        add_tensor_4: "f32[16384]" = torch.ops.aten.add.Tensor(add_tensor_3, reshape_default_5);  add_tensor_3 = reshape_default_5 = None
        sum_dim_int_list_6: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_450, [0], True);  view_450 = None
        reshape_default_6: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, [16384]);  sum_dim_int_list_6 = None
        add_tensor_5: "f32[16384]" = torch.ops.aten.add.Tensor(add_tensor_4, reshape_default_6);  add_tensor_4 = reshape_default_6 = None
        sum_dim_int_list_7: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_478, [0], True);  view_478 = None
        reshape_default_7: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, [16384]);  sum_dim_int_list_7 = None
        add_tensor_6: "f32[16384]" = torch.ops.aten.add.Tensor(add_tensor_5, reshape_default_7);  add_tensor_5 = reshape_default_7 = None
        sum_dim_int_list_8: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_506, [0], True);  view_506 = None
        reshape_default_8: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_8, [16384]);  sum_dim_int_list_8 = None
        add_tensor_7: "f32[16384]" = torch.ops.aten.add.Tensor(add_tensor_6, reshape_default_8);  add_tensor_6 = reshape_default_8 = None
        sum_dim_int_list_9: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_534, [0], True);  view_534 = None
        reshape_default_9: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_9, [16384]);  sum_dim_int_list_9 = None
        add_tensor_8: "f32[16384]" = torch.ops.aten.add.Tensor(add_tensor_7, reshape_default_9);  add_tensor_7 = reshape_default_9 = None
        sum_dim_int_list_10: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_562, [0], True);  view_562 = None
        reshape_default_10: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_10, [16384]);  sum_dim_int_list_10 = None
        add_tensor_9: "f32[16384]" = torch.ops.aten.add.Tensor(add_tensor_8, reshape_default_10);  add_tensor_8 = reshape_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        reshape_default_11: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(mm_136, [8, 512, 16384]);  mm_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        reshape_default_12: "f32[8, 512, 16384]" = torch.ops.aten.reshape.default(addmm_5, [8, 512, 16384]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_tensor: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(reshape_default_12, 0.5)
        mul_tensor_1: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(reshape_default_11, mul_tensor);  mul_tensor = None
        pow_tensor_scalar: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(reshape_default_12, 3.0)
        mul_tensor_2: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor_10: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(reshape_default_12, mul_tensor_2);  mul_tensor_2 = None
        mul_tensor_3: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_tensor_10, 0.7978845608028654);  add_tensor_10 = None
        tanh_default: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_tensor_3);  mul_tensor_3 = None
        add_tensor_11: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_default, 1.0)
        mul_tensor_4: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(reshape_default_11, add_tensor_11);  reshape_default_11 = add_tensor_11 = None
        mul_tensor_5: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(tanh_default, tanh_default);  tanh_default = None
        sub_tensor: "f32[8, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_tensor_5);  mul_tensor_5 = None
        mul_tensor_6: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_tensor_1, sub_tensor);  mul_tensor_1 = sub_tensor = None
        mul_tensor_7: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_tensor_6, 0.7978845608028654);  mul_tensor_6 = None
        mul_tensor_8: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_tensor_7, 0.044715)
        pow_tensor_scalar_1: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(reshape_default_12, 2.0);  reshape_default_12 = None
        mul_scalar: "f32[8, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 3.0);  pow_tensor_scalar_1 = None
        mul_tensor_9: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_tensor_8, mul_scalar);  mul_tensor_8 = mul_scalar = None
        add_tensor_12: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(mul_tensor_7, mul_tensor_9);  mul_tensor_7 = mul_tensor_9 = None
        mul_tensor_10: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_tensor_4, 0.5);  mul_tensor_4 = None
        add_tensor_13: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(add_tensor_12, mul_tensor_10);  add_tensor_12 = mul_tensor_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        reshape_default_13: "f32[4096, 16384]" = torch.ops.aten.reshape.default(add_tensor_13, [4096, 16384]);  add_tensor_13 = None
        permute_default: "f32[16384, 4096]" = torch.ops.aten.permute.default(reshape_default_13, [1, 0])
        sum_dim_int_list_11: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(reshape_default_13, [0], True);  reshape_default_13 = None
        reshape_default_14: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, [16384]);  sum_dim_int_list_11 = None
        add_tensor_14: "f32[16384]" = torch.ops.aten.add.Tensor(add_tensor_9, reshape_default_14);  add_tensor_9 = reshape_default_14 = None
        return (permute_default, add_tensor_14)


def _default_make_inputs():
    return [
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 16384], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
