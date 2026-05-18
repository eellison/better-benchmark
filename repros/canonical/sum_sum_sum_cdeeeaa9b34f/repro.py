"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['8', '512', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['8', '512', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '4096'], reduction_ranges=[]
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
    def forward(self, add_123: "f32[8, 512, 4096]", mul_114: "f32[8, 512, 4096]", view_285: "f32[4096, 4096]", add_135: "f32[8, 512, 4096]", mul_104: "f32[8, 512, 4096]", view_313: "f32[4096, 4096]", add_157: "f32[8, 512, 4096]", mul_94: "f32[8, 512, 4096]", view_341: "f32[4096, 4096]", add_179: "f32[8, 512, 4096]", mul_84: "f32[8, 512, 4096]", view_369: "f32[4096, 4096]", add_201: "f32[8, 512, 4096]", mul_74: "f32[8, 512, 4096]", view_397: "f32[4096, 4096]", add_223: "f32[8, 512, 4096]", mul_64: "f32[8, 512, 4096]", view_425: "f32[4096, 4096]", add_245: "f32[8, 512, 4096]", mul_54: "f32[8, 512, 4096]", view_453: "f32[4096, 4096]", add_267: "f32[8, 512, 4096]", mul_44: "f32[8, 512, 4096]", view_481: "f32[4096, 4096]", add_289: "f32[8, 512, 4096]", mul_34: "f32[8, 512, 4096]", view_509: "f32[4096, 4096]", add_311: "f32[8, 512, 4096]", mul_24: "f32[8, 512, 4096]", view_537: "f32[4096, 4096]", add_333: "f32[8, 512, 4096]", mul_14: "f32[8, 512, 4096]", view_565: "f32[4096, 4096]", mm_138: "f32[4096, 4096]", mul_437: "f32[8, 512, 4096]", primals_19: "f32[4096]", mul_4: "f32[8, 512, 4096]", div_38: "f32[8, 512, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_tensor: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_123, mul_114);  mul_114 = None
        sum_dim_int_list: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_1: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_123, [0, 1]);  add_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_2: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_285, [0], True);  view_285 = None
        reshape_default: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, [4096]);  sum_dim_int_list_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_tensor_1: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_135, mul_104);  mul_104 = None
        sum_dim_int_list_3: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_4: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_135, [0, 1]);  add_135 = None
        add_tensor: "f32[4096]" = torch.ops.aten.add.Tensor(sum_dim_int_list, sum_dim_int_list_3);  sum_dim_int_list = sum_dim_int_list_3 = None
        add_tensor_1: "f32[4096]" = torch.ops.aten.add.Tensor(sum_dim_int_list_1, sum_dim_int_list_4);  sum_dim_int_list_1 = sum_dim_int_list_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_5: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_313, [0], True);  view_313 = None
        reshape_default_1: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_5, [4096]);  sum_dim_int_list_5 = None
        add_tensor_2: "f32[4096]" = torch.ops.aten.add.Tensor(reshape_default, reshape_default_1);  reshape_default = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_tensor_2: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_157, mul_94);  mul_94 = None
        sum_dim_int_list_6: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1]);  mul_tensor_2 = None
        sum_dim_int_list_7: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_157, [0, 1]);  add_157 = None
        add_tensor_3: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor, sum_dim_int_list_6);  add_tensor = sum_dim_int_list_6 = None
        add_tensor_4: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_1, sum_dim_int_list_7);  add_tensor_1 = sum_dim_int_list_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_8: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_341, [0], True);  view_341 = None
        reshape_default_2: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_8, [4096]);  sum_dim_int_list_8 = None
        add_tensor_5: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_2, reshape_default_2);  add_tensor_2 = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_tensor_3: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_179, mul_84);  mul_84 = None
        sum_dim_int_list_9: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1]);  mul_tensor_3 = None
        sum_dim_int_list_10: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_179, [0, 1]);  add_179 = None
        add_tensor_6: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_3, sum_dim_int_list_9);  add_tensor_3 = sum_dim_int_list_9 = None
        add_tensor_7: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_4, sum_dim_int_list_10);  add_tensor_4 = sum_dim_int_list_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_11: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_369, [0], True);  view_369 = None
        reshape_default_3: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, [4096]);  sum_dim_int_list_11 = None
        add_tensor_8: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_5, reshape_default_3);  add_tensor_5 = reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_tensor_4: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_201, mul_74);  mul_74 = None
        sum_dim_int_list_12: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1]);  mul_tensor_4 = None
        sum_dim_int_list_13: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_201, [0, 1]);  add_201 = None
        add_tensor_9: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_6, sum_dim_int_list_12);  add_tensor_6 = sum_dim_int_list_12 = None
        add_tensor_10: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_7, sum_dim_int_list_13);  add_tensor_7 = sum_dim_int_list_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_14: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_397, [0], True);  view_397 = None
        reshape_default_4: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_14, [4096]);  sum_dim_int_list_14 = None
        add_tensor_11: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_8, reshape_default_4);  add_tensor_8 = reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_tensor_5: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_223, mul_64);  mul_64 = None
        sum_dim_int_list_15: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_16: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_223, [0, 1]);  add_223 = None
        add_tensor_12: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_9, sum_dim_int_list_15);  add_tensor_9 = sum_dim_int_list_15 = None
        add_tensor_13: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_10, sum_dim_int_list_16);  add_tensor_10 = sum_dim_int_list_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_17: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_425, [0], True);  view_425 = None
        reshape_default_5: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_17, [4096]);  sum_dim_int_list_17 = None
        add_tensor_14: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_11, reshape_default_5);  add_tensor_11 = reshape_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_tensor_6: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_245, mul_54);  mul_54 = None
        sum_dim_int_list_18: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_19: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_245, [0, 1]);  add_245 = None
        add_tensor_15: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_12, sum_dim_int_list_18);  add_tensor_12 = sum_dim_int_list_18 = None
        add_tensor_16: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_13, sum_dim_int_list_19);  add_tensor_13 = sum_dim_int_list_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_20: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_453, [0], True);  view_453 = None
        reshape_default_6: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_20, [4096]);  sum_dim_int_list_20 = None
        add_tensor_17: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_14, reshape_default_6);  add_tensor_14 = reshape_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_tensor_7: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_267, mul_44);  mul_44 = None
        sum_dim_int_list_21: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_22: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_267, [0, 1]);  add_267 = None
        add_tensor_18: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_15, sum_dim_int_list_21);  add_tensor_15 = sum_dim_int_list_21 = None
        add_tensor_19: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_16, sum_dim_int_list_22);  add_tensor_16 = sum_dim_int_list_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_23: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_481, [0], True);  view_481 = None
        reshape_default_7: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, [4096]);  sum_dim_int_list_23 = None
        add_tensor_20: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_17, reshape_default_7);  add_tensor_17 = reshape_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_tensor_8: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_289, mul_34);  mul_34 = None
        sum_dim_int_list_24: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_25: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_289, [0, 1]);  add_289 = None
        add_tensor_21: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_18, sum_dim_int_list_24);  add_tensor_18 = sum_dim_int_list_24 = None
        add_tensor_22: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_19, sum_dim_int_list_25);  add_tensor_19 = sum_dim_int_list_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_26: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_509, [0], True);  view_509 = None
        reshape_default_8: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_26, [4096]);  sum_dim_int_list_26 = None
        add_tensor_23: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_20, reshape_default_8);  add_tensor_20 = reshape_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_tensor_9: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_311, mul_24);  mul_24 = None
        sum_dim_int_list_27: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1]);  mul_tensor_9 = None
        sum_dim_int_list_28: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_311, [0, 1]);  add_311 = None
        add_tensor_24: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_21, sum_dim_int_list_27);  add_tensor_21 = sum_dim_int_list_27 = None
        add_tensor_25: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_22, sum_dim_int_list_28);  add_tensor_22 = sum_dim_int_list_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_29: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_537, [0], True);  view_537 = None
        reshape_default_9: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_29, [4096]);  sum_dim_int_list_29 = None
        add_tensor_26: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_23, reshape_default_9);  add_tensor_23 = reshape_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_tensor_10: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_333, mul_14);  mul_14 = None
        sum_dim_int_list_30: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        sum_dim_int_list_31: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_333, [0, 1]);  add_333 = None
        add_tensor_27: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_24, sum_dim_int_list_30);  add_tensor_24 = sum_dim_int_list_30 = None
        add_tensor_28: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_25, sum_dim_int_list_31);  add_tensor_25 = sum_dim_int_list_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        sum_dim_int_list_32: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_565, [0], True);  view_565 = None
        reshape_default_10: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_32, [4096]);  sum_dim_int_list_32 = None
        add_tensor_29: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_26, reshape_default_10);  add_tensor_26 = reshape_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:239 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        reshape_default_11: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_138, [8, 512, 4096]);  mm_138 = None
        add_tensor_30: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_437, reshape_default_11);  mul_437 = reshape_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        mul_tensor_11: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_tensor_30, primals_19);  primals_19 = None
        mul_tensor_12: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor_11, 4096)
        sum_dim_int_list_33: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [2], True)
        mul_tensor_13: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor_11, mul_4);  mul_tensor_11 = None
        sum_dim_int_list_34: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [2], True);  mul_tensor_13 = None
        mul_tensor_14: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_4, sum_dim_int_list_34);  sum_dim_int_list_34 = None
        sub_tensor: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_tensor_12, sum_dim_int_list_33);  mul_tensor_12 = sum_dim_int_list_33 = None
        sub_tensor_1: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_14);  sub_tensor = mul_tensor_14 = None
        mul_tensor_15: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(div_38, sub_tensor_1);  div_38 = sub_tensor_1 = None
        mul_tensor_16: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_tensor_30, mul_4);  mul_4 = None
        sum_dim_int_list_35: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1]);  mul_tensor_16 = None
        sum_dim_int_list_36: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_tensor_30, [0, 1]);  add_tensor_30 = None
        add_tensor_31: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_27, sum_dim_int_list_35);  add_tensor_27 = sum_dim_int_list_35 = None
        add_tensor_32: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_28, sum_dim_int_list_36);  add_tensor_28 = sum_dim_int_list_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:200 in forward, code: attn_output = self.dense(attn_output)
        reshape_default_12: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_tensor_15, [4096, 4096]);  mul_tensor_15 = None
        permute_default: "f32[4096, 4096]" = torch.ops.aten.permute.default(reshape_default_12, [1, 0])
        sum_dim_int_list_37: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(reshape_default_12, [0], True);  reshape_default_12 = None
        reshape_default_13: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_37, [4096]);  sum_dim_int_list_37 = None
        add_tensor_33: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_29, reshape_default_13);  add_tensor_29 = reshape_default_13 = None
        return (add_tensor_31, add_tensor_32, permute_default, add_tensor_33)


def _default_make_inputs():
    return [
    torch.randn([8, 512, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
