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
    def forward(self, view_278: "f32[8, 512, 4096]", mul_120: "f32[8, 512, 4096]", view_279: "f32[4096, 4096]", add_126: "f32[8, 512, 4096]", mul_110: "f32[8, 512, 4096]", view_307: "f32[4096, 4096]", add_148: "f32[8, 512, 4096]", mul_100: "f32[8, 512, 4096]", view_335: "f32[4096, 4096]", add_170: "f32[8, 512, 4096]", mul_90: "f32[8, 512, 4096]", view_363: "f32[4096, 4096]", add_192: "f32[8, 512, 4096]", mul_80: "f32[8, 512, 4096]", view_391: "f32[4096, 4096]", add_214: "f32[8, 512, 4096]", mul_70: "f32[8, 512, 4096]", view_419: "f32[4096, 4096]", add_236: "f32[8, 512, 4096]", mul_60: "f32[8, 512, 4096]", view_447: "f32[4096, 4096]", add_258: "f32[8, 512, 4096]", mul_50: "f32[8, 512, 4096]", view_475: "f32[4096, 4096]", add_280: "f32[8, 512, 4096]", mul_40: "f32[8, 512, 4096]", view_503: "f32[4096, 4096]", add_302: "f32[8, 512, 4096]", mul_30: "f32[8, 512, 4096]", view_531: "f32[4096, 4096]", add_324: "f32[8, 512, 4096]", mul_20: "f32[8, 512, 4096]", view_559: "f32[4096, 4096]", mm_130: "f32[4096, 4096]", mul_427: "f32[8, 512, 4096]", mm_132: "f32[4096, 4096]", mm_134: "f32[4096, 4096]", primals_25: "f32[4096]", mul_10: "f32[8, 512, 4096]", div_37: "f32[8, 512, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_tensor: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(view_278, mul_120);  mul_120 = None
        sum_dim_int_list: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_1: "f32[4096]" = torch.ops.aten.sum.dim_IntList(view_278, [0, 1]);  view_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        sum_dim_int_list_2: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_279, [0], True);  view_279 = None
        reshape_default: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, _shape_param_0);  sum_dim_int_list_2 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_tensor_1: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_126, mul_110);  mul_110 = None
        sum_dim_int_list_3: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_4: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_126, [0, 1]);  add_126 = None
        add_tensor: "f32[4096]" = torch.ops.aten.add.Tensor(sum_dim_int_list, sum_dim_int_list_3);  sum_dim_int_list = sum_dim_int_list_3 = None
        add_tensor_1: "f32[4096]" = torch.ops.aten.add.Tensor(sum_dim_int_list_1, sum_dim_int_list_4);  sum_dim_int_list_1 = sum_dim_int_list_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        sum_dim_int_list_5: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_307, [0], True);  view_307 = None
        reshape_default_1: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_5, _shape_param_1);  sum_dim_int_list_5 = _shape_param_1 = None
        add_tensor_2: "f32[4096]" = torch.ops.aten.add.Tensor(reshape_default, reshape_default_1);  reshape_default = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_tensor_2: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_148, mul_100);  mul_100 = None
        sum_dim_int_list_6: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1]);  mul_tensor_2 = None
        sum_dim_int_list_7: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_148, [0, 1]);  add_148 = None
        add_tensor_3: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor, sum_dim_int_list_6);  add_tensor = sum_dim_int_list_6 = None
        add_tensor_4: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_1, sum_dim_int_list_7);  add_tensor_1 = sum_dim_int_list_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        sum_dim_int_list_8: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_335, [0], True);  view_335 = None
        reshape_default_2: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_8, _shape_param_2);  sum_dim_int_list_8 = _shape_param_2 = None
        add_tensor_5: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_2, reshape_default_2);  add_tensor_2 = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_tensor_3: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_170, mul_90);  mul_90 = None
        sum_dim_int_list_9: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1]);  mul_tensor_3 = None
        sum_dim_int_list_10: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_170, [0, 1]);  add_170 = None
        add_tensor_6: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_3, sum_dim_int_list_9);  add_tensor_3 = sum_dim_int_list_9 = None
        add_tensor_7: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_4, sum_dim_int_list_10);  add_tensor_4 = sum_dim_int_list_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        sum_dim_int_list_11: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_363, [0], True);  view_363 = None
        reshape_default_3: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, _shape_param_3);  sum_dim_int_list_11 = _shape_param_3 = None
        add_tensor_8: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_5, reshape_default_3);  add_tensor_5 = reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_tensor_4: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_192, mul_80);  mul_80 = None
        sum_dim_int_list_12: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1]);  mul_tensor_4 = None
        sum_dim_int_list_13: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_192, [0, 1]);  add_192 = None
        add_tensor_9: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_6, sum_dim_int_list_12);  add_tensor_6 = sum_dim_int_list_12 = None
        add_tensor_10: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_7, sum_dim_int_list_13);  add_tensor_7 = sum_dim_int_list_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        sum_dim_int_list_14: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_391, [0], True);  view_391 = None
        reshape_default_4: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_14, _shape_param_4);  sum_dim_int_list_14 = _shape_param_4 = None
        add_tensor_11: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_8, reshape_default_4);  add_tensor_8 = reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_tensor_5: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_214, mul_70);  mul_70 = None
        sum_dim_int_list_15: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_16: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_214, [0, 1]);  add_214 = None
        add_tensor_12: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_9, sum_dim_int_list_15);  add_tensor_9 = sum_dim_int_list_15 = None
        add_tensor_13: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_10, sum_dim_int_list_16);  add_tensor_10 = sum_dim_int_list_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        sum_dim_int_list_17: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_419, [0], True);  view_419 = None
        reshape_default_5: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_17, _shape_param_5);  sum_dim_int_list_17 = _shape_param_5 = None
        add_tensor_14: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_11, reshape_default_5);  add_tensor_11 = reshape_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_tensor_6: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_236, mul_60);  mul_60 = None
        sum_dim_int_list_18: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_19: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_236, [0, 1]);  add_236 = None
        add_tensor_15: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_12, sum_dim_int_list_18);  add_tensor_12 = sum_dim_int_list_18 = None
        add_tensor_16: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_13, sum_dim_int_list_19);  add_tensor_13 = sum_dim_int_list_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        sum_dim_int_list_20: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_447, [0], True);  view_447 = None
        reshape_default_6: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_20, _shape_param_6);  sum_dim_int_list_20 = _shape_param_6 = None
        add_tensor_17: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_14, reshape_default_6);  add_tensor_14 = reshape_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_tensor_7: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_258, mul_50);  mul_50 = None
        sum_dim_int_list_21: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_22: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_258, [0, 1]);  add_258 = None
        add_tensor_18: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_15, sum_dim_int_list_21);  add_tensor_15 = sum_dim_int_list_21 = None
        add_tensor_19: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_16, sum_dim_int_list_22);  add_tensor_16 = sum_dim_int_list_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        sum_dim_int_list_23: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_475, [0], True);  view_475 = None
        reshape_default_7: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, _shape_param_7);  sum_dim_int_list_23 = _shape_param_7 = None
        add_tensor_20: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_17, reshape_default_7);  add_tensor_17 = reshape_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_tensor_8: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_280, mul_40);  mul_40 = None
        sum_dim_int_list_24: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_25: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_280, [0, 1]);  add_280 = None
        add_tensor_21: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_18, sum_dim_int_list_24);  add_tensor_18 = sum_dim_int_list_24 = None
        add_tensor_22: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_19, sum_dim_int_list_25);  add_tensor_19 = sum_dim_int_list_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        sum_dim_int_list_26: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_503, [0], True);  view_503 = None
        reshape_default_8: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_26, _shape_param_8);  sum_dim_int_list_26 = _shape_param_8 = None
        add_tensor_23: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_20, reshape_default_8);  add_tensor_20 = reshape_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_tensor_9: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_302, mul_30);  mul_30 = None
        sum_dim_int_list_27: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1]);  mul_tensor_9 = None
        sum_dim_int_list_28: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_302, [0, 1]);  add_302 = None
        add_tensor_24: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_21, sum_dim_int_list_27);  add_tensor_21 = sum_dim_int_list_27 = None
        add_tensor_25: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_22, sum_dim_int_list_28);  add_tensor_22 = sum_dim_int_list_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        sum_dim_int_list_29: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_531, [0], True);  view_531 = None
        reshape_default_9: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_29, _shape_param_9);  sum_dim_int_list_29 = _shape_param_9 = None
        add_tensor_26: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_23, reshape_default_9);  add_tensor_23 = reshape_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_tensor_10: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_324, mul_20);  mul_20 = None
        sum_dim_int_list_30: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        sum_dim_int_list_31: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_324, [0, 1]);  add_324 = None
        add_tensor_27: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_24, sum_dim_int_list_30);  add_tensor_24 = sum_dim_int_list_30 = None
        add_tensor_28: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_25, sum_dim_int_list_31);  add_tensor_25 = sum_dim_int_list_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        sum_dim_int_list_32: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_559, [0], True);  view_559 = None
        reshape_default_10: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_32, _shape_param_10);  sum_dim_int_list_32 = _shape_param_10 = None
        add_tensor_29: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_26, reshape_default_10);  add_tensor_26 = reshape_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_11: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_130, _shape_param_11);  mm_130 = _shape_param_11 = None
        add_tensor_30: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_427, reshape_default_11);  mul_427 = reshape_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_12: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_132, _shape_param_12);  mm_132 = _shape_param_12 = None
        add_tensor_31: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_tensor_30, reshape_default_12);  add_tensor_30 = reshape_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_13: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(mm_134, _shape_param_13);  mm_134 = _shape_param_13 = None
        add_tensor_32: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_tensor_31, reshape_default_13);  add_tensor_31 = reshape_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        mul_tensor_11: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_tensor_32, primals_25);  primals_25 = None
        mul_tensor_12: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor_11, 4096)
        sum_dim_int_list_33: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [2], True)
        mul_tensor_13: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor_11, mul_10);  mul_tensor_11 = None
        sum_dim_int_list_34: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [2], True);  mul_tensor_13 = None
        mul_tensor_14: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_10, sum_dim_int_list_34);  sum_dim_int_list_34 = None
        sub_tensor: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_tensor_12, sum_dim_int_list_33);  mul_tensor_12 = sum_dim_int_list_33 = None
        sub_tensor_1: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_14);  sub_tensor = mul_tensor_14 = None
        mul_tensor_15: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(div_37, sub_tensor_1);  div_37 = sub_tensor_1 = None
        mul_tensor_16: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_tensor_32, mul_10);  mul_10 = None
        sum_dim_int_list_35: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1]);  mul_tensor_16 = None
        sum_dim_int_list_36: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_tensor_32, [0, 1]);  add_tensor_32 = None
        add_tensor_33: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_27, sum_dim_int_list_35);  add_tensor_27 = sum_dim_int_list_35 = None
        add_tensor_34: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_28, sum_dim_int_list_36);  add_tensor_28 = sum_dim_int_list_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:241 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        reshape_default_14: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_tensor_15, _shape_param_14);  mul_tensor_15 = _shape_param_14 = None
        permute_default: "f32[4096, 4096]" = torch.ops.aten.permute.default(reshape_default_14, [1, 0])
        sum_dim_int_list_37: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(reshape_default_14, [0], True);  reshape_default_14 = None
        reshape_default_15: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_37, _shape_param_15);  sum_dim_int_list_37 = _shape_param_15 = None
        add_tensor_35: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_29, reshape_default_15);  add_tensor_29 = reshape_default_15 = None
        return (add_tensor_33, add_tensor_34, permute_default, add_tensor_35)


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
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    [4096],  # _shape_param_0
    [4096],  # _shape_param_1
    [4096],  # _shape_param_2
    [4096],  # _shape_param_3
    [4096],  # _shape_param_4
    [4096],  # _shape_param_5
    [4096],  # _shape_param_6
    [4096],  # _shape_param_7
    [4096],  # _shape_param_8
    [4096],  # _shape_param_9
    [4096],  # _shape_param_10
    [8, 512, 4096],  # _shape_param_11
    [8, 512, 4096],  # _shape_param_12
    [8, 512, 4096],  # _shape_param_13
    [4096, 4096],  # _shape_param_14
    [4096],  # _shape_param_15
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
