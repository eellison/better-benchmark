"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_train
Pattern hash: 37d94555aad7
Shape hash: 485c0134
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), S([38809, 12]), S([38809, 12]), S([38809, 12]), S([38809, 12]), S([38809, 12]), S([38809, 12]), S([38809, 12]), S([38809, 12]), S([38809, 12]), S([38809, 12]), S([38809, 12]), S([38809, 12]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_137: "f32[128, 12, 197, 197]", primals_211: "i64[197, 197]", getitem_141: "f32[128, 12, 197, 197]", primals_193: "i64[197, 197]", getitem_145: "f32[128, 12, 197, 197]", primals_175: "i64[197, 197]", getitem_149: "f32[128, 12, 197, 197]", primals_157: "i64[197, 197]", getitem_153: "f32[128, 12, 197, 197]", primals_139: "i64[197, 197]", getitem_157: "f32[128, 12, 197, 197]", primals_121: "i64[197, 197]", getitem_161: "f32[128, 12, 197, 197]", primals_103: "i64[197, 197]", getitem_165: "f32[128, 12, 197, 197]", primals_85: "i64[197, 197]", getitem_169: "f32[128, 12, 197, 197]", primals_67: "i64[197, 197]", getitem_173: "f32[128, 12, 197, 197]", primals_49: "i64[197, 197]", getitem_177: "f32[128, 12, 197, 197]", primals_31: "i64[197, 197]", getitem_181: "f32[128, 12, 197, 197]", primals_13: "i64[197, 197]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        sum_dim_int_list: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_137, [0], True);  getitem_137 = None
        full_default: "f32[1, 12, 197, 200]" = torch.ops.aten.full.default([1, 12, 197, 200], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list, -1, 0, 197);  sum_dim_int_list = None
        constant_pad_nd_default: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default, [0, -3]);  slice_scatter_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default, 0);  constant_pad_nd_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim, [1, 2, 0]);  squeeze_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        reshape_default: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_default, _shape_param_0);  permute_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        full_default_1: "f32[732, 12]" = torch.ops.aten.full.default([732, 12], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        reshape_default_1: "i64[38809]" = torch.ops.aten.reshape.default(primals_211, [-1]);  primals_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_1], reshape_default, True);  reshape_default_1 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        sum_dim_int_list_1: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_141, [0], True);  getitem_141 = None
        slice_scatter_default_1: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list_1, -1, 0, 197);  sum_dim_int_list_1 = None
        constant_pad_nd_default_1: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default_1, [0, -3]);  slice_scatter_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_1: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default_1, 0);  constant_pad_nd_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_1: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim_1, [1, 2, 0]);  squeeze_dim_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        reshape_default_2: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_default_1, _shape_param_1);  permute_default_1 = _shape_param_1 = None
        reshape_default_3: "i64[38809]" = torch.ops.aten.reshape.default(primals_193, [-1]);  primals_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_1: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_3], reshape_default_2, True);  reshape_default_3 = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        sum_dim_int_list_2: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_145, [0], True);  getitem_145 = None
        slice_scatter_default_2: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list_2, -1, 0, 197);  sum_dim_int_list_2 = None
        constant_pad_nd_default_2: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default_2, [0, -3]);  slice_scatter_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_2: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default_2, 0);  constant_pad_nd_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_2: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim_2, [1, 2, 0]);  squeeze_dim_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        reshape_default_4: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_default_2, _shape_param_2);  permute_default_2 = _shape_param_2 = None
        reshape_default_5: "i64[38809]" = torch.ops.aten.reshape.default(primals_175, [-1]);  primals_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_2: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_5], reshape_default_4, True);  reshape_default_5 = reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        sum_dim_int_list_3: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_149, [0], True);  getitem_149 = None
        slice_scatter_default_3: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list_3, -1, 0, 197);  sum_dim_int_list_3 = None
        constant_pad_nd_default_3: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default_3, [0, -3]);  slice_scatter_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_3: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default_3, 0);  constant_pad_nd_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_3: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim_3, [1, 2, 0]);  squeeze_dim_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        reshape_default_6: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_default_3, _shape_param_3);  permute_default_3 = _shape_param_3 = None
        reshape_default_7: "i64[38809]" = torch.ops.aten.reshape.default(primals_157, [-1]);  primals_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_3: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_7], reshape_default_6, True);  reshape_default_7 = reshape_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        sum_dim_int_list_4: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_153, [0], True);  getitem_153 = None
        slice_scatter_default_4: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list_4, -1, 0, 197);  sum_dim_int_list_4 = None
        constant_pad_nd_default_4: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default_4, [0, -3]);  slice_scatter_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_4: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default_4, 0);  constant_pad_nd_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_4: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim_4, [1, 2, 0]);  squeeze_dim_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        reshape_default_8: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_default_4, _shape_param_4);  permute_default_4 = _shape_param_4 = None
        reshape_default_9: "i64[38809]" = torch.ops.aten.reshape.default(primals_139, [-1]);  primals_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_4: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_9], reshape_default_8, True);  reshape_default_9 = reshape_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        sum_dim_int_list_5: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_157, [0], True);  getitem_157 = None
        slice_scatter_default_5: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list_5, -1, 0, 197);  sum_dim_int_list_5 = None
        constant_pad_nd_default_5: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default_5, [0, -3]);  slice_scatter_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_5: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default_5, 0);  constant_pad_nd_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_5: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim_5, [1, 2, 0]);  squeeze_dim_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        reshape_default_10: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_default_5, _shape_param_5);  permute_default_5 = _shape_param_5 = None
        reshape_default_11: "i64[38809]" = torch.ops.aten.reshape.default(primals_121, [-1]);  primals_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_5: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_11], reshape_default_10, True);  reshape_default_11 = reshape_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        sum_dim_int_list_6: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_161, [0], True);  getitem_161 = None
        slice_scatter_default_6: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list_6, -1, 0, 197);  sum_dim_int_list_6 = None
        constant_pad_nd_default_6: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default_6, [0, -3]);  slice_scatter_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_6: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default_6, 0);  constant_pad_nd_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_6: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim_6, [1, 2, 0]);  squeeze_dim_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        reshape_default_12: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_default_6, _shape_param_6);  permute_default_6 = _shape_param_6 = None
        reshape_default_13: "i64[38809]" = torch.ops.aten.reshape.default(primals_103, [-1]);  primals_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_6: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_13], reshape_default_12, True);  reshape_default_13 = reshape_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        sum_dim_int_list_7: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_165, [0], True);  getitem_165 = None
        slice_scatter_default_7: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list_7, -1, 0, 197);  sum_dim_int_list_7 = None
        constant_pad_nd_default_7: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default_7, [0, -3]);  slice_scatter_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_7: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default_7, 0);  constant_pad_nd_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_7: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim_7, [1, 2, 0]);  squeeze_dim_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        reshape_default_14: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_default_7, _shape_param_7);  permute_default_7 = _shape_param_7 = None
        reshape_default_15: "i64[38809]" = torch.ops.aten.reshape.default(primals_85, [-1]);  primals_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_7: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_15], reshape_default_14, True);  reshape_default_15 = reshape_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        sum_dim_int_list_8: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_169, [0], True);  getitem_169 = None
        slice_scatter_default_8: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list_8, -1, 0, 197);  sum_dim_int_list_8 = None
        constant_pad_nd_default_8: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default_8, [0, -3]);  slice_scatter_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_8: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default_8, 0);  constant_pad_nd_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_8: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim_8, [1, 2, 0]);  squeeze_dim_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        reshape_default_16: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_default_8, _shape_param_8);  permute_default_8 = _shape_param_8 = None
        reshape_default_17: "i64[38809]" = torch.ops.aten.reshape.default(primals_67, [-1]);  primals_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_8: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_17], reshape_default_16, True);  reshape_default_17 = reshape_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        sum_dim_int_list_9: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_173, [0], True);  getitem_173 = None
        slice_scatter_default_9: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list_9, -1, 0, 197);  sum_dim_int_list_9 = None
        constant_pad_nd_default_9: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default_9, [0, -3]);  slice_scatter_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_9: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default_9, 0);  constant_pad_nd_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_9: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim_9, [1, 2, 0]);  squeeze_dim_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        reshape_default_18: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_default_9, _shape_param_9);  permute_default_9 = _shape_param_9 = None
        reshape_default_19: "i64[38809]" = torch.ops.aten.reshape.default(primals_49, [-1]);  primals_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_9: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_19], reshape_default_18, True);  reshape_default_19 = reshape_default_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        sum_dim_int_list_10: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_177, [0], True);  getitem_177 = None
        slice_scatter_default_10: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list_10, -1, 0, 197);  sum_dim_int_list_10 = None
        constant_pad_nd_default_10: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default_10, [0, -3]);  slice_scatter_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_10: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default_10, 0);  constant_pad_nd_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_10: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim_10, [1, 2, 0]);  squeeze_dim_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        reshape_default_20: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_default_10, _shape_param_10);  permute_default_10 = _shape_param_10 = None
        reshape_default_21: "i64[38809]" = torch.ops.aten.reshape.default(primals_31, [-1]);  primals_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_10: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_21], reshape_default_20, True);  reshape_default_21 = reshape_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        sum_dim_int_list_11: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_181, [0], True);  getitem_181 = None
        slice_scatter_default_11: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list_11, -1, 0, 197);  full_default = sum_dim_int_list_11 = None
        constant_pad_nd_default_11: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default_11, [0, -3]);  slice_scatter_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_11: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default_11, 0);  constant_pad_nd_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_11: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim_11, [1, 2, 0]);  squeeze_dim_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        reshape_default_22: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_default_11, _shape_param_11);  permute_default_11 = _shape_param_11 = None
        reshape_default_23: "i64[38809]" = torch.ops.aten.reshape.default(primals_13, [-1]);  primals_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_11: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_23], reshape_default_22, True);  full_default_1 = reshape_default_23 = reshape_default_22 = None
        return (index_put_default, index_put_default_10, index_put_default_8, index_put_default_11, index_put_default_1, index_put_default_2, index_put_default_3, index_put_default_4, index_put_default_5, index_put_default_7, index_put_default_6, index_put_default_9)



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
