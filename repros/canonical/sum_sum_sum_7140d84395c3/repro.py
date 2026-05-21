"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['1', '1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['4', '512', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import glob
import os
import torch
from math import inf
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([2048, 32768], bf16), T([4, 512, 4096], f32), T([4, 512, 1], f32), T([4, 512, 4096], bf16), T([2048, 4096], bf16), T([2048, 14336], bf16), T([2048, 14336], bf16), T([4, 512, 4096], f32), T([4, 512, 1], f32), T([4, 512, 4096], bf16), T([2048, 4096], bf16), T([4, 32, 512, 128], bf16, stride=(2097152, 128, 4096, 1)), T([2048, 1024], bf16), T([2048, 1024], bf16), T([2048, 4096], bf16), T([4, 512, 4096], f32), T([4, 512, 1], f32), T([4, 512, 4096], bf16), T([2048, 4096], bf16), T([2048, 14336], bf16), T([2048, 14336], bf16), T([4, 512, 4096], f32), T([4, 512, 1], f32), T([4, 512, 4096], bf16), T([2048, 4096], bf16), T([4, 32, 512, 128], bf16, stride=(2097152, 128, 4096, 1)), T([2048, 1024], bf16), T([2048, 1024], bf16), T([2048, 4096], bf16), T([4, 512, 4096], f32), T([4, 512, 1], f32), T([4, 512, 4096], bf16), T([2048, 4096], bf16), T([2048, 14336], bf16), T([2048, 14336], bf16), T([4, 512, 4096], f32), T([4, 512, 1], f32), T([4, 512, 4096], bf16), T([2048, 4096], bf16), T([4, 32, 512, 128], bf16, stride=(2097152, 128, 4096, 1)), T([2048, 1024], bf16), T([2048, 1024], bf16), T([2048, 4096], bf16), T([4, 512, 4096], f32), T([4, 512, 1], f32), T([4, 512, 4096], bf16), T([2048, 4096], bf16), T([2048, 14336], bf16), T([2048, 14336], bf16), T([4, 512, 4096], f32), T([4, 512, 1], f32), T([4, 512, 4096], bf16), T([2048, 4096], bf16), T([4, 32, 512, 128], bf16, stride=(2097152, 128, 4096, 1)), T([2048, 1024], bf16), T([2048, 1024], bf16), T([2048, 4096], bf16), T([4, 512, 4096], f32), T([4, 512, 1], f32), T([4, 512, 4096], bf16), T([2048, 4096], bf16), T([2048, 14336], bf16), T([2048, 14336], bf16), T([4, 512, 4096], f32), T([4, 512, 1], f32), T([4, 512, 4096], bf16), T([2048, 4096], bf16), T([4, 32, 512, 128], bf16, stride=(2097152, 128, 4096, 1)), T([2048, 1024], bf16), T([2048, 1024], bf16), T([2048, 4096], bf16), T([4, 512, 4096], f32), T([4, 512, 1], f32), T([4, 512, 4096], bf16), T([2048, 4096], bf16), T([2048, 14336], bf16), T([2048, 14336], bf16), T([4, 512, 4096], f32), T([4, 512, 1], f32), T([4, 512, 4096], bf16), T([2048, 4096], bf16), T([4, 32, 512, 128], bf16, stride=(2097152, 128, 4096, 1)), T([2048, 1024], bf16), T([2048, 1024], bf16), T([2048, 4096], bf16), T([4, 512, 4096], f32), T([4, 512, 1], f32), T([4, 512, 4096], bf16), T([2048, 4096], bf16), T([2048, 14336], bf16), T([2048, 14336], bf16), T([4, 512, 4096], f32), T([4, 512, 1], f32), T([4, 512, 4096], bf16), T([2048, 4096], bf16), T([4, 32, 512, 128], bf16, stride=(2097152, 128, 4096, 1)), T([2048, 1024], bf16), T([2048, 1024], bf16), T([2048, 4096], bf16), T([4, 512, 4096], f32), T([4, 512, 1], f32), T([4, 512, 4096], bf16), T([2048, 4096], bf16), T([2048, 14336], bf16), T([2048, 14336], bf16), T([4, 512, 4096], f32), T([4, 512, 1], f32), T([4, 512, 4096], bf16), T([2048, 4096], bf16), T([4, 32, 512, 128], bf16, stride=(2097152, 128, 4096, 1)), T([2048, 1024], bf16), T([2048, 4096], bf16), T([2048, 1024], bf16), T([2048, 4096], bf16), T([2048, 4096], bf16), T([2048, 4096], bf16), T([4096], bf16), T([4, 512, 4096], bf16), T([4, 512, 1], f32), T([4, 512, 4096], bf16), T([4, 512], i64, gen=Index(100)), T([], f32))"

class Repro(torch.nn.Module):
    def forward(self, view_169: "bf16[2048, 32768]", convert_element_type_163: "f32[4, 512, 4096]", rsqrt_16: "f32[4, 512, 1]", view_170: "bf16[4, 512, 4096]", view_172: "bf16[2048, 4096]", view_174: "bf16[2048, 14336]", view_176: "bf16[2048, 14336]", convert_element_type_153: "f32[4, 512, 4096]", rsqrt_15: "f32[4, 512, 1]", add_64: "bf16[4, 512, 4096]", view_179: "bf16[2048, 4096]", getitem_63: "bf16[4, 32, 512, 128]", view_185: "bf16[2048, 1024]", view_188: "bf16[2048, 1024]", view_191: "bf16[2048, 4096]", convert_element_type_143: "f32[4, 512, 4096]", rsqrt_14: "f32[4, 512, 1]", add_72: "bf16[4, 512, 4096]", view_194: "bf16[2048, 4096]", view_196: "bf16[2048, 14336]", view_198: "bf16[2048, 14336]", convert_element_type_133: "f32[4, 512, 4096]", rsqrt_13: "f32[4, 512, 1]", add_77: "bf16[4, 512, 4096]", view_201: "bf16[2048, 4096]", getitem_54: "bf16[4, 32, 512, 128]", view_207: "bf16[2048, 1024]", view_210: "bf16[2048, 1024]", view_213: "bf16[2048, 4096]", convert_element_type_123: "f32[4, 512, 4096]", rsqrt_12: "f32[4, 512, 1]", add_85: "bf16[4, 512, 4096]", view_216: "bf16[2048, 4096]", view_218: "bf16[2048, 14336]", view_220: "bf16[2048, 14336]", convert_element_type_113: "f32[4, 512, 4096]", rsqrt_11: "f32[4, 512, 1]", add_90: "bf16[4, 512, 4096]", view_223: "bf16[2048, 4096]", getitem_45: "bf16[4, 32, 512, 128]", view_229: "bf16[2048, 1024]", view_232: "bf16[2048, 1024]", view_235: "bf16[2048, 4096]", convert_element_type_103: "f32[4, 512, 4096]", rsqrt_10: "f32[4, 512, 1]", add_98: "bf16[4, 512, 4096]", view_238: "bf16[2048, 4096]", view_240: "bf16[2048, 14336]", view_242: "bf16[2048, 14336]", convert_element_type_93: "f32[4, 512, 4096]", rsqrt_9: "f32[4, 512, 1]", add_103: "bf16[4, 512, 4096]", view_245: "bf16[2048, 4096]", getitem_36: "bf16[4, 32, 512, 128]", view_251: "bf16[2048, 1024]", view_254: "bf16[2048, 1024]", view_257: "bf16[2048, 4096]", convert_element_type_83: "f32[4, 512, 4096]", rsqrt_8: "f32[4, 512, 1]", add_111: "bf16[4, 512, 4096]", view_260: "bf16[2048, 4096]", view_262: "bf16[2048, 14336]", view_264: "bf16[2048, 14336]", convert_element_type_73: "f32[4, 512, 4096]", rsqrt_7: "f32[4, 512, 1]", add_116: "bf16[4, 512, 4096]", view_267: "bf16[2048, 4096]", getitem_27: "bf16[4, 32, 512, 128]", view_273: "bf16[2048, 1024]", view_276: "bf16[2048, 1024]", view_279: "bf16[2048, 4096]", convert_element_type_63: "f32[4, 512, 4096]", rsqrt_6: "f32[4, 512, 1]", add_124: "bf16[4, 512, 4096]", view_282: "bf16[2048, 4096]", view_284: "bf16[2048, 14336]", view_286: "bf16[2048, 14336]", convert_element_type_53: "f32[4, 512, 4096]", rsqrt_5: "f32[4, 512, 1]", add_129: "bf16[4, 512, 4096]", view_289: "bf16[2048, 4096]", getitem_18: "bf16[4, 32, 512, 128]", view_295: "bf16[2048, 1024]", view_298: "bf16[2048, 1024]", view_301: "bf16[2048, 4096]", convert_element_type_43: "f32[4, 512, 4096]", rsqrt_4: "f32[4, 512, 1]", add_137: "bf16[4, 512, 4096]", view_304: "bf16[2048, 4096]", view_306: "bf16[2048, 14336]", view_308: "bf16[2048, 14336]", convert_element_type_33: "f32[4, 512, 4096]", rsqrt_3: "f32[4, 512, 1]", add_142: "bf16[4, 512, 4096]", view_311: "bf16[2048, 4096]", getitem_9: "bf16[4, 32, 512, 128]", view_317: "bf16[2048, 1024]", view_320: "bf16[2048, 1024]", view_323: "bf16[2048, 4096]", convert_element_type_23: "f32[4, 512, 4096]", rsqrt_2: "f32[4, 512, 1]", add_150: "bf16[4, 512, 4096]", view_326: "bf16[2048, 4096]", view_328: "bf16[2048, 14336]", view_330: "bf16[2048, 14336]", convert_element_type_13: "f32[4, 512, 4096]", rsqrt_1: "f32[4, 512, 1]", add_155: "bf16[4, 512, 4096]", view_333: "bf16[2048, 4096]", getitem: "bf16[4, 32, 512, 128]", view_339: "bf16[2048, 1024]", mm_166: "bf16[2048, 4096]", view_342: "bf16[2048, 1024]", mm_168: "bf16[2048, 4096]", view_345: "bf16[2048, 4096]", mm_170: "bf16[2048, 4096]", primals_4: "bf16[4096]", embedding: "bf16[4, 512, 4096]", rsqrt: "f32[4, 512, 1]", add_157: "bf16[4, 512, 4096]", primals_1: "i64[4, 512]", full_default_18: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:460 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        permute_default: "bf16[32768, 2048]" = torch.ops.aten.permute.default(view_169, [1, 0]);  view_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_163, rsqrt_16);  convert_element_type_163 = rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        mul_tensor_1: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_170, convert_element_type_default);  view_170 = convert_element_type_default = None
        sum_dim_int_list: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1], True);  mul_tensor_1 = None
        reshape_default: "bf16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list, [4096]);  sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_1: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_172, [1, 0]);  view_172 = None
        permute_default_2: "bf16[14336, 2048]" = torch.ops.aten.permute.default(view_174, [1, 0]);  view_174 = None
        permute_default_3: "bf16[14336, 2048]" = torch.ops.aten.permute.default(view_176, [1, 0]);  view_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_2: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_153, rsqrt_15);  convert_element_type_153 = rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_1: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.bfloat16);  mul_tensor_2 = None
        mul_tensor_3: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_64, convert_element_type_default_1);  add_64 = convert_element_type_default_1 = None
        sum_dim_int_list_1: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1], True);  mul_tensor_3 = None
        reshape_default_1: "bf16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, [4096]);  sum_dim_int_list_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_4: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_179, [1, 0]);  view_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_5: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_63, [0, 2, 1, 3]);  getitem_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_2: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_default_5, [4, 512, -1]);  permute_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_3: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(reshape_default_2, [2048, 4096]);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_6: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_185, [1, 0]);  view_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_7: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_188, [1, 0]);  view_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_8: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_191, [1, 0]);  view_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_4: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_143, rsqrt_14);  convert_element_type_143 = rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_2: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.bfloat16);  mul_tensor_4 = None
        mul_tensor_5: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_72, convert_element_type_default_2);  add_72 = convert_element_type_default_2 = None
        sum_dim_int_list_2: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1], True);  mul_tensor_5 = None
        reshape_default_4: "bf16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, [4096]);  sum_dim_int_list_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_9: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_194, [1, 0]);  view_194 = None
        permute_default_10: "bf16[14336, 2048]" = torch.ops.aten.permute.default(view_196, [1, 0]);  view_196 = None
        permute_default_11: "bf16[14336, 2048]" = torch.ops.aten.permute.default(view_198, [1, 0]);  view_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_6: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_133, rsqrt_13);  convert_element_type_133 = rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_3: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_tensor_6, torch.bfloat16);  mul_tensor_6 = None
        mul_tensor_7: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_77, convert_element_type_default_3);  add_77 = convert_element_type_default_3 = None
        sum_dim_int_list_3: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1], True);  mul_tensor_7 = None
        reshape_default_5: "bf16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, [4096]);  sum_dim_int_list_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_12: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_201, [1, 0]);  view_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_13: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_54, [0, 2, 1, 3]);  getitem_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_6: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_default_13, [4, 512, -1]);  permute_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_7: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(reshape_default_6, [2048, 4096]);  reshape_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_14: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_207, [1, 0]);  view_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_15: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_210, [1, 0]);  view_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_16: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_213, [1, 0]);  view_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_8: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_123, rsqrt_12);  convert_element_type_123 = rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_4: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_tensor_8, torch.bfloat16);  mul_tensor_8 = None
        mul_tensor_9: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_85, convert_element_type_default_4);  add_85 = convert_element_type_default_4 = None
        sum_dim_int_list_4: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1], True);  mul_tensor_9 = None
        reshape_default_8: "bf16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, [4096]);  sum_dim_int_list_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_17: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_216, [1, 0]);  view_216 = None
        permute_default_18: "bf16[14336, 2048]" = torch.ops.aten.permute.default(view_218, [1, 0]);  view_218 = None
        permute_default_19: "bf16[14336, 2048]" = torch.ops.aten.permute.default(view_220, [1, 0]);  view_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_10: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_113, rsqrt_11);  convert_element_type_113 = rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_5: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_tensor_10, torch.bfloat16);  mul_tensor_10 = None
        mul_tensor_11: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_90, convert_element_type_default_5);  add_90 = convert_element_type_default_5 = None
        sum_dim_int_list_5: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1], True);  mul_tensor_11 = None
        reshape_default_9: "bf16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_5, [4096]);  sum_dim_int_list_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_20: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_223, [1, 0]);  view_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_21: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_45, [0, 2, 1, 3]);  getitem_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_10: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_default_21, [4, 512, -1]);  permute_default_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_11: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(reshape_default_10, [2048, 4096]);  reshape_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_22: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_229, [1, 0]);  view_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_23: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_232, [1, 0]);  view_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_24: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_235, [1, 0]);  view_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_12: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_103, rsqrt_10);  convert_element_type_103 = rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_6: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_tensor_12, torch.bfloat16);  mul_tensor_12 = None
        mul_tensor_13: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_98, convert_element_type_default_6);  add_98 = convert_element_type_default_6 = None
        sum_dim_int_list_6: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1], True);  mul_tensor_13 = None
        reshape_default_12: "bf16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, [4096]);  sum_dim_int_list_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_25: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_238, [1, 0]);  view_238 = None
        permute_default_26: "bf16[14336, 2048]" = torch.ops.aten.permute.default(view_240, [1, 0]);  view_240 = None
        permute_default_27: "bf16[14336, 2048]" = torch.ops.aten.permute.default(view_242, [1, 0]);  view_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_14: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_93, rsqrt_9);  convert_element_type_93 = rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_7: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_tensor_14, torch.bfloat16);  mul_tensor_14 = None
        mul_tensor_15: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_103, convert_element_type_default_7);  add_103 = convert_element_type_default_7 = None
        sum_dim_int_list_7: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1], True);  mul_tensor_15 = None
        reshape_default_13: "bf16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, [4096]);  sum_dim_int_list_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_28: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_245, [1, 0]);  view_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_29: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_36, [0, 2, 1, 3]);  getitem_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_14: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_default_29, [4, 512, -1]);  permute_default_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_15: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(reshape_default_14, [2048, 4096]);  reshape_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_30: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_251, [1, 0]);  view_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_31: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_254, [1, 0]);  view_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_32: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_257, [1, 0]);  view_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_16: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_83, rsqrt_8);  convert_element_type_83 = rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_8: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_tensor_16, torch.bfloat16);  mul_tensor_16 = None
        mul_tensor_17: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_111, convert_element_type_default_8);  add_111 = convert_element_type_default_8 = None
        sum_dim_int_list_8: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1], True);  mul_tensor_17 = None
        reshape_default_16: "bf16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_8, [4096]);  sum_dim_int_list_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_33: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_260, [1, 0]);  view_260 = None
        permute_default_34: "bf16[14336, 2048]" = torch.ops.aten.permute.default(view_262, [1, 0]);  view_262 = None
        permute_default_35: "bf16[14336, 2048]" = torch.ops.aten.permute.default(view_264, [1, 0]);  view_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_18: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_73, rsqrt_7);  convert_element_type_73 = rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_9: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_tensor_18, torch.bfloat16);  mul_tensor_18 = None
        mul_tensor_19: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_116, convert_element_type_default_9);  add_116 = convert_element_type_default_9 = None
        sum_dim_int_list_9: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1], True);  mul_tensor_19 = None
        reshape_default_17: "bf16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_9, [4096]);  sum_dim_int_list_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_36: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_267, [1, 0]);  view_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_37: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_18: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_default_37, [4, 512, -1]);  permute_default_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_19: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(reshape_default_18, [2048, 4096]);  reshape_default_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_38: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_273, [1, 0]);  view_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_39: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_276, [1, 0]);  view_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_40: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_279, [1, 0]);  view_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_20: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_63, rsqrt_6);  convert_element_type_63 = rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_10: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_tensor_20, torch.bfloat16);  mul_tensor_20 = None
        mul_tensor_21: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_124, convert_element_type_default_10);  add_124 = convert_element_type_default_10 = None
        sum_dim_int_list_10: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1], True);  mul_tensor_21 = None
        reshape_default_20: "bf16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_10, [4096]);  sum_dim_int_list_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_41: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_282, [1, 0]);  view_282 = None
        permute_default_42: "bf16[14336, 2048]" = torch.ops.aten.permute.default(view_284, [1, 0]);  view_284 = None
        permute_default_43: "bf16[14336, 2048]" = torch.ops.aten.permute.default(view_286, [1, 0]);  view_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_22: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_53, rsqrt_5);  convert_element_type_53 = rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_11: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_tensor_22, torch.bfloat16);  mul_tensor_22 = None
        mul_tensor_23: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_129, convert_element_type_default_11);  add_129 = convert_element_type_default_11 = None
        sum_dim_int_list_11: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1], True);  mul_tensor_23 = None
        reshape_default_21: "bf16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, [4096]);  sum_dim_int_list_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_44: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_289, [1, 0]);  view_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_45: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_18, [0, 2, 1, 3]);  getitem_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_22: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_default_45, [4, 512, -1]);  permute_default_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_23: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(reshape_default_22, [2048, 4096]);  reshape_default_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_46: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_295, [1, 0]);  view_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_47: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_298, [1, 0]);  view_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_48: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_301, [1, 0]);  view_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_24: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_43, rsqrt_4);  convert_element_type_43 = rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_12: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_tensor_24, torch.bfloat16);  mul_tensor_24 = None
        mul_tensor_25: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_137, convert_element_type_default_12);  add_137 = convert_element_type_default_12 = None
        sum_dim_int_list_12: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_25, [0, 1], True);  mul_tensor_25 = None
        reshape_default_24: "bf16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, [4096]);  sum_dim_int_list_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_49: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_304, [1, 0]);  view_304 = None
        permute_default_50: "bf16[14336, 2048]" = torch.ops.aten.permute.default(view_306, [1, 0]);  view_306 = None
        permute_default_51: "bf16[14336, 2048]" = torch.ops.aten.permute.default(view_308, [1, 0]);  view_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_26: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_33, rsqrt_3);  convert_element_type_33 = rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_13: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_tensor_26, torch.bfloat16);  mul_tensor_26 = None
        mul_tensor_27: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_142, convert_element_type_default_13);  add_142 = convert_element_type_default_13 = None
        sum_dim_int_list_13: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_27, [0, 1], True);  mul_tensor_27 = None
        reshape_default_25: "bf16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_13, [4096]);  sum_dim_int_list_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_52: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_311, [1, 0]);  view_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_53: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_9, [0, 2, 1, 3]);  getitem_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_26: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_default_53, [4, 512, -1]);  permute_default_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_27: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(reshape_default_26, [2048, 4096]);  reshape_default_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_54: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_317, [1, 0]);  view_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_55: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_320, [1, 0]);  view_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_56: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_323, [1, 0]);  view_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_28: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_23, rsqrt_2);  convert_element_type_23 = rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_14: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_tensor_28, torch.bfloat16);  mul_tensor_28 = None
        mul_tensor_29: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_150, convert_element_type_default_14);  add_150 = convert_element_type_default_14 = None
        sum_dim_int_list_14: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_29, [0, 1], True);  mul_tensor_29 = None
        reshape_default_28: "bf16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_14, [4096]);  sum_dim_int_list_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_57: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_326, [1, 0]);  view_326 = None
        permute_default_58: "bf16[14336, 2048]" = torch.ops.aten.permute.default(view_328, [1, 0]);  view_328 = None
        permute_default_59: "bf16[14336, 2048]" = torch.ops.aten.permute.default(view_330, [1, 0]);  view_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_30: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_13, rsqrt_1);  convert_element_type_13 = rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_15: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_tensor_30, torch.bfloat16);  mul_tensor_30 = None
        mul_tensor_31: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_155, convert_element_type_default_15);  add_155 = convert_element_type_default_15 = None
        sum_dim_int_list_15: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_31, [0, 1], True);  mul_tensor_31 = None
        reshape_default_29: "bf16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_15, [4096]);  sum_dim_int_list_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_60: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_333, [1, 0]);  view_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_61: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3]);  getitem = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_30: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_default_61, [4, 512, -1]);  permute_default_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_31: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(reshape_default_30, [2048, 4096]);  reshape_default_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_62: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_339, [1, 0]);  view_339 = None
        reshape_default_32: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_166, [4, 512, 4096]);  mm_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_63: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_342, [1, 0]);  view_342 = None
        reshape_default_33: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_168, [4, 512, 4096]);  mm_168 = None
        add_tensor: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(reshape_default_32, reshape_default_33);  reshape_default_32 = reshape_default_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_64: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_345, [1, 0]);  view_345 = None
        reshape_default_34: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_170, [4, 512, 4096]);  mm_170 = None
        add_tensor_1: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_34);  add_tensor = reshape_default_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_tensor_32: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_tensor_1, primals_4);  primals_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default_16: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(embedding, torch.float32);  embedding = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_33: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_default_16, rsqrt)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_17: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_tensor_33, torch.bfloat16);  mul_tensor_33 = None
        mul_tensor_34: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_tensor_1, convert_element_type_default_17);  add_tensor_1 = convert_element_type_default_17 = None
        sum_dim_int_list_16: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_34, [0, 1], True);  mul_tensor_34 = None
        reshape_default_35: "bf16[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_16, [4096]);  sum_dim_int_list_16 = None
        convert_element_type_default_18: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_tensor_32, torch.float32);  mul_tensor_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_35: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_default_18, convert_element_type_default_16)
        mul_tensor_36: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_default_18, rsqrt);  convert_element_type_default_18 = None
        sum_dim_int_list_17: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_35, [2], True);  mul_tensor_35 = None
        pow_tensor_scalar: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_scalar: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_17, -0.5);  sum_dim_int_list_17 = None
        mul_tensor_37: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_default: "f32[4, 512, 4096]" = torch.ops.aten.expand.default(mul_tensor_37, [4, 512, 4096]);  mul_tensor_37 = None
        div_scalar: "f32[4, 512, 4096]" = torch.ops.aten.div.Scalar(expand_default, 4096);  expand_default = None
        pow_tensor_scalar_1: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_16, 1.0);  convert_element_type_default_16 = None
        mul_scalar_1: "f32[4, 512, 4096]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_38: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_2: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_tensor_36, mul_tensor_38);  mul_tensor_36 = mul_tensor_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default_19: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None
        add_tensor_3: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_157, convert_element_type_default_19);  add_157 = convert_element_type_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:362 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        convert_element_type_default_20: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float32);  add_tensor_3 = None
        eq_scalar: "b8[4, 512]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_default: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[4, 512, 4096]" = torch.ops.aten.where.self(unsqueeze_default, full_default_18, convert_element_type_default_20);  unsqueeze_default = full_default_18 = convert_element_type_default_20 = None
        full_default: "f32[32768, 4096]" = torch.ops.aten.full.default([32768, 4096], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[32768, 4096]" = torch.ops.aten.index_put.default(full_default, [primals_1], where_self, True);  full_default = primals_1 = where_self = None
        convert_element_type_default_21: "bf16[32768, 4096]" = torch.ops.prims.convert_element_type.default(index_put_default, torch.bfloat16);  index_put_default = None
        return (permute_default, reshape_default, permute_default_1, permute_default_2, permute_default_3, reshape_default_1, permute_default_4, reshape_default_3, permute_default_6, permute_default_7, permute_default_8, reshape_default_4, permute_default_9, permute_default_10, permute_default_11, reshape_default_5, permute_default_12, reshape_default_7, permute_default_14, permute_default_15, permute_default_16, reshape_default_8, permute_default_17, permute_default_18, permute_default_19, reshape_default_9, permute_default_20, reshape_default_11, permute_default_22, permute_default_23, permute_default_24, reshape_default_12, permute_default_25, permute_default_26, permute_default_27, reshape_default_13, permute_default_28, reshape_default_15, permute_default_30, permute_default_31, permute_default_32, reshape_default_16, permute_default_33, permute_default_34, permute_default_35, reshape_default_17, permute_default_36, reshape_default_19, permute_default_38, permute_default_39, permute_default_40, reshape_default_20, permute_default_41, permute_default_42, permute_default_43, reshape_default_21, permute_default_44, reshape_default_23, permute_default_46, permute_default_47, permute_default_48, reshape_default_24, permute_default_49, permute_default_50, permute_default_51, reshape_default_25, permute_default_52, reshape_default_27, permute_default_54, permute_default_55, permute_default_56, reshape_default_28, permute_default_57, permute_default_58, permute_default_59, reshape_default_29, permute_default_60, reshape_default_31, permute_default_62, permute_default_63, permute_default_64, reshape_default_35, convert_element_type_default_21)


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
