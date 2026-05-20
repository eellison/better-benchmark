"""
Standalone repro captured via capture_hook.
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([2048, 151936], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([16384, 2048], bf16), T([16384, 1536], bf16), T([2048, 128], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([4, 32, 512, 128], bf16, stride=(2097152, 128, 4096, 1)), T([2048, 512], bf16), T([4, 512, 4, 128], f32), T([4, 512, 4, 1], f32), T([4, 512, 4, 128], bf16, stride=(262144, 128, 65536, 1)), T([2048, 512], bf16), T([4, 512, 32, 128], f32), T([4, 512, 32, 1], f32), T([4, 512, 32, 128], bf16, stride=(2097152, 128, 65536, 1)), T([2048, 4096], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([16384, 2048], bf16), T([16384, 1536], bf16), T([2048, 128], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([4, 32, 512, 128], bf16, stride=(2097152, 128, 4096, 1)), T([2048, 512], bf16), T([4, 512, 4, 128], f32), T([4, 512, 4, 1], f32), T([4, 512, 4, 128], bf16, stride=(262144, 128, 65536, 1)), T([2048, 512], bf16), T([4, 512, 32, 128], f32), T([4, 512, 32, 1], f32), T([4, 512, 32, 128], bf16, stride=(2097152, 128, 65536, 1)), T([2048, 4096], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([16384, 2048], bf16), T([16384, 1536], bf16), T([2048, 128], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([4, 32, 512, 128], bf16, stride=(2097152, 128, 4096, 1)), T([2048, 512], bf16), T([4, 512, 4, 128], f32), T([4, 512, 4, 1], f32), T([4, 512, 4, 128], bf16, stride=(262144, 128, 65536, 1)), T([2048, 512], bf16), T([4, 512, 32, 128], f32), T([4, 512, 32, 1], f32), T([4, 512, 32, 128], bf16, stride=(2097152, 128, 65536, 1)), T([2048, 4096], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([16384, 2048], bf16), T([16384, 1536], bf16), T([2048, 128], bf16), T([4, 512, 2048], f32), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 2048], bf16), T([4, 32, 512, 128], bf16, stride=(2097152, 128, 4096, 1)), T([2048, 512], bf16), T([2048, 2048], bf16), T([4, 512, 4, 128], f32), T([4, 512, 4, 1], f32), T([4, 512, 4, 128], bf16, stride=(262144, 128, 65536, 1)), T([2048, 512], bf16), T([2048, 2048], bf16), T([4, 512, 32, 128], f32), T([4, 512, 32, 1], f32), T([4, 512, 32, 128], bf16, stride=(2097152, 128, 65536, 1)), T([2048, 4096], bf16), T([2048, 2048], bf16), T([2048], bf16), T([4, 512, 2048], bf16), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([4, 512], i64, gen=Index(100)), T([], f32))"

class Repro(torch.nn.Module):
    def forward(self, view_89: "bf16[2048, 151936]", convert_element_type_95: "f32[4, 512, 2048]", rsqrt_16: "f32[4, 512, 1]", view_90: "bf16[4, 512, 2048]", mul_72: "bf16[16384, 2048]", cat_9: "bf16[16384, 1536]", convert_element_type_112: "bf16[2048, 128]", convert_element_type_86: "f32[4, 512, 2048]", rsqrt_15: "f32[4, 512, 1]", view_96: "bf16[4, 512, 2048]", view_98: "bf16[2048, 2048]", getitem_45: "bf16[4, 32, 512, 128]", view_104: "bf16[2048, 512]", convert_element_type_80: "f32[4, 512, 4, 128]", rsqrt_14: "f32[4, 512, 4, 1]", permute_72: "bf16[4, 512, 4, 128]", view_108: "bf16[2048, 512]", convert_element_type_76: "f32[4, 512, 32, 128]", rsqrt_13: "f32[4, 512, 32, 1]", permute_77: "bf16[4, 512, 32, 128]", view_112: "bf16[2048, 4096]", convert_element_type_72: "f32[4, 512, 2048]", rsqrt_12: "f32[4, 512, 1]", add_55: "bf16[4, 512, 2048]", mul_118: "bf16[16384, 2048]", cat_10: "bf16[16384, 1536]", convert_element_type_145: "bf16[2048, 128]", convert_element_type_63: "f32[4, 512, 2048]", rsqrt_11: "f32[4, 512, 1]", view_119: "bf16[4, 512, 2048]", view_121: "bf16[2048, 2048]", getitem_30: "bf16[4, 32, 512, 128]", view_127: "bf16[2048, 512]", convert_element_type_57: "f32[4, 512, 4, 128]", rsqrt_10: "f32[4, 512, 4, 1]", permute_104: "bf16[4, 512, 4, 128]", view_131: "bf16[2048, 512]", convert_element_type_53: "f32[4, 512, 32, 128]", rsqrt_9: "f32[4, 512, 32, 1]", permute_109: "bf16[4, 512, 32, 128]", view_135: "bf16[2048, 4096]", convert_element_type_49: "f32[4, 512, 2048]", rsqrt_8: "f32[4, 512, 1]", add_71: "bf16[4, 512, 2048]", mul_164: "bf16[16384, 2048]", cat_11: "bf16[16384, 1536]", convert_element_type_178: "bf16[2048, 128]", convert_element_type_40: "f32[4, 512, 2048]", rsqrt_7: "f32[4, 512, 1]", view_142: "bf16[4, 512, 2048]", view_144: "bf16[2048, 2048]", getitem_15: "bf16[4, 32, 512, 128]", view_150: "bf16[2048, 512]", convert_element_type_34: "f32[4, 512, 4, 128]", rsqrt_6: "f32[4, 512, 4, 1]", permute_136: "bf16[4, 512, 4, 128]", view_154: "bf16[2048, 512]", convert_element_type_30: "f32[4, 512, 32, 128]", rsqrt_5: "f32[4, 512, 32, 1]", permute_141: "bf16[4, 512, 32, 128]", view_158: "bf16[2048, 4096]", convert_element_type_26: "f32[4, 512, 2048]", rsqrt_4: "f32[4, 512, 1]", add_87: "bf16[4, 512, 2048]", mul_210: "bf16[16384, 2048]", cat_12: "bf16[16384, 1536]", convert_element_type_211: "bf16[2048, 128]", convert_element_type_17: "f32[4, 512, 2048]", rsqrt_3: "f32[4, 512, 1]", view_165: "bf16[4, 512, 2048]", view_167: "bf16[2048, 2048]", getitem: "bf16[4, 32, 512, 128]", view_173: "bf16[2048, 512]", mm_58: "bf16[2048, 2048]", convert_element_type_11: "f32[4, 512, 4, 128]", rsqrt_2: "f32[4, 512, 4, 1]", permute_168: "bf16[4, 512, 4, 128]", view_177: "bf16[2048, 512]", mm_60: "bf16[2048, 2048]", convert_element_type_7: "f32[4, 512, 32, 128]", rsqrt_1: "f32[4, 512, 32, 1]", permute_173: "bf16[4, 512, 32, 128]", view_181: "bf16[2048, 4096]", mm_62: "bf16[2048, 2048]", primals_4: "bf16[2048]", embedding: "bf16[4, 512, 2048]", rsqrt: "f32[4, 512, 1]", add_95: "bf16[4, 512, 2048]", primals_1: "i64[4, 512]", full_default_18: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:684 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        permute_default: "bf16[151936, 2048]" = torch.ops.aten.permute.default(view_89, [1, 0]);  view_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_95, rsqrt_16);  convert_element_type_95 = rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        mul_tensor_1: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(view_90, convert_element_type_default);  view_90 = convert_element_type_default = None
        sum_dim_int_list: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1], True);  mul_tensor_1 = None
        reshape_default: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list, [2048]);  sum_dim_int_list = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:6781 in grouped_mm, code: return torch._grouped_mm(mat_a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype)
        permute_default_1: "bf16[2048, 16384]" = torch.ops.aten.permute.default(mul_72, [1, 0]);  mul_72 = None
        permute_default_2: "bf16[1536, 16384]" = torch.ops.aten.permute.default(cat_9, [1, 0]);  cat_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:265 in forward, code: router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        permute_default_3: "bf16[128, 2048]" = torch.ops.aten.permute.default(convert_element_type_112, [1, 0]);  convert_element_type_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_2: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_86, rsqrt_15);  convert_element_type_86 = rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_1: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.bfloat16);  mul_tensor_2 = None
        mul_tensor_3: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(view_96, convert_element_type_default_1);  view_96 = convert_element_type_default_1 = None
        sum_dim_int_list_1: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1], True);  mul_tensor_3 = None
        reshape_default_1: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, [2048]);  sum_dim_int_list_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:194 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_4: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_98, [1, 0]);  view_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_5: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_45, [0, 2, 1, 3]);  getitem_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:193 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_2: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_default_5, [4, 512, -1]);  permute_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:194 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_3: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(reshape_default_2, [2048, 4096]);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:169 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_6: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_104, [1, 0]);  view_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_4: "f32[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_80, rsqrt_14);  convert_element_type_80 = rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_2: "bf16[4, 512, 4, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.bfloat16);  mul_tensor_4 = None
        mul_tensor_5: "bf16[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(permute_72, convert_element_type_default_2);  permute_72 = convert_element_type_default_2 = None
        sum_dim_int_list_2: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1, 2], True);  mul_tensor_5 = None
        reshape_default_4: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, [128]);  sum_dim_int_list_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:168 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_7: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_108, [1, 0]);  view_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_6: "f32[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_76, rsqrt_13);  convert_element_type_76 = rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_3: "bf16[4, 512, 32, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_6, torch.bfloat16);  mul_tensor_6 = None
        mul_tensor_7: "bf16[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(permute_77, convert_element_type_default_3);  permute_77 = convert_element_type_default_3 = None
        sum_dim_int_list_3: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1, 2], True);  mul_tensor_7 = None
        reshape_default_5: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, [128]);  sum_dim_int_list_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:167 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_8: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_112, [1, 0]);  view_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_8: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_72, rsqrt_12);  convert_element_type_72 = rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_4: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_8, torch.bfloat16);  mul_tensor_8 = None
        mul_tensor_9: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_55, convert_element_type_default_4);  add_55 = convert_element_type_default_4 = None
        sum_dim_int_list_4: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1], True);  mul_tensor_9 = None
        reshape_default_6: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, [2048]);  sum_dim_int_list_4 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:6781 in grouped_mm, code: return torch._grouped_mm(mat_a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype)
        permute_default_9: "bf16[2048, 16384]" = torch.ops.aten.permute.default(mul_118, [1, 0]);  mul_118 = None
        permute_default_10: "bf16[1536, 16384]" = torch.ops.aten.permute.default(cat_10, [1, 0]);  cat_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:265 in forward, code: router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        permute_default_11: "bf16[128, 2048]" = torch.ops.aten.permute.default(convert_element_type_145, [1, 0]);  convert_element_type_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_10: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_63, rsqrt_11);  convert_element_type_63 = rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_5: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_10, torch.bfloat16);  mul_tensor_10 = None
        mul_tensor_11: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(view_119, convert_element_type_default_5);  view_119 = convert_element_type_default_5 = None
        sum_dim_int_list_5: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1], True);  mul_tensor_11 = None
        reshape_default_7: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_5, [2048]);  sum_dim_int_list_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:194 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_12: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_121, [1, 0]);  view_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_13: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_30, [0, 2, 1, 3]);  getitem_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:193 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_8: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_default_13, [4, 512, -1]);  permute_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:194 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_9: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(reshape_default_8, [2048, 4096]);  reshape_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:169 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_14: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_127, [1, 0]);  view_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_12: "f32[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_57, rsqrt_10);  convert_element_type_57 = rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_6: "bf16[4, 512, 4, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_12, torch.bfloat16);  mul_tensor_12 = None
        mul_tensor_13: "bf16[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(permute_104, convert_element_type_default_6);  permute_104 = convert_element_type_default_6 = None
        sum_dim_int_list_6: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1, 2], True);  mul_tensor_13 = None
        reshape_default_10: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, [128]);  sum_dim_int_list_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:168 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_15: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_131, [1, 0]);  view_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_14: "f32[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_53, rsqrt_9);  convert_element_type_53 = rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_7: "bf16[4, 512, 32, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_14, torch.bfloat16);  mul_tensor_14 = None
        mul_tensor_15: "bf16[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(permute_109, convert_element_type_default_7);  permute_109 = convert_element_type_default_7 = None
        sum_dim_int_list_7: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1, 2], True);  mul_tensor_15 = None
        reshape_default_11: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, [128]);  sum_dim_int_list_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:167 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_16: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_135, [1, 0]);  view_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_16: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_49, rsqrt_8);  convert_element_type_49 = rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_8: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_16, torch.bfloat16);  mul_tensor_16 = None
        mul_tensor_17: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_71, convert_element_type_default_8);  add_71 = convert_element_type_default_8 = None
        sum_dim_int_list_8: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1], True);  mul_tensor_17 = None
        reshape_default_12: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_8, [2048]);  sum_dim_int_list_8 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:6781 in grouped_mm, code: return torch._grouped_mm(mat_a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype)
        permute_default_17: "bf16[2048, 16384]" = torch.ops.aten.permute.default(mul_164, [1, 0]);  mul_164 = None
        permute_default_18: "bf16[1536, 16384]" = torch.ops.aten.permute.default(cat_11, [1, 0]);  cat_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:265 in forward, code: router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        permute_default_19: "bf16[128, 2048]" = torch.ops.aten.permute.default(convert_element_type_178, [1, 0]);  convert_element_type_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_18: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_40, rsqrt_7);  convert_element_type_40 = rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_9: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_18, torch.bfloat16);  mul_tensor_18 = None
        mul_tensor_19: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(view_142, convert_element_type_default_9);  view_142 = convert_element_type_default_9 = None
        sum_dim_int_list_9: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1], True);  mul_tensor_19 = None
        reshape_default_13: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_9, [2048]);  sum_dim_int_list_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:194 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_20: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_144, [1, 0]);  view_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_21: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_15, [0, 2, 1, 3]);  getitem_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:193 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_14: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_default_21, [4, 512, -1]);  permute_default_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:194 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_15: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(reshape_default_14, [2048, 4096]);  reshape_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:169 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_22: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_150, [1, 0]);  view_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_20: "f32[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_34, rsqrt_6);  convert_element_type_34 = rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_10: "bf16[4, 512, 4, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_20, torch.bfloat16);  mul_tensor_20 = None
        mul_tensor_21: "bf16[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(permute_136, convert_element_type_default_10);  permute_136 = convert_element_type_default_10 = None
        sum_dim_int_list_10: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1, 2], True);  mul_tensor_21 = None
        reshape_default_16: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_10, [128]);  sum_dim_int_list_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:168 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_23: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_154, [1, 0]);  view_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_22: "f32[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_30, rsqrt_5);  convert_element_type_30 = rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_11: "bf16[4, 512, 32, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_22, torch.bfloat16);  mul_tensor_22 = None
        mul_tensor_23: "bf16[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(permute_141, convert_element_type_default_11);  permute_141 = convert_element_type_default_11 = None
        sum_dim_int_list_11: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1, 2], True);  mul_tensor_23 = None
        reshape_default_17: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, [128]);  sum_dim_int_list_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:167 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_24: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_158, [1, 0]);  view_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_24: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_26, rsqrt_4);  convert_element_type_26 = rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_12: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_24, torch.bfloat16);  mul_tensor_24 = None
        mul_tensor_25: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_87, convert_element_type_default_12);  add_87 = convert_element_type_default_12 = None
        sum_dim_int_list_12: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_25, [0, 1], True);  mul_tensor_25 = None
        reshape_default_18: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, [2048]);  sum_dim_int_list_12 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:6781 in grouped_mm, code: return torch._grouped_mm(mat_a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype)
        permute_default_25: "bf16[2048, 16384]" = torch.ops.aten.permute.default(mul_210, [1, 0]);  mul_210 = None
        permute_default_26: "bf16[1536, 16384]" = torch.ops.aten.permute.default(cat_12, [1, 0]);  cat_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:265 in forward, code: router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        permute_default_27: "bf16[128, 2048]" = torch.ops.aten.permute.default(convert_element_type_211, [1, 0]);  convert_element_type_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_26: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_17, rsqrt_3);  convert_element_type_17 = rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_13: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_26, torch.bfloat16);  mul_tensor_26 = None
        mul_tensor_27: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(view_165, convert_element_type_default_13);  view_165 = convert_element_type_default_13 = None
        sum_dim_int_list_13: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_27, [0, 1], True);  mul_tensor_27 = None
        reshape_default_19: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_13, [2048]);  sum_dim_int_list_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:194 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_28: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_167, [1, 0]);  view_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_29: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3]);  getitem = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:193 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_20: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_default_29, [4, 512, -1]);  permute_default_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:194 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_21: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(reshape_default_20, [2048, 4096]);  reshape_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:169 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_30: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_173, [1, 0]);  view_173 = None
        reshape_default_22: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_58, [4, 512, 2048]);  mm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_28: "f32[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_11, rsqrt_2);  convert_element_type_11 = rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_14: "bf16[4, 512, 4, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_28, torch.bfloat16);  mul_tensor_28 = None
        mul_tensor_29: "bf16[4, 512, 4, 128]" = torch.ops.aten.mul.Tensor(permute_168, convert_element_type_default_14);  permute_168 = convert_element_type_default_14 = None
        sum_dim_int_list_14: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_29, [0, 1, 2], True);  mul_tensor_29 = None
        reshape_default_23: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_14, [128]);  sum_dim_int_list_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:168 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_31: "bf16[512, 2048]" = torch.ops.aten.permute.default(view_177, [1, 0]);  view_177 = None
        reshape_default_24: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_60, [4, 512, 2048]);  mm_60 = None
        add_tensor: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(reshape_default_22, reshape_default_24);  reshape_default_22 = reshape_default_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_30: "f32[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_7, rsqrt_1);  convert_element_type_7 = rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_15: "bf16[4, 512, 32, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_30, torch.bfloat16);  mul_tensor_30 = None
        mul_tensor_31: "bf16[4, 512, 32, 128]" = torch.ops.aten.mul.Tensor(permute_173, convert_element_type_default_15);  permute_173 = convert_element_type_default_15 = None
        sum_dim_int_list_15: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_31, [0, 1, 2], True);  mul_tensor_31 = None
        reshape_default_25: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_15, [128]);  sum_dim_int_list_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:167 in forward, code: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_default_32: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_181, [1, 0]);  view_181 = None
        reshape_default_26: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(mm_62, [4, 512, 2048]);  mm_62 = None
        add_tensor_1: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_26);  add_tensor = reshape_default_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_tensor_32: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_tensor_1, primals_4);  primals_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default_16: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(embedding, torch.float32);  embedding = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_33: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default_16, rsqrt)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_17: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_33, torch.bfloat16);  mul_tensor_33 = None
        mul_tensor_34: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_tensor_1, convert_element_type_default_17);  add_tensor_1 = convert_element_type_default_17 = None
        sum_dim_int_list_16: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_34, [0, 1], True);  mul_tensor_34 = None
        reshape_default_27: "bf16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_16, [2048]);  sum_dim_int_list_16 = None
        convert_element_type_default_18: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_32, torch.float32);  mul_tensor_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_35: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default_18, convert_element_type_default_16)
        mul_tensor_36: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default_18, rsqrt);  convert_element_type_default_18 = None
        sum_dim_int_list_17: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_35, [2], True);  mul_tensor_35 = None
        pow_tensor_scalar: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_scalar: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_17, -0.5);  sum_dim_int_list_17 = None
        mul_tensor_37: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_default: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_tensor_37, [4, 512, 2048]);  mul_tensor_37 = None
        div_scalar: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_default, 2048);  expand_default = None
        pow_tensor_scalar_1: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_16, 1.0);  convert_element_type_default_16 = None
        mul_scalar_1: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_38: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_2: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_tensor_36, mul_tensor_38);  mul_tensor_36 = mul_tensor_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default_19: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None
        add_tensor_3: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_95, convert_element_type_default_19);  add_95 = convert_element_type_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:489 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        convert_element_type_default_20: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float32);  add_tensor_3 = None
        eq_scalar: "b8[4, 512]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_default: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[4, 512, 2048]" = torch.ops.aten.where.self(unsqueeze_default, full_default_18, convert_element_type_default_20);  unsqueeze_default = full_default_18 = convert_element_type_default_20 = None
        full_default: "f32[151936, 2048]" = torch.ops.aten.full.default([151936, 2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[151936, 2048]" = torch.ops.aten.index_put.default(full_default, [primals_1], where_self, True);  full_default = primals_1 = where_self = None
        convert_element_type_default_21: "bf16[151936, 2048]" = torch.ops.prims.convert_element_type.default(index_put_default, torch.bfloat16);  index_put_default = None
        return (permute_default, reshape_default, permute_default_1, permute_default_2, permute_default_3, reshape_default_1, permute_default_4, reshape_default_3, permute_default_6, reshape_default_4, permute_default_7, reshape_default_5, permute_default_8, reshape_default_6, permute_default_9, permute_default_10, permute_default_11, reshape_default_7, permute_default_12, reshape_default_9, permute_default_14, reshape_default_10, permute_default_15, reshape_default_11, permute_default_16, reshape_default_12, permute_default_17, permute_default_18, permute_default_19, reshape_default_13, permute_default_20, reshape_default_15, permute_default_22, reshape_default_16, permute_default_23, reshape_default_17, permute_default_24, reshape_default_18, permute_default_25, permute_default_26, permute_default_27, reshape_default_19, permute_default_28, reshape_default_21, permute_default_30, reshape_default_23, permute_default_31, reshape_default_25, permute_default_32, reshape_default_27, convert_element_type_default_21)


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
