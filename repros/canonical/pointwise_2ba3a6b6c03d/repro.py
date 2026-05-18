"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_training
Pattern hash: 2ba3a6b6c03d
Shape hash: 35298243
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_48: "f32[48, 128, 64]", _shape_param_0, bmm_50: "f32[48, 64, 128]", _shape_param_1, _shape_param_2, _shape_param_3, primals_185: "f32[384, 512]", _shape_param_4, _shape_param_5, primals_184: "f32[384, 512]", bmm_56: "f32[48, 128, 64]", _shape_param_6, bmm_58: "f32[48, 64, 128]", _shape_param_7, _shape_param_8, _shape_param_9, primals_171: "f32[384, 512]", _shape_param_10, _shape_param_11, primals_170: "f32[384, 512]", bmm_64: "f32[48, 128, 64]", _shape_param_12, bmm_66: "f32[48, 64, 128]", _shape_param_13, _shape_param_14, _shape_param_15, primals_157: "f32[384, 512]", _shape_param_16, _shape_param_17, primals_156: "f32[384, 512]", bmm_72: "f32[48, 128, 64]", _shape_param_18, bmm_74: "f32[48, 64, 128]", _shape_param_19, _shape_param_20, _shape_param_21, primals_143: "f32[384, 512]", _shape_param_22, _shape_param_23, primals_142: "f32[384, 512]", bmm_80: "f32[48, 128, 64]", _shape_param_24, bmm_82: "f32[48, 64, 128]", _shape_param_25, _shape_param_26, _shape_param_27, primals_129: "f32[384, 512]", _shape_param_28, _shape_param_29, primals_128: "f32[384, 512]", bmm_88: "f32[48, 128, 64]", _shape_param_30, bmm_90: "f32[48, 64, 128]", _shape_param_31, _shape_param_32, _shape_param_33, primals_115: "f32[384, 512]", _shape_param_34, _shape_param_35, primals_114: "f32[384, 512]", bmm_96: "f32[48, 128, 64]", _shape_param_36, bmm_98: "f32[48, 64, 128]", _shape_param_37, _shape_param_38, _shape_param_39, primals_101: "f32[384, 512]", _shape_param_40, _shape_param_41, primals_100: "f32[384, 512]", bmm_104: "f32[48, 128, 64]", _shape_param_42, bmm_106: "f32[48, 64, 128]", _shape_param_43, _shape_param_44, _shape_param_45, primals_87: "f32[384, 512]", _shape_param_46, _shape_param_47, primals_86: "f32[384, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default: "f32[8, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_48, _shape_param_0);  bmm_48 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_1: "f32[8, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_50, _shape_param_1);  bmm_50 = _shape_param_1 = None
        permute_default: "f32[8, 6, 128, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_1: "f32[8, 128, 6, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f32[8, 128, 6, 64]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_2: "f32[8, 128, 384]" = torch.ops.aten.reshape.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None
        reshape_default_3: "f32[1024, 384]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_2: "f32[512, 384]" = torch.ops.aten.permute.default(primals_185, [1, 0]);  primals_185 = None
        permute_default_3: "f32[384, 512]" = torch.ops.aten.permute.default(permute_default_2, [1, 0]);  permute_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_4: "f32[8, 128, 6, 64]" = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        reshape_default_4: "f32[8, 128, 384]" = torch.ops.aten.reshape.default(permute_default_4, _shape_param_4);  permute_default_4 = _shape_param_4 = None
        clone_default_1: "f32[8, 128, 384]" = torch.ops.aten.clone.default(reshape_default_4, memory_format = torch.contiguous_format);  reshape_default_4 = None
        reshape_default_5: "f32[1024, 384]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_5);  clone_default_1 = _shape_param_5 = None
        permute_default_5: "f32[512, 384]" = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
        permute_default_6: "f32[384, 512]" = torch.ops.aten.permute.default(permute_default_5, [1, 0]);  permute_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default_6: "f32[8, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_56, _shape_param_6);  bmm_56 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_7: "f32[8, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_58, _shape_param_7);  bmm_58 = _shape_param_7 = None
        permute_default_7: "f32[8, 6, 128, 64]" = torch.ops.aten.permute.default(reshape_default_7, [0, 1, 3, 2]);  reshape_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_8: "f32[8, 128, 6, 64]" = torch.ops.aten.permute.default(reshape_default_6, [0, 2, 1, 3]);  reshape_default_6 = None
        clone_default_2: "f32[8, 128, 6, 64]" = torch.ops.aten.clone.default(permute_default_8, memory_format = torch.contiguous_format);  permute_default_8 = None
        reshape_default_8: "f32[8, 128, 384]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_8);  clone_default_2 = _shape_param_8 = None
        reshape_default_9: "f32[1024, 384]" = torch.ops.aten.reshape.default(reshape_default_8, _shape_param_9);  reshape_default_8 = _shape_param_9 = None
        permute_default_9: "f32[512, 384]" = torch.ops.aten.permute.default(primals_171, [1, 0]);  primals_171 = None
        permute_default_10: "f32[384, 512]" = torch.ops.aten.permute.default(permute_default_9, [1, 0]);  permute_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_11: "f32[8, 128, 6, 64]" = torch.ops.aten.permute.default(permute_default_7, [0, 2, 1, 3]);  permute_default_7 = None
        reshape_default_10: "f32[8, 128, 384]" = torch.ops.aten.reshape.default(permute_default_11, _shape_param_10);  permute_default_11 = _shape_param_10 = None
        clone_default_3: "f32[8, 128, 384]" = torch.ops.aten.clone.default(reshape_default_10, memory_format = torch.contiguous_format);  reshape_default_10 = None
        reshape_default_11: "f32[1024, 384]" = torch.ops.aten.reshape.default(clone_default_3, _shape_param_11);  clone_default_3 = _shape_param_11 = None
        permute_default_12: "f32[512, 384]" = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
        permute_default_13: "f32[384, 512]" = torch.ops.aten.permute.default(permute_default_12, [1, 0]);  permute_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default_12: "f32[8, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_64, _shape_param_12);  bmm_64 = _shape_param_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_13: "f32[8, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_66, _shape_param_13);  bmm_66 = _shape_param_13 = None
        permute_default_14: "f32[8, 6, 128, 64]" = torch.ops.aten.permute.default(reshape_default_13, [0, 1, 3, 2]);  reshape_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_15: "f32[8, 128, 6, 64]" = torch.ops.aten.permute.default(reshape_default_12, [0, 2, 1, 3]);  reshape_default_12 = None
        clone_default_4: "f32[8, 128, 6, 64]" = torch.ops.aten.clone.default(permute_default_15, memory_format = torch.contiguous_format);  permute_default_15 = None
        reshape_default_14: "f32[8, 128, 384]" = torch.ops.aten.reshape.default(clone_default_4, _shape_param_14);  clone_default_4 = _shape_param_14 = None
        reshape_default_15: "f32[1024, 384]" = torch.ops.aten.reshape.default(reshape_default_14, _shape_param_15);  reshape_default_14 = _shape_param_15 = None
        permute_default_16: "f32[512, 384]" = torch.ops.aten.permute.default(primals_157, [1, 0]);  primals_157 = None
        permute_default_17: "f32[384, 512]" = torch.ops.aten.permute.default(permute_default_16, [1, 0]);  permute_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_18: "f32[8, 128, 6, 64]" = torch.ops.aten.permute.default(permute_default_14, [0, 2, 1, 3]);  permute_default_14 = None
        reshape_default_16: "f32[8, 128, 384]" = torch.ops.aten.reshape.default(permute_default_18, _shape_param_16);  permute_default_18 = _shape_param_16 = None
        clone_default_5: "f32[8, 128, 384]" = torch.ops.aten.clone.default(reshape_default_16, memory_format = torch.contiguous_format);  reshape_default_16 = None
        reshape_default_17: "f32[1024, 384]" = torch.ops.aten.reshape.default(clone_default_5, _shape_param_17);  clone_default_5 = _shape_param_17 = None
        permute_default_19: "f32[512, 384]" = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
        permute_default_20: "f32[384, 512]" = torch.ops.aten.permute.default(permute_default_19, [1, 0]);  permute_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default_18: "f32[8, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_72, _shape_param_18);  bmm_72 = _shape_param_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_19: "f32[8, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_74, _shape_param_19);  bmm_74 = _shape_param_19 = None
        permute_default_21: "f32[8, 6, 128, 64]" = torch.ops.aten.permute.default(reshape_default_19, [0, 1, 3, 2]);  reshape_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_22: "f32[8, 128, 6, 64]" = torch.ops.aten.permute.default(reshape_default_18, [0, 2, 1, 3]);  reshape_default_18 = None
        clone_default_6: "f32[8, 128, 6, 64]" = torch.ops.aten.clone.default(permute_default_22, memory_format = torch.contiguous_format);  permute_default_22 = None
        reshape_default_20: "f32[8, 128, 384]" = torch.ops.aten.reshape.default(clone_default_6, _shape_param_20);  clone_default_6 = _shape_param_20 = None
        reshape_default_21: "f32[1024, 384]" = torch.ops.aten.reshape.default(reshape_default_20, _shape_param_21);  reshape_default_20 = _shape_param_21 = None
        permute_default_23: "f32[512, 384]" = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
        permute_default_24: "f32[384, 512]" = torch.ops.aten.permute.default(permute_default_23, [1, 0]);  permute_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_25: "f32[8, 128, 6, 64]" = torch.ops.aten.permute.default(permute_default_21, [0, 2, 1, 3]);  permute_default_21 = None
        reshape_default_22: "f32[8, 128, 384]" = torch.ops.aten.reshape.default(permute_default_25, _shape_param_22);  permute_default_25 = _shape_param_22 = None
        clone_default_7: "f32[8, 128, 384]" = torch.ops.aten.clone.default(reshape_default_22, memory_format = torch.contiguous_format);  reshape_default_22 = None
        reshape_default_23: "f32[1024, 384]" = torch.ops.aten.reshape.default(clone_default_7, _shape_param_23);  clone_default_7 = _shape_param_23 = None
        permute_default_26: "f32[512, 384]" = torch.ops.aten.permute.default(primals_142, [1, 0]);  primals_142 = None
        permute_default_27: "f32[384, 512]" = torch.ops.aten.permute.default(permute_default_26, [1, 0]);  permute_default_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default_24: "f32[8, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_80, _shape_param_24);  bmm_80 = _shape_param_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_25: "f32[8, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_82, _shape_param_25);  bmm_82 = _shape_param_25 = None
        permute_default_28: "f32[8, 6, 128, 64]" = torch.ops.aten.permute.default(reshape_default_25, [0, 1, 3, 2]);  reshape_default_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_29: "f32[8, 128, 6, 64]" = torch.ops.aten.permute.default(reshape_default_24, [0, 2, 1, 3]);  reshape_default_24 = None
        clone_default_8: "f32[8, 128, 6, 64]" = torch.ops.aten.clone.default(permute_default_29, memory_format = torch.contiguous_format);  permute_default_29 = None
        reshape_default_26: "f32[8, 128, 384]" = torch.ops.aten.reshape.default(clone_default_8, _shape_param_26);  clone_default_8 = _shape_param_26 = None
        reshape_default_27: "f32[1024, 384]" = torch.ops.aten.reshape.default(reshape_default_26, _shape_param_27);  reshape_default_26 = _shape_param_27 = None
        permute_default_30: "f32[512, 384]" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
        permute_default_31: "f32[384, 512]" = torch.ops.aten.permute.default(permute_default_30, [1, 0]);  permute_default_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_32: "f32[8, 128, 6, 64]" = torch.ops.aten.permute.default(permute_default_28, [0, 2, 1, 3]);  permute_default_28 = None
        reshape_default_28: "f32[8, 128, 384]" = torch.ops.aten.reshape.default(permute_default_32, _shape_param_28);  permute_default_32 = _shape_param_28 = None
        clone_default_9: "f32[8, 128, 384]" = torch.ops.aten.clone.default(reshape_default_28, memory_format = torch.contiguous_format);  reshape_default_28 = None
        reshape_default_29: "f32[1024, 384]" = torch.ops.aten.reshape.default(clone_default_9, _shape_param_29);  clone_default_9 = _shape_param_29 = None
        permute_default_33: "f32[512, 384]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
        permute_default_34: "f32[384, 512]" = torch.ops.aten.permute.default(permute_default_33, [1, 0]);  permute_default_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default_30: "f32[8, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_88, _shape_param_30);  bmm_88 = _shape_param_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_31: "f32[8, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_90, _shape_param_31);  bmm_90 = _shape_param_31 = None
        permute_default_35: "f32[8, 6, 128, 64]" = torch.ops.aten.permute.default(reshape_default_31, [0, 1, 3, 2]);  reshape_default_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_36: "f32[8, 128, 6, 64]" = torch.ops.aten.permute.default(reshape_default_30, [0, 2, 1, 3]);  reshape_default_30 = None
        clone_default_10: "f32[8, 128, 6, 64]" = torch.ops.aten.clone.default(permute_default_36, memory_format = torch.contiguous_format);  permute_default_36 = None
        reshape_default_32: "f32[8, 128, 384]" = torch.ops.aten.reshape.default(clone_default_10, _shape_param_32);  clone_default_10 = _shape_param_32 = None
        reshape_default_33: "f32[1024, 384]" = torch.ops.aten.reshape.default(reshape_default_32, _shape_param_33);  reshape_default_32 = _shape_param_33 = None
        permute_default_37: "f32[512, 384]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
        permute_default_38: "f32[384, 512]" = torch.ops.aten.permute.default(permute_default_37, [1, 0]);  permute_default_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_39: "f32[8, 128, 6, 64]" = torch.ops.aten.permute.default(permute_default_35, [0, 2, 1, 3]);  permute_default_35 = None
        reshape_default_34: "f32[8, 128, 384]" = torch.ops.aten.reshape.default(permute_default_39, _shape_param_34);  permute_default_39 = _shape_param_34 = None
        clone_default_11: "f32[8, 128, 384]" = torch.ops.aten.clone.default(reshape_default_34, memory_format = torch.contiguous_format);  reshape_default_34 = None
        reshape_default_35: "f32[1024, 384]" = torch.ops.aten.reshape.default(clone_default_11, _shape_param_35);  clone_default_11 = _shape_param_35 = None
        permute_default_40: "f32[512, 384]" = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
        permute_default_41: "f32[384, 512]" = torch.ops.aten.permute.default(permute_default_40, [1, 0]);  permute_default_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default_36: "f32[8, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_96, _shape_param_36);  bmm_96 = _shape_param_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_37: "f32[8, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_98, _shape_param_37);  bmm_98 = _shape_param_37 = None
        permute_default_42: "f32[8, 6, 128, 64]" = torch.ops.aten.permute.default(reshape_default_37, [0, 1, 3, 2]);  reshape_default_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_43: "f32[8, 128, 6, 64]" = torch.ops.aten.permute.default(reshape_default_36, [0, 2, 1, 3]);  reshape_default_36 = None
        clone_default_12: "f32[8, 128, 6, 64]" = torch.ops.aten.clone.default(permute_default_43, memory_format = torch.contiguous_format);  permute_default_43 = None
        reshape_default_38: "f32[8, 128, 384]" = torch.ops.aten.reshape.default(clone_default_12, _shape_param_38);  clone_default_12 = _shape_param_38 = None
        reshape_default_39: "f32[1024, 384]" = torch.ops.aten.reshape.default(reshape_default_38, _shape_param_39);  reshape_default_38 = _shape_param_39 = None
        permute_default_44: "f32[512, 384]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
        permute_default_45: "f32[384, 512]" = torch.ops.aten.permute.default(permute_default_44, [1, 0]);  permute_default_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_46: "f32[8, 128, 6, 64]" = torch.ops.aten.permute.default(permute_default_42, [0, 2, 1, 3]);  permute_default_42 = None
        reshape_default_40: "f32[8, 128, 384]" = torch.ops.aten.reshape.default(permute_default_46, _shape_param_40);  permute_default_46 = _shape_param_40 = None
        clone_default_13: "f32[8, 128, 384]" = torch.ops.aten.clone.default(reshape_default_40, memory_format = torch.contiguous_format);  reshape_default_40 = None
        reshape_default_41: "f32[1024, 384]" = torch.ops.aten.reshape.default(clone_default_13, _shape_param_41);  clone_default_13 = _shape_param_41 = None
        permute_default_47: "f32[512, 384]" = torch.ops.aten.permute.default(primals_100, [1, 0]);  primals_100 = None
        permute_default_48: "f32[384, 512]" = torch.ops.aten.permute.default(permute_default_47, [1, 0]);  permute_default_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default_42: "f32[8, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_104, _shape_param_42);  bmm_104 = _shape_param_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_43: "f32[8, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_106, _shape_param_43);  bmm_106 = _shape_param_43 = None
        permute_default_49: "f32[8, 6, 128, 64]" = torch.ops.aten.permute.default(reshape_default_43, [0, 1, 3, 2]);  reshape_default_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_50: "f32[8, 128, 6, 64]" = torch.ops.aten.permute.default(reshape_default_42, [0, 2, 1, 3]);  reshape_default_42 = None
        clone_default_14: "f32[8, 128, 6, 64]" = torch.ops.aten.clone.default(permute_default_50, memory_format = torch.contiguous_format);  permute_default_50 = None
        reshape_default_44: "f32[8, 128, 384]" = torch.ops.aten.reshape.default(clone_default_14, _shape_param_44);  clone_default_14 = _shape_param_44 = None
        reshape_default_45: "f32[1024, 384]" = torch.ops.aten.reshape.default(reshape_default_44, _shape_param_45);  reshape_default_44 = _shape_param_45 = None
        permute_default_51: "f32[512, 384]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
        permute_default_52: "f32[384, 512]" = torch.ops.aten.permute.default(permute_default_51, [1, 0]);  permute_default_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_53: "f32[8, 128, 6, 64]" = torch.ops.aten.permute.default(permute_default_49, [0, 2, 1, 3]);  permute_default_49 = None
        reshape_default_46: "f32[8, 128, 384]" = torch.ops.aten.reshape.default(permute_default_53, _shape_param_46);  permute_default_53 = _shape_param_46 = None
        clone_default_15: "f32[8, 128, 384]" = torch.ops.aten.clone.default(reshape_default_46, memory_format = torch.contiguous_format);  reshape_default_46 = None
        reshape_default_47: "f32[1024, 384]" = torch.ops.aten.reshape.default(clone_default_15, _shape_param_47);  clone_default_15 = _shape_param_47 = None
        permute_default_54: "f32[512, 384]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
        permute_default_55: "f32[384, 512]" = torch.ops.aten.permute.default(permute_default_54, [1, 0]);  permute_default_54 = None
        return (reshape_default_3, permute_default_3, reshape_default_5, permute_default_6, reshape_default_9, permute_default_10, reshape_default_11, permute_default_13, reshape_default_15, permute_default_17, reshape_default_17, permute_default_20, reshape_default_21, permute_default_24, reshape_default_23, permute_default_27, reshape_default_27, permute_default_31, reshape_default_29, permute_default_34, reshape_default_33, permute_default_38, reshape_default_35, permute_default_41, reshape_default_39, permute_default_45, reshape_default_41, permute_default_48, reshape_default_45, permute_default_52, reshape_default_47, permute_default_55)


def _default_make_inputs():
    return [
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    [8, 6, 128, 64],  # _shape_param_0
    torch.randn([48, 64, 128], dtype=torch.float32, device='cuda'),
    [8, 6, 64, 128],  # _shape_param_1
    [8, 128, 384],  # _shape_param_2
    [1024, 384],  # _shape_param_3
    torch.randn([384, 512], dtype=torch.float32, device='cuda'),
    [8, 128, 384],  # _shape_param_4
    [1024, 384],  # _shape_param_5
    torch.randn([384, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    [8, 6, 128, 64],  # _shape_param_6
    torch.randn([48, 64, 128], dtype=torch.float32, device='cuda'),
    [8, 6, 64, 128],  # _shape_param_7
    [8, 128, 384],  # _shape_param_8
    [1024, 384],  # _shape_param_9
    torch.randn([384, 512], dtype=torch.float32, device='cuda'),
    [8, 128, 384],  # _shape_param_10
    [1024, 384],  # _shape_param_11
    torch.randn([384, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    [8, 6, 128, 64],  # _shape_param_12
    torch.randn([48, 64, 128], dtype=torch.float32, device='cuda'),
    [8, 6, 64, 128],  # _shape_param_13
    [8, 128, 384],  # _shape_param_14
    [1024, 384],  # _shape_param_15
    torch.randn([384, 512], dtype=torch.float32, device='cuda'),
    [8, 128, 384],  # _shape_param_16
    [1024, 384],  # _shape_param_17
    torch.randn([384, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    [8, 6, 128, 64],  # _shape_param_18
    torch.randn([48, 64, 128], dtype=torch.float32, device='cuda'),
    [8, 6, 64, 128],  # _shape_param_19
    [8, 128, 384],  # _shape_param_20
    [1024, 384],  # _shape_param_21
    torch.randn([384, 512], dtype=torch.float32, device='cuda'),
    [8, 128, 384],  # _shape_param_22
    [1024, 384],  # _shape_param_23
    torch.randn([384, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    [8, 6, 128, 64],  # _shape_param_24
    torch.randn([48, 64, 128], dtype=torch.float32, device='cuda'),
    [8, 6, 64, 128],  # _shape_param_25
    [8, 128, 384],  # _shape_param_26
    [1024, 384],  # _shape_param_27
    torch.randn([384, 512], dtype=torch.float32, device='cuda'),
    [8, 128, 384],  # _shape_param_28
    [1024, 384],  # _shape_param_29
    torch.randn([384, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    [8, 6, 128, 64],  # _shape_param_30
    torch.randn([48, 64, 128], dtype=torch.float32, device='cuda'),
    [8, 6, 64, 128],  # _shape_param_31
    [8, 128, 384],  # _shape_param_32
    [1024, 384],  # _shape_param_33
    torch.randn([384, 512], dtype=torch.float32, device='cuda'),
    [8, 128, 384],  # _shape_param_34
    [1024, 384],  # _shape_param_35
    torch.randn([384, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    [8, 6, 128, 64],  # _shape_param_36
    torch.randn([48, 64, 128], dtype=torch.float32, device='cuda'),
    [8, 6, 64, 128],  # _shape_param_37
    [8, 128, 384],  # _shape_param_38
    [1024, 384],  # _shape_param_39
    torch.randn([384, 512], dtype=torch.float32, device='cuda'),
    [8, 128, 384],  # _shape_param_40
    [1024, 384],  # _shape_param_41
    torch.randn([384, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    [8, 6, 128, 64],  # _shape_param_42
    torch.randn([48, 64, 128], dtype=torch.float32, device='cuda'),
    [8, 6, 64, 128],  # _shape_param_43
    [8, 128, 384],  # _shape_param_44
    [1024, 384],  # _shape_param_45
    torch.randn([384, 512], dtype=torch.float32, device='cuda'),
    [8, 128, 384],  # _shape_param_46
    [1024, 384],  # _shape_param_47
    torch.randn([384, 512], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
