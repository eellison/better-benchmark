"""
Standalone repro captured via capture_hook.
Label: hf_t5_base_train
Pattern hash: 614292c5c828
Shape hash: 70142a7c
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
    def forward(self, bmm_48: "f32[48, 512, 64]", _shape_param_0, tangents_23: "f32[4, 12, 512, 64]", bmm_50: "f32[48, 64, 512]", _shape_param_1, tangents_24: "f32[4, 12, 512, 64]", _shape_param_2, _shape_param_3, primals_155: "f32[768, 768]", _shape_param_4, _shape_param_5, primals_154: "f32[768, 768]", bmm_56: "f32[48, 512, 64]", _shape_param_6, tangents_21: "f32[4, 12, 512, 64]", bmm_58: "f32[48, 64, 512]", _shape_param_7, tangents_22: "f32[4, 12, 512, 64]", _shape_param_8, _shape_param_9, primals_142: "f32[768, 768]", _shape_param_10, _shape_param_11, primals_141: "f32[768, 768]", bmm_64: "f32[48, 512, 64]", _shape_param_12, tangents_19: "f32[4, 12, 512, 64]", bmm_66: "f32[48, 64, 512]", _shape_param_13, tangents_20: "f32[4, 12, 512, 64]", _shape_param_14, _shape_param_15, primals_129: "f32[768, 768]", _shape_param_16, _shape_param_17, primals_128: "f32[768, 768]", bmm_72: "f32[48, 512, 64]", _shape_param_18, tangents_17: "f32[4, 12, 512, 64]", bmm_74: "f32[48, 64, 512]", _shape_param_19, tangents_18: "f32[4, 12, 512, 64]", _shape_param_20, _shape_param_21, primals_116: "f32[768, 768]", _shape_param_22, _shape_param_23, primals_115: "f32[768, 768]", bmm_80: "f32[48, 512, 64]", _shape_param_24, tangents_15: "f32[4, 12, 512, 64]", bmm_82: "f32[48, 64, 512]", _shape_param_25, tangents_16: "f32[4, 12, 512, 64]", _shape_param_26, _shape_param_27, primals_103: "f32[768, 768]", _shape_param_28, _shape_param_29, primals_102: "f32[768, 768]", bmm_88: "f32[48, 512, 64]", _shape_param_30, tangents_13: "f32[4, 12, 512, 64]", bmm_90: "f32[48, 64, 512]", _shape_param_31, tangents_14: "f32[4, 12, 512, 64]", _shape_param_32, _shape_param_33, primals_90: "f32[768, 768]", _shape_param_34, _shape_param_35, primals_89: "f32[768, 768]", bmm_96: "f32[48, 512, 64]", _shape_param_36, tangents_11: "f32[4, 12, 512, 64]", bmm_98: "f32[48, 64, 512]", _shape_param_37, tangents_12: "f32[4, 12, 512, 64]", _shape_param_38, _shape_param_39, primals_77: "f32[768, 768]", _shape_param_40, _shape_param_41, primals_76: "f32[768, 768]", bmm_104: "f32[48, 512, 64]", _shape_param_42, tangents_9: "f32[4, 12, 512, 64]", bmm_106: "f32[48, 64, 512]", _shape_param_43, tangents_10: "f32[4, 12, 512, 64]", _shape_param_44, _shape_param_45, primals_64: "f32[768, 768]", _shape_param_46, _shape_param_47, primals_63: "f32[768, 768]", bmm_112: "f32[48, 512, 64]", _shape_param_48, tangents_7: "f32[4, 12, 512, 64]", bmm_114: "f32[48, 64, 512]", _shape_param_49, tangents_8: "f32[4, 12, 512, 64]", _shape_param_50, _shape_param_51, primals_51: "f32[768, 768]", _shape_param_52, _shape_param_53, primals_50: "f32[768, 768]", bmm_120: "f32[48, 512, 64]", _shape_param_54, tangents_5: "f32[4, 12, 512, 64]", bmm_122: "f32[48, 64, 512]", _shape_param_55, tangents_6: "f32[4, 12, 512, 64]", _shape_param_56, _shape_param_57, primals_38: "f32[768, 768]", _shape_param_58, _shape_param_59, primals_37: "f32[768, 768]", bmm_128: "f32[48, 512, 64]", _shape_param_60, tangents_3: "f32[4, 12, 512, 64]", bmm_130: "f32[48, 64, 512]", _shape_param_61, tangents_4: "f32[4, 12, 512, 64]", _shape_param_62, _shape_param_63, primals_25: "f32[768, 768]", _shape_param_64, _shape_param_65, primals_24: "f32[768, 768]", bmm_136: "f32[48, 512, 64]", _shape_param_66, tangents_1: "f32[4, 12, 512, 64]", bmm_138: "f32[48, 64, 512]", _shape_param_67, tangents_2: "f32[4, 12, 512, 64]", _shape_param_68, _shape_param_69, primals_12: "f32[768, 768]", _shape_param_70, _shape_param_71, primals_11: "f32[768, 768]", bmm_140: "f32[48, 128, 64]", _shape_param_72, tangents_25: "f32[4, 12, 128, 64]", bmm_142: "f32[48, 64, 128]", _shape_param_73, bmm_143: "f32[48, 128, 64]", _shape_param_74, tangents_26: "f32[4, 12, 128, 64]", _shape_param_75, _shape_param_76, primals_6: "f32[768, 768]", _shape_param_77, _shape_param_78, primals_5: "f32[768, 768]", _shape_param_79, _shape_param_80, primals_4: "f32[768, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_48, _shape_param_0);  bmm_48 = _shape_param_0 = None
        add_tensor: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_23, reshape_default);  tangents_23 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_1: "f32[4, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_50, _shape_param_1);  bmm_50 = _shape_param_1 = None
        permute_default: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2]);  reshape_default_1 = None
        add_tensor_1: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_24, permute_default);  tangents_24 = permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_1: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_tensor, [0, 2, 1, 3]);  add_tensor = None
        clone_default: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_2: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None
        reshape_default_3: "f32[2048, 768]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
        permute_default_3: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_2, [1, 0]);  permute_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_4: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_tensor_1, [0, 2, 1, 3]);  add_tensor_1 = None
        clone_default_1: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_4, memory_format = torch.contiguous_format);  permute_default_4 = None
        reshape_default_4: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_4);  clone_default_1 = _shape_param_4 = None
        reshape_default_5: "f32[2048, 768]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_5);  reshape_default_4 = _shape_param_5 = None
        permute_default_5: "f32[768, 768]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
        permute_default_6: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_5, [1, 0]);  permute_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default_6: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_56, _shape_param_6);  bmm_56 = _shape_param_6 = None
        add_tensor_2: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_21, reshape_default_6);  tangents_21 = reshape_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_7: "f32[4, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_58, _shape_param_7);  bmm_58 = _shape_param_7 = None
        permute_default_7: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_7, [0, 1, 3, 2]);  reshape_default_7 = None
        add_tensor_3: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_22, permute_default_7);  tangents_22 = permute_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_8: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_tensor_2, [0, 2, 1, 3]);  add_tensor_2 = None
        clone_default_2: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_8, memory_format = torch.contiguous_format);  permute_default_8 = None
        reshape_default_8: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_8);  clone_default_2 = _shape_param_8 = None
        reshape_default_9: "f32[2048, 768]" = torch.ops.aten.reshape.default(reshape_default_8, _shape_param_9);  reshape_default_8 = _shape_param_9 = None
        permute_default_9: "f32[768, 768]" = torch.ops.aten.permute.default(primals_142, [1, 0]);  primals_142 = None
        permute_default_10: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_9, [1, 0]);  permute_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_11: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_tensor_3, [0, 2, 1, 3]);  add_tensor_3 = None
        clone_default_3: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_11, memory_format = torch.contiguous_format);  permute_default_11 = None
        reshape_default_10: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_default_3, _shape_param_10);  clone_default_3 = _shape_param_10 = None
        reshape_default_11: "f32[2048, 768]" = torch.ops.aten.reshape.default(reshape_default_10, _shape_param_11);  reshape_default_10 = _shape_param_11 = None
        permute_default_12: "f32[768, 768]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
        permute_default_13: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_12, [1, 0]);  permute_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default_12: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_64, _shape_param_12);  bmm_64 = _shape_param_12 = None
        add_tensor_4: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_19, reshape_default_12);  tangents_19 = reshape_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_13: "f32[4, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_66, _shape_param_13);  bmm_66 = _shape_param_13 = None
        permute_default_14: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_13, [0, 1, 3, 2]);  reshape_default_13 = None
        add_tensor_5: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_20, permute_default_14);  tangents_20 = permute_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_15: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_tensor_4, [0, 2, 1, 3]);  add_tensor_4 = None
        clone_default_4: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_15, memory_format = torch.contiguous_format);  permute_default_15 = None
        reshape_default_14: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_default_4, _shape_param_14);  clone_default_4 = _shape_param_14 = None
        reshape_default_15: "f32[2048, 768]" = torch.ops.aten.reshape.default(reshape_default_14, _shape_param_15);  reshape_default_14 = _shape_param_15 = None
        permute_default_16: "f32[768, 768]" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
        permute_default_17: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_16, [1, 0]);  permute_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_18: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_tensor_5, [0, 2, 1, 3]);  add_tensor_5 = None
        clone_default_5: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_18, memory_format = torch.contiguous_format);  permute_default_18 = None
        reshape_default_16: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_default_5, _shape_param_16);  clone_default_5 = _shape_param_16 = None
        reshape_default_17: "f32[2048, 768]" = torch.ops.aten.reshape.default(reshape_default_16, _shape_param_17);  reshape_default_16 = _shape_param_17 = None
        permute_default_19: "f32[768, 768]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
        permute_default_20: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_19, [1, 0]);  permute_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default_18: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_72, _shape_param_18);  bmm_72 = _shape_param_18 = None
        add_tensor_6: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_17, reshape_default_18);  tangents_17 = reshape_default_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_19: "f32[4, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_74, _shape_param_19);  bmm_74 = _shape_param_19 = None
        permute_default_21: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_19, [0, 1, 3, 2]);  reshape_default_19 = None
        add_tensor_7: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_18, permute_default_21);  tangents_18 = permute_default_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_22: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_tensor_6, [0, 2, 1, 3]);  add_tensor_6 = None
        clone_default_6: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_22, memory_format = torch.contiguous_format);  permute_default_22 = None
        reshape_default_20: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_default_6, _shape_param_20);  clone_default_6 = _shape_param_20 = None
        reshape_default_21: "f32[2048, 768]" = torch.ops.aten.reshape.default(reshape_default_20, _shape_param_21);  reshape_default_20 = _shape_param_21 = None
        permute_default_23: "f32[768, 768]" = torch.ops.aten.permute.default(primals_116, [1, 0]);  primals_116 = None
        permute_default_24: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_23, [1, 0]);  permute_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_25: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_tensor_7, [0, 2, 1, 3]);  add_tensor_7 = None
        clone_default_7: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_25, memory_format = torch.contiguous_format);  permute_default_25 = None
        reshape_default_22: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_default_7, _shape_param_22);  clone_default_7 = _shape_param_22 = None
        reshape_default_23: "f32[2048, 768]" = torch.ops.aten.reshape.default(reshape_default_22, _shape_param_23);  reshape_default_22 = _shape_param_23 = None
        permute_default_26: "f32[768, 768]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
        permute_default_27: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_26, [1, 0]);  permute_default_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default_24: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_80, _shape_param_24);  bmm_80 = _shape_param_24 = None
        add_tensor_8: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_15, reshape_default_24);  tangents_15 = reshape_default_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_25: "f32[4, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_82, _shape_param_25);  bmm_82 = _shape_param_25 = None
        permute_default_28: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_25, [0, 1, 3, 2]);  reshape_default_25 = None
        add_tensor_9: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_16, permute_default_28);  tangents_16 = permute_default_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_29: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_tensor_8, [0, 2, 1, 3]);  add_tensor_8 = None
        clone_default_8: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_29, memory_format = torch.contiguous_format);  permute_default_29 = None
        reshape_default_26: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_default_8, _shape_param_26);  clone_default_8 = _shape_param_26 = None
        reshape_default_27: "f32[2048, 768]" = torch.ops.aten.reshape.default(reshape_default_26, _shape_param_27);  reshape_default_26 = _shape_param_27 = None
        permute_default_30: "f32[768, 768]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
        permute_default_31: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_30, [1, 0]);  permute_default_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_32: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_tensor_9, [0, 2, 1, 3]);  add_tensor_9 = None
        clone_default_9: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_32, memory_format = torch.contiguous_format);  permute_default_32 = None
        reshape_default_28: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_default_9, _shape_param_28);  clone_default_9 = _shape_param_28 = None
        reshape_default_29: "f32[2048, 768]" = torch.ops.aten.reshape.default(reshape_default_28, _shape_param_29);  reshape_default_28 = _shape_param_29 = None
        permute_default_33: "f32[768, 768]" = torch.ops.aten.permute.default(primals_102, [1, 0]);  primals_102 = None
        permute_default_34: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_33, [1, 0]);  permute_default_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default_30: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_88, _shape_param_30);  bmm_88 = _shape_param_30 = None
        add_tensor_10: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_13, reshape_default_30);  tangents_13 = reshape_default_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_31: "f32[4, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_90, _shape_param_31);  bmm_90 = _shape_param_31 = None
        permute_default_35: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_31, [0, 1, 3, 2]);  reshape_default_31 = None
        add_tensor_11: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_14, permute_default_35);  tangents_14 = permute_default_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_36: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_tensor_10, [0, 2, 1, 3]);  add_tensor_10 = None
        clone_default_10: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_36, memory_format = torch.contiguous_format);  permute_default_36 = None
        reshape_default_32: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_default_10, _shape_param_32);  clone_default_10 = _shape_param_32 = None
        reshape_default_33: "f32[2048, 768]" = torch.ops.aten.reshape.default(reshape_default_32, _shape_param_33);  reshape_default_32 = _shape_param_33 = None
        permute_default_37: "f32[768, 768]" = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
        permute_default_38: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_37, [1, 0]);  permute_default_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_39: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_tensor_11, [0, 2, 1, 3]);  add_tensor_11 = None
        clone_default_11: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_39, memory_format = torch.contiguous_format);  permute_default_39 = None
        reshape_default_34: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_default_11, _shape_param_34);  clone_default_11 = _shape_param_34 = None
        reshape_default_35: "f32[2048, 768]" = torch.ops.aten.reshape.default(reshape_default_34, _shape_param_35);  reshape_default_34 = _shape_param_35 = None
        permute_default_40: "f32[768, 768]" = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
        permute_default_41: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_40, [1, 0]);  permute_default_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default_36: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_96, _shape_param_36);  bmm_96 = _shape_param_36 = None
        add_tensor_12: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_11, reshape_default_36);  tangents_11 = reshape_default_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_37: "f32[4, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_98, _shape_param_37);  bmm_98 = _shape_param_37 = None
        permute_default_42: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_37, [0, 1, 3, 2]);  reshape_default_37 = None
        add_tensor_13: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_12, permute_default_42);  tangents_12 = permute_default_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_43: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_tensor_12, [0, 2, 1, 3]);  add_tensor_12 = None
        clone_default_12: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_43, memory_format = torch.contiguous_format);  permute_default_43 = None
        reshape_default_38: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_default_12, _shape_param_38);  clone_default_12 = _shape_param_38 = None
        reshape_default_39: "f32[2048, 768]" = torch.ops.aten.reshape.default(reshape_default_38, _shape_param_39);  reshape_default_38 = _shape_param_39 = None
        permute_default_44: "f32[768, 768]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
        permute_default_45: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_44, [1, 0]);  permute_default_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_46: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_tensor_13, [0, 2, 1, 3]);  add_tensor_13 = None
        clone_default_13: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_46, memory_format = torch.contiguous_format);  permute_default_46 = None
        reshape_default_40: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_default_13, _shape_param_40);  clone_default_13 = _shape_param_40 = None
        reshape_default_41: "f32[2048, 768]" = torch.ops.aten.reshape.default(reshape_default_40, _shape_param_41);  reshape_default_40 = _shape_param_41 = None
        permute_default_47: "f32[768, 768]" = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
        permute_default_48: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_47, [1, 0]);  permute_default_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default_42: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_104, _shape_param_42);  bmm_104 = _shape_param_42 = None
        add_tensor_14: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_9, reshape_default_42);  tangents_9 = reshape_default_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_43: "f32[4, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_106, _shape_param_43);  bmm_106 = _shape_param_43 = None
        permute_default_49: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_43, [0, 1, 3, 2]);  reshape_default_43 = None
        add_tensor_15: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_10, permute_default_49);  tangents_10 = permute_default_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_50: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_tensor_14, [0, 2, 1, 3]);  add_tensor_14 = None
        clone_default_14: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_50, memory_format = torch.contiguous_format);  permute_default_50 = None
        reshape_default_44: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_default_14, _shape_param_44);  clone_default_14 = _shape_param_44 = None
        reshape_default_45: "f32[2048, 768]" = torch.ops.aten.reshape.default(reshape_default_44, _shape_param_45);  reshape_default_44 = _shape_param_45 = None
        permute_default_51: "f32[768, 768]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
        permute_default_52: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_51, [1, 0]);  permute_default_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_53: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_tensor_15, [0, 2, 1, 3]);  add_tensor_15 = None
        clone_default_15: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_53, memory_format = torch.contiguous_format);  permute_default_53 = None
        reshape_default_46: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_default_15, _shape_param_46);  clone_default_15 = _shape_param_46 = None
        reshape_default_47: "f32[2048, 768]" = torch.ops.aten.reshape.default(reshape_default_46, _shape_param_47);  reshape_default_46 = _shape_param_47 = None
        permute_default_54: "f32[768, 768]" = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
        permute_default_55: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_54, [1, 0]);  permute_default_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default_48: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_112, _shape_param_48);  bmm_112 = _shape_param_48 = None
        add_tensor_16: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_7, reshape_default_48);  tangents_7 = reshape_default_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_49: "f32[4, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_114, _shape_param_49);  bmm_114 = _shape_param_49 = None
        permute_default_56: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_49, [0, 1, 3, 2]);  reshape_default_49 = None
        add_tensor_17: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_8, permute_default_56);  tangents_8 = permute_default_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_57: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_tensor_16, [0, 2, 1, 3]);  add_tensor_16 = None
        clone_default_16: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_57, memory_format = torch.contiguous_format);  permute_default_57 = None
        reshape_default_50: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_default_16, _shape_param_50);  clone_default_16 = _shape_param_50 = None
        reshape_default_51: "f32[2048, 768]" = torch.ops.aten.reshape.default(reshape_default_50, _shape_param_51);  reshape_default_50 = _shape_param_51 = None
        permute_default_58: "f32[768, 768]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
        permute_default_59: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_58, [1, 0]);  permute_default_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_60: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_tensor_17, [0, 2, 1, 3]);  add_tensor_17 = None
        clone_default_17: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_60, memory_format = torch.contiguous_format);  permute_default_60 = None
        reshape_default_52: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_default_17, _shape_param_52);  clone_default_17 = _shape_param_52 = None
        reshape_default_53: "f32[2048, 768]" = torch.ops.aten.reshape.default(reshape_default_52, _shape_param_53);  reshape_default_52 = _shape_param_53 = None
        permute_default_61: "f32[768, 768]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
        permute_default_62: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_61, [1, 0]);  permute_default_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default_54: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_120, _shape_param_54);  bmm_120 = _shape_param_54 = None
        add_tensor_18: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_5, reshape_default_54);  tangents_5 = reshape_default_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_55: "f32[4, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_122, _shape_param_55);  bmm_122 = _shape_param_55 = None
        permute_default_63: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_55, [0, 1, 3, 2]);  reshape_default_55 = None
        add_tensor_19: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_6, permute_default_63);  tangents_6 = permute_default_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_64: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_tensor_18, [0, 2, 1, 3]);  add_tensor_18 = None
        clone_default_18: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_64, memory_format = torch.contiguous_format);  permute_default_64 = None
        reshape_default_56: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_default_18, _shape_param_56);  clone_default_18 = _shape_param_56 = None
        reshape_default_57: "f32[2048, 768]" = torch.ops.aten.reshape.default(reshape_default_56, _shape_param_57);  reshape_default_56 = _shape_param_57 = None
        permute_default_65: "f32[768, 768]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
        permute_default_66: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_65, [1, 0]);  permute_default_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_67: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_tensor_19, [0, 2, 1, 3]);  add_tensor_19 = None
        clone_default_19: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_67, memory_format = torch.contiguous_format);  permute_default_67 = None
        reshape_default_58: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_default_19, _shape_param_58);  clone_default_19 = _shape_param_58 = None
        reshape_default_59: "f32[2048, 768]" = torch.ops.aten.reshape.default(reshape_default_58, _shape_param_59);  reshape_default_58 = _shape_param_59 = None
        permute_default_68: "f32[768, 768]" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
        permute_default_69: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_68, [1, 0]);  permute_default_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default_60: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_128, _shape_param_60);  bmm_128 = _shape_param_60 = None
        add_tensor_20: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_3, reshape_default_60);  tangents_3 = reshape_default_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_61: "f32[4, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_130, _shape_param_61);  bmm_130 = _shape_param_61 = None
        permute_default_70: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_61, [0, 1, 3, 2]);  reshape_default_61 = None
        add_tensor_21: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_4, permute_default_70);  tangents_4 = permute_default_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_71: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_tensor_20, [0, 2, 1, 3]);  add_tensor_20 = None
        clone_default_20: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_71, memory_format = torch.contiguous_format);  permute_default_71 = None
        reshape_default_62: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_default_20, _shape_param_62);  clone_default_20 = _shape_param_62 = None
        reshape_default_63: "f32[2048, 768]" = torch.ops.aten.reshape.default(reshape_default_62, _shape_param_63);  reshape_default_62 = _shape_param_63 = None
        permute_default_72: "f32[768, 768]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
        permute_default_73: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_72, [1, 0]);  permute_default_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_74: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_tensor_21, [0, 2, 1, 3]);  add_tensor_21 = None
        clone_default_21: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_74, memory_format = torch.contiguous_format);  permute_default_74 = None
        reshape_default_64: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_default_21, _shape_param_64);  clone_default_21 = _shape_param_64 = None
        reshape_default_65: "f32[2048, 768]" = torch.ops.aten.reshape.default(reshape_default_64, _shape_param_65);  reshape_default_64 = _shape_param_65 = None
        permute_default_75: "f32[768, 768]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
        permute_default_76: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_75, [1, 0]);  permute_default_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default_66: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_136, _shape_param_66);  bmm_136 = _shape_param_66 = None
        add_tensor_22: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_1, reshape_default_66);  tangents_1 = reshape_default_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_67: "f32[4, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_138, _shape_param_67);  bmm_138 = _shape_param_67 = None
        permute_default_77: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_67, [0, 1, 3, 2]);  reshape_default_67 = None
        add_tensor_23: "f32[4, 12, 512, 64]" = torch.ops.aten.add.Tensor(tangents_2, permute_default_77);  tangents_2 = permute_default_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_78: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_tensor_22, [0, 2, 1, 3]);  add_tensor_22 = None
        clone_default_22: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_78, memory_format = torch.contiguous_format);  permute_default_78 = None
        reshape_default_68: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_default_22, _shape_param_68);  clone_default_22 = _shape_param_68 = None
        reshape_default_69: "f32[2048, 768]" = torch.ops.aten.reshape.default(reshape_default_68, _shape_param_69);  reshape_default_68 = _shape_param_69 = None
        permute_default_79: "f32[768, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        permute_default_80: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_79, [1, 0]);  permute_default_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_81: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(add_tensor_23, [0, 2, 1, 3]);  add_tensor_23 = None
        clone_default_23: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_81, memory_format = torch.contiguous_format);  permute_default_81 = None
        reshape_default_70: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_default_23, _shape_param_70);  clone_default_23 = _shape_param_70 = None
        reshape_default_71: "f32[2048, 768]" = torch.ops.aten.reshape.default(reshape_default_70, _shape_param_71);  reshape_default_70 = _shape_param_71 = None
        permute_default_82: "f32[768, 768]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_default_83: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_82, [1, 0]);  permute_default_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default_72: "f32[4, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_140, _shape_param_72);  bmm_140 = _shape_param_72 = None
        add_tensor_24: "f32[4, 12, 128, 64]" = torch.ops.aten.add.Tensor(tangents_25, reshape_default_72);  tangents_25 = reshape_default_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_73: "f32[4, 12, 64, 128]" = torch.ops.aten.reshape.default(bmm_142, _shape_param_73);  bmm_142 = _shape_param_73 = None
        reshape_default_74: "f32[4, 12, 128, 64]" = torch.ops.aten.reshape.default(bmm_143, _shape_param_74);  bmm_143 = _shape_param_74 = None
        permute_default_84: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(reshape_default_73, [0, 1, 3, 2]);  reshape_default_73 = None
        add_tensor_25: "f32[4, 12, 128, 64]" = torch.ops.aten.add.Tensor(tangents_26, permute_default_84);  tangents_26 = permute_default_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_85: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(add_tensor_24, [0, 2, 1, 3]);  add_tensor_24 = None
        clone_default_24: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_default_85, memory_format = torch.contiguous_format);  permute_default_85 = None
        reshape_default_75: "f32[4, 128, 768]" = torch.ops.aten.reshape.default(clone_default_24, _shape_param_75);  clone_default_24 = _shape_param_75 = None
        reshape_default_76: "f32[512, 768]" = torch.ops.aten.reshape.default(reshape_default_75, _shape_param_76);  reshape_default_75 = _shape_param_76 = None
        permute_default_86: "f32[768, 768]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_default_87: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_86, [1, 0]);  permute_default_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_88: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(add_tensor_25, [0, 2, 1, 3]);  add_tensor_25 = None
        clone_default_25: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_default_88, memory_format = torch.contiguous_format);  permute_default_88 = None
        reshape_default_77: "f32[4, 128, 768]" = torch.ops.aten.reshape.default(clone_default_25, _shape_param_77);  clone_default_25 = _shape_param_77 = None
        reshape_default_78: "f32[512, 768]" = torch.ops.aten.reshape.default(reshape_default_77, _shape_param_78);  reshape_default_77 = _shape_param_78 = None
        permute_default_89: "f32[768, 768]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        permute_default_90: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_89, [1, 0]);  permute_default_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_91: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(reshape_default_74, [0, 2, 1, 3]);  reshape_default_74 = None
        clone_default_26: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_default_91, memory_format = torch.contiguous_format);  permute_default_91 = None
        reshape_default_79: "f32[4, 128, 768]" = torch.ops.aten.reshape.default(clone_default_26, _shape_param_79);  clone_default_26 = _shape_param_79 = None
        reshape_default_80: "f32[512, 768]" = torch.ops.aten.reshape.default(reshape_default_79, _shape_param_80);  reshape_default_79 = _shape_param_80 = None
        permute_default_92: "f32[768, 768]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_default_93: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_92, [1, 0]);  permute_default_92 = None
        return (reshape_default_3, permute_default_3, reshape_default_5, permute_default_6, reshape_default_9, permute_default_10, reshape_default_11, permute_default_13, reshape_default_15, permute_default_17, reshape_default_17, permute_default_20, reshape_default_21, permute_default_24, reshape_default_23, permute_default_27, reshape_default_27, permute_default_31, reshape_default_29, permute_default_34, reshape_default_33, permute_default_38, reshape_default_35, permute_default_41, reshape_default_39, permute_default_45, reshape_default_41, permute_default_48, reshape_default_45, permute_default_52, reshape_default_47, permute_default_55, reshape_default_51, permute_default_59, reshape_default_53, permute_default_62, reshape_default_57, permute_default_66, reshape_default_59, permute_default_69, reshape_default_63, permute_default_73, reshape_default_65, permute_default_76, reshape_default_69, permute_default_80, reshape_default_71, permute_default_83, reshape_default_76, permute_default_87, reshape_default_78, permute_default_90, reshape_default_80, permute_default_93)


def _default_make_inputs():
    return [
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    [4, 12, 512, 64],  # _shape_param_0
    torch.randn([4, 12, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    [4, 12, 64, 512],  # _shape_param_1
    torch.randn([4, 12, 512, 64], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_2
    [2048, 768],  # _shape_param_3
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_4
    [2048, 768],  # _shape_param_5
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    [4, 12, 512, 64],  # _shape_param_6
    torch.randn([4, 12, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    [4, 12, 64, 512],  # _shape_param_7
    torch.randn([4, 12, 512, 64], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_8
    [2048, 768],  # _shape_param_9
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_10
    [2048, 768],  # _shape_param_11
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    [4, 12, 512, 64],  # _shape_param_12
    torch.randn([4, 12, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    [4, 12, 64, 512],  # _shape_param_13
    torch.randn([4, 12, 512, 64], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_14
    [2048, 768],  # _shape_param_15
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_16
    [2048, 768],  # _shape_param_17
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    [4, 12, 512, 64],  # _shape_param_18
    torch.randn([4, 12, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    [4, 12, 64, 512],  # _shape_param_19
    torch.randn([4, 12, 512, 64], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_20
    [2048, 768],  # _shape_param_21
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_22
    [2048, 768],  # _shape_param_23
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    [4, 12, 512, 64],  # _shape_param_24
    torch.randn([4, 12, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    [4, 12, 64, 512],  # _shape_param_25
    torch.randn([4, 12, 512, 64], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_26
    [2048, 768],  # _shape_param_27
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_28
    [2048, 768],  # _shape_param_29
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    [4, 12, 512, 64],  # _shape_param_30
    torch.randn([4, 12, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    [4, 12, 64, 512],  # _shape_param_31
    torch.randn([4, 12, 512, 64], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_32
    [2048, 768],  # _shape_param_33
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_34
    [2048, 768],  # _shape_param_35
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    [4, 12, 512, 64],  # _shape_param_36
    torch.randn([4, 12, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    [4, 12, 64, 512],  # _shape_param_37
    torch.randn([4, 12, 512, 64], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_38
    [2048, 768],  # _shape_param_39
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_40
    [2048, 768],  # _shape_param_41
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    [4, 12, 512, 64],  # _shape_param_42
    torch.randn([4, 12, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    [4, 12, 64, 512],  # _shape_param_43
    torch.randn([4, 12, 512, 64], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_44
    [2048, 768],  # _shape_param_45
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_46
    [2048, 768],  # _shape_param_47
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    [4, 12, 512, 64],  # _shape_param_48
    torch.randn([4, 12, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    [4, 12, 64, 512],  # _shape_param_49
    torch.randn([4, 12, 512, 64], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_50
    [2048, 768],  # _shape_param_51
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_52
    [2048, 768],  # _shape_param_53
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    [4, 12, 512, 64],  # _shape_param_54
    torch.randn([4, 12, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    [4, 12, 64, 512],  # _shape_param_55
    torch.randn([4, 12, 512, 64], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_56
    [2048, 768],  # _shape_param_57
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_58
    [2048, 768],  # _shape_param_59
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    [4, 12, 512, 64],  # _shape_param_60
    torch.randn([4, 12, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    [4, 12, 64, 512],  # _shape_param_61
    torch.randn([4, 12, 512, 64], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_62
    [2048, 768],  # _shape_param_63
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_64
    [2048, 768],  # _shape_param_65
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    [4, 12, 512, 64],  # _shape_param_66
    torch.randn([4, 12, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    [4, 12, 64, 512],  # _shape_param_67
    torch.randn([4, 12, 512, 64], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_68
    [2048, 768],  # _shape_param_69
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [4, 512, 768],  # _shape_param_70
    [2048, 768],  # _shape_param_71
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    [4, 12, 128, 64],  # _shape_param_72
    torch.randn([4, 12, 128, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 128], dtype=torch.float32, device='cuda'),
    [4, 12, 64, 128],  # _shape_param_73
    torch.randn([48, 128, 64], dtype=torch.float32, device='cuda'),
    [4, 12, 128, 64],  # _shape_param_74
    torch.randn([4, 12, 128, 64], dtype=torch.float32, device='cuda'),
    [4, 128, 768],  # _shape_param_75
    [512, 768],  # _shape_param_76
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [4, 128, 768],  # _shape_param_77
    [512, 768],  # _shape_param_78
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [4, 128, 768],  # _shape_param_79
    [512, 768],  # _shape_param_80
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
