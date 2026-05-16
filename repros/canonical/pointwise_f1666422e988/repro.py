"""
Standalone repro captured via capture_hook.
Label: hf_T5ForConditionalGeneration_training
Pattern hash: f1666422e988
Shape hash: 8530cebd
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_36: "f32[64, 1024, 64]", _shape_param_0, bmm_38: "f32[64, 64, 1024]", _shape_param_1, _shape_param_2, _shape_param_3, primals_128: "f32[512, 512]", _shape_param_4, _shape_param_5, primals_127: "f32[512, 512]", bmm_44: "f32[64, 1024, 64]", _shape_param_6, bmm_46: "f32[64, 64, 1024]", _shape_param_7, _shape_param_8, _shape_param_9, primals_115: "f32[512, 512]", _shape_param_10, _shape_param_11, primals_114: "f32[512, 512]", bmm_52: "f32[64, 1024, 64]", _shape_param_12, bmm_54: "f32[64, 64, 1024]", _shape_param_13, _shape_param_14, _shape_param_15, primals_102: "f32[512, 512]", _shape_param_16, _shape_param_17, primals_101: "f32[512, 512]", bmm_60: "f32[64, 1024, 64]", _shape_param_18, bmm_62: "f32[64, 64, 1024]", _shape_param_19, _shape_param_20, _shape_param_21, primals_89: "f32[512, 512]", _shape_param_22, _shape_param_23, primals_88: "f32[512, 512]", bmm_68: "f32[64, 1024, 64]", _shape_param_24, bmm_70: "f32[64, 64, 1024]", _shape_param_25, _shape_param_26, _shape_param_27, primals_76: "f32[512, 512]", _shape_param_28, _shape_param_29, primals_75: "f32[512, 512]", bmm_76: "f32[64, 1024, 64]", _shape_param_30, bmm_78: "f32[64, 64, 1024]", _shape_param_31, _shape_param_32, _shape_param_33, primals_63: "f32[512, 512]", _shape_param_34, _shape_param_35, primals_62: "f32[512, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_36, _shape_param_0);  bmm_36 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_1: "f32[8, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_38, _shape_param_1);  bmm_38 = _shape_param_1 = None
        permute_default: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_1: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_2: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None
        reshape_default_3: "f32[8192, 512]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_2: "f32[512, 512]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
        permute_default_3: "f32[512, 512]" = torch.ops.aten.permute.default(permute_default_2, [1, 0]);  permute_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_4: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        reshape_default_4: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(permute_default_4, _shape_param_4);  permute_default_4 = _shape_param_4 = None
        clone_default_1: "f32[8, 1024, 512]" = torch.ops.aten.clone.default(reshape_default_4, memory_format = torch.contiguous_format);  reshape_default_4 = None
        reshape_default_5: "f32[8192, 512]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_5);  clone_default_1 = _shape_param_5 = None
        permute_default_5: "f32[512, 512]" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
        permute_default_6: "f32[512, 512]" = torch.ops.aten.permute.default(permute_default_5, [1, 0]);  permute_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default_6: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_44, _shape_param_6);  bmm_44 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_7: "f32[8, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_46, _shape_param_7);  bmm_46 = _shape_param_7 = None
        permute_default_7: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_7, [0, 1, 3, 2]);  reshape_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_8: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default_6, [0, 2, 1, 3]);  reshape_default_6 = None
        clone_default_2: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default_8, memory_format = torch.contiguous_format);  permute_default_8 = None
        reshape_default_8: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_8);  clone_default_2 = _shape_param_8 = None
        reshape_default_9: "f32[8192, 512]" = torch.ops.aten.reshape.default(reshape_default_8, _shape_param_9);  reshape_default_8 = _shape_param_9 = None
        permute_default_9: "f32[512, 512]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
        permute_default_10: "f32[512, 512]" = torch.ops.aten.permute.default(permute_default_9, [1, 0]);  permute_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_11: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_default_7, [0, 2, 1, 3]);  permute_default_7 = None
        reshape_default_10: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(permute_default_11, _shape_param_10);  permute_default_11 = _shape_param_10 = None
        clone_default_3: "f32[8, 1024, 512]" = torch.ops.aten.clone.default(reshape_default_10, memory_format = torch.contiguous_format);  reshape_default_10 = None
        reshape_default_11: "f32[8192, 512]" = torch.ops.aten.reshape.default(clone_default_3, _shape_param_11);  clone_default_3 = _shape_param_11 = None
        permute_default_12: "f32[512, 512]" = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
        permute_default_13: "f32[512, 512]" = torch.ops.aten.permute.default(permute_default_12, [1, 0]);  permute_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default_12: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_52, _shape_param_12);  bmm_52 = _shape_param_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_13: "f32[8, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_54, _shape_param_13);  bmm_54 = _shape_param_13 = None
        permute_default_14: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_13, [0, 1, 3, 2]);  reshape_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_15: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default_12, [0, 2, 1, 3]);  reshape_default_12 = None
        clone_default_4: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default_15, memory_format = torch.contiguous_format);  permute_default_15 = None
        reshape_default_14: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_default_4, _shape_param_14);  clone_default_4 = _shape_param_14 = None
        reshape_default_15: "f32[8192, 512]" = torch.ops.aten.reshape.default(reshape_default_14, _shape_param_15);  reshape_default_14 = _shape_param_15 = None
        permute_default_16: "f32[512, 512]" = torch.ops.aten.permute.default(primals_102, [1, 0]);  primals_102 = None
        permute_default_17: "f32[512, 512]" = torch.ops.aten.permute.default(permute_default_16, [1, 0]);  permute_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_18: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_default_14, [0, 2, 1, 3]);  permute_default_14 = None
        reshape_default_16: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(permute_default_18, _shape_param_16);  permute_default_18 = _shape_param_16 = None
        clone_default_5: "f32[8, 1024, 512]" = torch.ops.aten.clone.default(reshape_default_16, memory_format = torch.contiguous_format);  reshape_default_16 = None
        reshape_default_17: "f32[8192, 512]" = torch.ops.aten.reshape.default(clone_default_5, _shape_param_17);  clone_default_5 = _shape_param_17 = None
        permute_default_19: "f32[512, 512]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
        permute_default_20: "f32[512, 512]" = torch.ops.aten.permute.default(permute_default_19, [1, 0]);  permute_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default_18: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_60, _shape_param_18);  bmm_60 = _shape_param_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_19: "f32[8, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_62, _shape_param_19);  bmm_62 = _shape_param_19 = None
        permute_default_21: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_19, [0, 1, 3, 2]);  reshape_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_22: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default_18, [0, 2, 1, 3]);  reshape_default_18 = None
        clone_default_6: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default_22, memory_format = torch.contiguous_format);  permute_default_22 = None
        reshape_default_20: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_default_6, _shape_param_20);  clone_default_6 = _shape_param_20 = None
        reshape_default_21: "f32[8192, 512]" = torch.ops.aten.reshape.default(reshape_default_20, _shape_param_21);  reshape_default_20 = _shape_param_21 = None
        permute_default_23: "f32[512, 512]" = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
        permute_default_24: "f32[512, 512]" = torch.ops.aten.permute.default(permute_default_23, [1, 0]);  permute_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_25: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_default_21, [0, 2, 1, 3]);  permute_default_21 = None
        reshape_default_22: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(permute_default_25, _shape_param_22);  permute_default_25 = _shape_param_22 = None
        clone_default_7: "f32[8, 1024, 512]" = torch.ops.aten.clone.default(reshape_default_22, memory_format = torch.contiguous_format);  reshape_default_22 = None
        reshape_default_23: "f32[8192, 512]" = torch.ops.aten.reshape.default(clone_default_7, _shape_param_23);  clone_default_7 = _shape_param_23 = None
        permute_default_26: "f32[512, 512]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
        permute_default_27: "f32[512, 512]" = torch.ops.aten.permute.default(permute_default_26, [1, 0]);  permute_default_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default_24: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_68, _shape_param_24);  bmm_68 = _shape_param_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_25: "f32[8, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_70, _shape_param_25);  bmm_70 = _shape_param_25 = None
        permute_default_28: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_25, [0, 1, 3, 2]);  reshape_default_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_29: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default_24, [0, 2, 1, 3]);  reshape_default_24 = None
        clone_default_8: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default_29, memory_format = torch.contiguous_format);  permute_default_29 = None
        reshape_default_26: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_default_8, _shape_param_26);  clone_default_8 = _shape_param_26 = None
        reshape_default_27: "f32[8192, 512]" = torch.ops.aten.reshape.default(reshape_default_26, _shape_param_27);  reshape_default_26 = _shape_param_27 = None
        permute_default_30: "f32[512, 512]" = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
        permute_default_31: "f32[512, 512]" = torch.ops.aten.permute.default(permute_default_30, [1, 0]);  permute_default_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_32: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_default_28, [0, 2, 1, 3]);  permute_default_28 = None
        reshape_default_28: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(permute_default_32, _shape_param_28);  permute_default_32 = _shape_param_28 = None
        clone_default_9: "f32[8, 1024, 512]" = torch.ops.aten.clone.default(reshape_default_28, memory_format = torch.contiguous_format);  reshape_default_28 = None
        reshape_default_29: "f32[8192, 512]" = torch.ops.aten.reshape.default(clone_default_9, _shape_param_29);  clone_default_9 = _shape_param_29 = None
        permute_default_33: "f32[512, 512]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
        permute_default_34: "f32[512, 512]" = torch.ops.aten.permute.default(permute_default_33, [1, 0]);  permute_default_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default_30: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_76, _shape_param_30);  bmm_76 = _shape_param_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_31: "f32[8, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_78, _shape_param_31);  bmm_78 = _shape_param_31 = None
        permute_default_35: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_31, [0, 1, 3, 2]);  reshape_default_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_default_36: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(reshape_default_30, [0, 2, 1, 3]);  reshape_default_30 = None
        clone_default_10: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_default_36, memory_format = torch.contiguous_format);  permute_default_36 = None
        reshape_default_32: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_default_10, _shape_param_32);  clone_default_10 = _shape_param_32 = None
        reshape_default_33: "f32[8192, 512]" = torch.ops.aten.reshape.default(reshape_default_32, _shape_param_33);  reshape_default_32 = _shape_param_33 = None
        permute_default_37: "f32[512, 512]" = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
        permute_default_38: "f32[512, 512]" = torch.ops.aten.permute.default(permute_default_37, [1, 0]);  permute_default_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_39: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_default_35, [0, 2, 1, 3]);  permute_default_35 = None
        reshape_default_34: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(permute_default_39, _shape_param_34);  permute_default_39 = _shape_param_34 = None
        clone_default_11: "f32[8, 1024, 512]" = torch.ops.aten.clone.default(reshape_default_34, memory_format = torch.contiguous_format);  reshape_default_34 = None
        reshape_default_35: "f32[8192, 512]" = torch.ops.aten.reshape.default(clone_default_11, _shape_param_35);  clone_default_11 = _shape_param_35 = None
        permute_default_40: "f32[512, 512]" = torch.ops.aten.permute.default(primals_62, [1, 0]);  primals_62 = None
        permute_default_41: "f32[512, 512]" = torch.ops.aten.permute.default(permute_default_40, [1, 0]);  permute_default_40 = None
        return (reshape_default_3, permute_default_3, reshape_default_5, permute_default_6, reshape_default_9, permute_default_10, reshape_default_11, permute_default_13, reshape_default_15, permute_default_17, reshape_default_17, permute_default_20, reshape_default_21, permute_default_24, reshape_default_23, permute_default_27, reshape_default_27, permute_default_31, reshape_default_29, permute_default_34, reshape_default_33, permute_default_38, reshape_default_35, permute_default_41)


def _default_make_inputs():
    return [
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    [8, 8, 1024, 64],  # _shape_param_0
    torch.randn([64, 64, 1024], dtype=torch.float32, device='cuda'),
    [8, 8, 64, 1024],  # _shape_param_1
    [8, 1024, 512],  # _shape_param_2
    [8192, 512],  # _shape_param_3
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    [8, 1024, 512],  # _shape_param_4
    [8192, 512],  # _shape_param_5
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    [8, 8, 1024, 64],  # _shape_param_6
    torch.randn([64, 64, 1024], dtype=torch.float32, device='cuda'),
    [8, 8, 64, 1024],  # _shape_param_7
    [8, 1024, 512],  # _shape_param_8
    [8192, 512],  # _shape_param_9
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    [8, 1024, 512],  # _shape_param_10
    [8192, 512],  # _shape_param_11
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    [8, 8, 1024, 64],  # _shape_param_12
    torch.randn([64, 64, 1024], dtype=torch.float32, device='cuda'),
    [8, 8, 64, 1024],  # _shape_param_13
    [8, 1024, 512],  # _shape_param_14
    [8192, 512],  # _shape_param_15
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    [8, 1024, 512],  # _shape_param_16
    [8192, 512],  # _shape_param_17
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    [8, 8, 1024, 64],  # _shape_param_18
    torch.randn([64, 64, 1024], dtype=torch.float32, device='cuda'),
    [8, 8, 64, 1024],  # _shape_param_19
    [8, 1024, 512],  # _shape_param_20
    [8192, 512],  # _shape_param_21
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    [8, 1024, 512],  # _shape_param_22
    [8192, 512],  # _shape_param_23
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    [8, 8, 1024, 64],  # _shape_param_24
    torch.randn([64, 64, 1024], dtype=torch.float32, device='cuda'),
    [8, 8, 64, 1024],  # _shape_param_25
    [8, 1024, 512],  # _shape_param_26
    [8192, 512],  # _shape_param_27
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    [8, 1024, 512],  # _shape_param_28
    [8192, 512],  # _shape_param_29
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    [8, 8, 1024, 64],  # _shape_param_30
    torch.randn([64, 64, 1024], dtype=torch.float32, device='cuda'),
    [8, 8, 64, 1024],  # _shape_param_31
    [8, 1024, 512],  # _shape_param_32
    [8192, 512],  # _shape_param_33
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    [8, 1024, 512],  # _shape_param_34
    [8192, 512],  # _shape_param_35
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
