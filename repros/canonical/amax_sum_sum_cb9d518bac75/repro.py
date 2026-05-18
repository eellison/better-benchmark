"""
Standalone repro captured via capture_hook.
Label: hf_T5ForConditionalGeneration_training
Pattern hash: cb9d518bac75
Shape hash: c383a403
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
    def forward(self, mm_96: "f32[8192, 32128]", _shape_param_0, _shape_param_1, primals_53: "i64[8, 1024]", full_default: "f32[]", relu_11: "f32[8, 1024, 2048]", view_418: "f32[64, 1024, 1024]", view_419: "f32[64, 1024, 64]", view_412: "f32[64, 1024, 64]", view_413: "f32[64, 64, 1024]", view_397: "f32[64, 1024, 1024]", view_398: "f32[64, 1024, 64]", view_391: "f32[64, 1024, 64]", view_392: "f32[64, 64, 1024]", relu_10: "f32[8, 1024, 2048]", view_372: "f32[64, 1024, 1024]", view_373: "f32[64, 1024, 64]", view_366: "f32[64, 1024, 64]", view_367: "f32[64, 64, 1024]", view_351: "f32[64, 1024, 1024]", view_352: "f32[64, 1024, 64]", view_345: "f32[64, 1024, 64]", view_346: "f32[64, 64, 1024]", relu_9: "f32[8, 1024, 2048]", view_326: "f32[64, 1024, 1024]", view_327: "f32[64, 1024, 64]", view_320: "f32[64, 1024, 64]", view_321: "f32[64, 64, 1024]", view_305: "f32[64, 1024, 1024]", view_306: "f32[64, 1024, 64]", view_299: "f32[64, 1024, 64]", view_300: "f32[64, 64, 1024]", relu_8: "f32[8, 1024, 2048]", view_280: "f32[64, 1024, 1024]", view_281: "f32[64, 1024, 64]", view_274: "f32[64, 1024, 64]", view_275: "f32[64, 64, 1024]", view_259: "f32[64, 1024, 1024]", view_260: "f32[64, 1024, 64]", view_253: "f32[64, 1024, 64]", view_254: "f32[64, 64, 1024]", relu_7: "f32[8, 1024, 2048]", view_234: "f32[64, 1024, 1024]", view_235: "f32[64, 1024, 64]", view_228: "f32[64, 1024, 64]", view_229: "f32[64, 64, 1024]", view_213: "f32[64, 1024, 1024]", view_214: "f32[64, 1024, 64]", view_207: "f32[64, 1024, 64]", view_208: "f32[64, 64, 1024]", relu_6: "f32[8, 1024, 2048]", view_188: "f32[64, 1024, 1024]", view_189: "f32[64, 1024, 64]", view_182: "f32[64, 1024, 64]", view_183: "f32[64, 64, 1024]", view_167: "f32[64, 1024, 1024]", view_168: "f32[64, 1024, 64]", view_161: "f32[64, 1024, 64]", view_162: "f32[64, 64, 1024]", relu_5: "f32[8, 1024, 2048]", view_141: "f32[64, 1024, 1024]", view_142: "f32[64, 1024, 64]", view_135: "f32[64, 1024, 64]", view_136: "f32[64, 64, 1024]", relu_4: "f32[8, 1024, 2048]", view_116: "f32[64, 1024, 1024]", view_117: "f32[64, 1024, 64]", view_110: "f32[64, 1024, 64]", view_111: "f32[64, 64, 1024]", relu_3: "f32[8, 1024, 2048]", view_91: "f32[64, 1024, 1024]", view_92: "f32[64, 1024, 64]", view_85: "f32[64, 1024, 64]", view_86: "f32[64, 64, 1024]", relu_2: "f32[8, 1024, 2048]", view_66: "f32[64, 1024, 1024]", view_67: "f32[64, 1024, 64]", view_60: "f32[64, 1024, 64]", view_61: "f32[64, 64, 1024]", relu_1: "f32[8, 1024, 2048]", view_41: "f32[64, 1024, 1024]", view_42: "f32[64, 1024, 64]", view_35: "f32[64, 1024, 64]", view_36: "f32[64, 64, 1024]", relu: "f32[8, 1024, 2048]", view_16: "f32[64, 1024, 1024]", view_17: "f32[64, 1024, 64]", view_10: "f32[64, 1024, 64]", view_11: "f32[64, 64, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1097 in forward, code: lm_logits = self.lm_head(sequence_output)
        reshape_default: "f32[8, 1024, 32128]" = torch.ops.aten.reshape.default(mm_96, _shape_param_0);  mm_96 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1104 in forward, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        reshape_default_1: "f32[8192, 32128]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        reshape_default_2: "i64[8192]" = torch.ops.aten.reshape.default(primals_53, [-1]);  primals_53 = None
        amax_default: "f32[8192, 1]" = torch.ops.aten.amax.default(reshape_default_1, [1], True)
        sub_tensor: "f32[8192, 32128]" = torch.ops.aten.sub.Tensor(reshape_default_1, amax_default);  reshape_default_1 = amax_default = None
        exp_default: "f32[8192, 32128]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[8192, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[8192, 32128]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        ne_scalar: "b8[8192]" = torch.ops.aten.ne.Scalar(reshape_default_2, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[8192]" = torch.ops.aten.where.self(ne_scalar, reshape_default_2, full_default);  reshape_default_2 = full_default = None
        unsqueeze_default: "i64[8192, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "f32[8192, 1]" = torch.ops.aten.gather.default(sub_tensor_1, 1, unsqueeze_default);  sub_tensor_1 = unsqueeze_default = None
        squeeze_dim: "f32[8192]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "f32[8192]" = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        full_default_1 = full_default
        where_self_1: "f32[8192]" = torch.ops.aten.where.self(ne_scalar, neg_default, full_default_1);  neg_default = full_default_1 = None
        sum_default: "i64[]" = torch.ops.aten.sum.default(ne_scalar);  ne_scalar = None
        convert_element_type_default: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default, torch.float32);  sum_default = None
        sum_default_1: "f32[]" = torch.ops.aten.sum.default(where_self_1);  where_self_1 = None
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(sum_default_1, convert_element_type_default);  sum_default_1 = convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_418, [0, 2, 1]);  view_418 = None
        permute_default_1: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_419, [0, 2, 1]);  view_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_2: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_412, [0, 2, 1]);  view_412 = None
        permute_default_3: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_413, [0, 2, 1]);  view_413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_4: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_397, [0, 2, 1]);  view_397 = None
        permute_default_5: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_398, [0, 2, 1]);  view_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_6: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_391, [0, 2, 1]);  view_391 = None
        permute_default_7: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_392, [0, 2, 1]);  view_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar_1: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_8: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_372, [0, 2, 1]);  view_372 = None
        permute_default_9: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_373, [0, 2, 1]);  view_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_10: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_366, [0, 2, 1]);  view_366 = None
        permute_default_11: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_367, [0, 2, 1]);  view_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_12: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_351, [0, 2, 1]);  view_351 = None
        permute_default_13: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_352, [0, 2, 1]);  view_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_14: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_345, [0, 2, 1]);  view_345 = None
        permute_default_15: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_346, [0, 2, 1]);  view_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar_2: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_16: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_326, [0, 2, 1]);  view_326 = None
        permute_default_17: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_327, [0, 2, 1]);  view_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_18: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_320, [0, 2, 1]);  view_320 = None
        permute_default_19: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_321, [0, 2, 1]);  view_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_20: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_305, [0, 2, 1]);  view_305 = None
        permute_default_21: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_306, [0, 2, 1]);  view_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_22: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_299, [0, 2, 1]);  view_299 = None
        permute_default_23: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_300, [0, 2, 1]);  view_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar_3: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_24: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_280, [0, 2, 1]);  view_280 = None
        permute_default_25: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_281, [0, 2, 1]);  view_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_26: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_274, [0, 2, 1]);  view_274 = None
        permute_default_27: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_275, [0, 2, 1]);  view_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_28: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_259, [0, 2, 1]);  view_259 = None
        permute_default_29: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_260, [0, 2, 1]);  view_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_30: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_253, [0, 2, 1]);  view_253 = None
        permute_default_31: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar_4: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_32: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_234, [0, 2, 1]);  view_234 = None
        permute_default_33: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_235, [0, 2, 1]);  view_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_34: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_228, [0, 2, 1]);  view_228 = None
        permute_default_35: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_229, [0, 2, 1]);  view_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_36: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_213, [0, 2, 1]);  view_213 = None
        permute_default_37: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_214, [0, 2, 1]);  view_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_38: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_207, [0, 2, 1]);  view_207 = None
        permute_default_39: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1]);  view_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar_5: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_40: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
        permute_default_41: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_42: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_182, [0, 2, 1]);  view_182 = None
        permute_default_43: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_183, [0, 2, 1]);  view_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_44: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None
        permute_default_45: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_168, [0, 2, 1]);  view_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_46: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_161, [0, 2, 1]);  view_161 = None
        permute_default_47: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1]);  view_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar_6: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_48: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None
        permute_default_49: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_142, [0, 2, 1]);  view_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_50: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_135, [0, 2, 1]);  view_135 = None
        permute_default_51: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1]);  view_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar_7: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_52: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_116, [0, 2, 1]);  view_116 = None
        permute_default_53: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_117, [0, 2, 1]);  view_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_54: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_110, [0, 2, 1]);  view_110 = None
        permute_default_55: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_111, [0, 2, 1]);  view_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar_8: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_56: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_91, [0, 2, 1]);  view_91 = None
        permute_default_57: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_92, [0, 2, 1]);  view_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_58: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_85, [0, 2, 1]);  view_85 = None
        permute_default_59: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_86, [0, 2, 1]);  view_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar_9: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_60: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_66, [0, 2, 1]);  view_66 = None
        permute_default_61: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_67, [0, 2, 1]);  view_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_62: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_60, [0, 2, 1]);  view_60 = None
        permute_default_63: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_61, [0, 2, 1]);  view_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar_10: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_64: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_41, [0, 2, 1]);  view_41 = None
        permute_default_65: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_42, [0, 2, 1]);  view_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_66: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None
        permute_default_67: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_36, [0, 2, 1]);  view_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        le_scalar_11: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        permute_default_68: "f32[64, 1024, 1024]" = torch.ops.aten.permute.default(view_16, [0, 2, 1]);  view_16 = None
        permute_default_69: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_17, [0, 2, 1]);  view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        permute_default_70: "f32[64, 64, 1024]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
        permute_default_71: "f32[64, 1024, 64]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
        return (div_tensor, le_scalar, permute_default, permute_default_1, permute_default_2, permute_default_3, permute_default_4, permute_default_5, permute_default_6, permute_default_7, le_scalar_1, permute_default_8, permute_default_9, permute_default_10, permute_default_11, permute_default_12, permute_default_13, permute_default_14, permute_default_15, le_scalar_2, permute_default_16, permute_default_17, permute_default_18, permute_default_19, permute_default_20, permute_default_21, permute_default_22, permute_default_23, le_scalar_3, permute_default_24, permute_default_25, permute_default_26, permute_default_27, permute_default_28, permute_default_29, permute_default_30, permute_default_31, le_scalar_4, permute_default_32, permute_default_33, permute_default_34, permute_default_35, permute_default_36, permute_default_37, permute_default_38, permute_default_39, le_scalar_5, permute_default_40, permute_default_41, permute_default_42, permute_default_43, permute_default_44, permute_default_45, permute_default_46, permute_default_47, le_scalar_6, permute_default_48, permute_default_49, permute_default_50, permute_default_51, le_scalar_7, permute_default_52, permute_default_53, permute_default_54, permute_default_55, le_scalar_8, permute_default_56, permute_default_57, permute_default_58, permute_default_59, le_scalar_9, permute_default_60, permute_default_61, permute_default_62, permute_default_63, le_scalar_10, permute_default_64, permute_default_65, permute_default_66, permute_default_67, le_scalar_11, permute_default_68, permute_default_69, permute_default_70, permute_default_71)


def _default_make_inputs():
    return [
    torch.randn([8192, 32128], dtype=torch.float32, device='cuda'),
    [8, 1024, 32128],  # _shape_param_0
    [-1, 32128],  # _shape_param_1
    torch.randint(0, 2, [8, 1024], dtype=torch.int64, device='cuda'),
    torch.tensor(1),  # full_default_1 (unknown shape)
    torch.randn([8, 1024, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64, 1024], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
