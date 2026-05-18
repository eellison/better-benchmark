"""
Standalone repro captured via capture_hook.
Label: hf_MegatronBertForCausalLM_training
Pattern hash: 135732ae10ea
Shape hash: 7e940039
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
    def forward(self, addmm_145: "f32[4096, 29056]", _shape_param_0, primals_1: "i64[8, 512]", _shape_param_1, rsqrt_48: "f32[8, 512, 1]", rsqrt_47: "f32[8, 512, 1]", view_518: "f32[128, 512, 512]", view_519: "f32[128, 512, 64]", view_515: "f32[128, 512, 64]", view_516: "f32[128, 64, 512]", rsqrt_46: "f32[8, 512, 1]", rsqrt_45: "f32[8, 512, 1]", view_496: "f32[128, 512, 512]", view_497: "f32[128, 512, 64]", view_493: "f32[128, 512, 64]", view_494: "f32[128, 64, 512]", rsqrt_44: "f32[8, 512, 1]", rsqrt_43: "f32[8, 512, 1]", view_474: "f32[128, 512, 512]", view_475: "f32[128, 512, 64]", view_471: "f32[128, 512, 64]", view_472: "f32[128, 64, 512]", rsqrt_42: "f32[8, 512, 1]", rsqrt_41: "f32[8, 512, 1]", view_452: "f32[128, 512, 512]", view_453: "f32[128, 512, 64]", view_449: "f32[128, 512, 64]", view_450: "f32[128, 64, 512]", rsqrt_40: "f32[8, 512, 1]", rsqrt_39: "f32[8, 512, 1]", view_430: "f32[128, 512, 512]", view_431: "f32[128, 512, 64]", view_427: "f32[128, 512, 64]", view_428: "f32[128, 64, 512]", rsqrt_38: "f32[8, 512, 1]", rsqrt_37: "f32[8, 512, 1]", view_408: "f32[128, 512, 512]", view_409: "f32[128, 512, 64]", view_405: "f32[128, 512, 64]", view_406: "f32[128, 64, 512]", rsqrt_36: "f32[8, 512, 1]", rsqrt_35: "f32[8, 512, 1]", view_386: "f32[128, 512, 512]", view_387: "f32[128, 512, 64]", view_383: "f32[128, 512, 64]", view_384: "f32[128, 64, 512]", rsqrt_34: "f32[8, 512, 1]", rsqrt_33: "f32[8, 512, 1]", view_364: "f32[128, 512, 512]", view_365: "f32[128, 512, 64]", view_361: "f32[128, 512, 64]", view_362: "f32[128, 64, 512]", rsqrt_32: "f32[8, 512, 1]", rsqrt_31: "f32[8, 512, 1]", view_342: "f32[128, 512, 512]", view_343: "f32[128, 512, 64]", view_339: "f32[128, 512, 64]", view_340: "f32[128, 64, 512]", rsqrt_30: "f32[8, 512, 1]", rsqrt_29: "f32[8, 512, 1]", view_320: "f32[128, 512, 512]", view_321: "f32[128, 512, 64]", view_317: "f32[128, 512, 64]", view_318: "f32[128, 64, 512]", rsqrt_28: "f32[8, 512, 1]", rsqrt_27: "f32[8, 512, 1]", view_298: "f32[128, 512, 512]", view_299: "f32[128, 512, 64]", view_295: "f32[128, 512, 64]", view_296: "f32[128, 64, 512]", rsqrt_26: "f32[8, 512, 1]", rsqrt_25: "f32[8, 512, 1]", view_276: "f32[128, 512, 512]", view_277: "f32[128, 512, 64]", view_273: "f32[128, 512, 64]", view_274: "f32[128, 64, 512]", rsqrt_24: "f32[8, 512, 1]", rsqrt_23: "f32[8, 512, 1]", view_254: "f32[128, 512, 512]", view_255: "f32[128, 512, 64]", view_251: "f32[128, 512, 64]", view_252: "f32[128, 64, 512]", rsqrt_22: "f32[8, 512, 1]", rsqrt_21: "f32[8, 512, 1]", view_232: "f32[128, 512, 512]", view_233: "f32[128, 512, 64]", view_229: "f32[128, 512, 64]", view_230: "f32[128, 64, 512]", rsqrt_20: "f32[8, 512, 1]", rsqrt_19: "f32[8, 512, 1]", view_210: "f32[128, 512, 512]", view_211: "f32[128, 512, 64]", view_207: "f32[128, 512, 64]", view_208: "f32[128, 64, 512]", rsqrt_18: "f32[8, 512, 1]", rsqrt_17: "f32[8, 512, 1]", view_188: "f32[128, 512, 512]", view_189: "f32[128, 512, 64]", view_185: "f32[128, 512, 64]", view_186: "f32[128, 64, 512]", rsqrt_16: "f32[8, 512, 1]", rsqrt_15: "f32[8, 512, 1]", view_166: "f32[128, 512, 512]", view_167: "f32[128, 512, 64]", view_163: "f32[128, 512, 64]", view_164: "f32[128, 64, 512]", rsqrt_14: "f32[8, 512, 1]", rsqrt_13: "f32[8, 512, 1]", view_144: "f32[128, 512, 512]", view_145: "f32[128, 512, 64]", view_141: "f32[128, 512, 64]", view_142: "f32[128, 64, 512]", rsqrt_12: "f32[8, 512, 1]", rsqrt_11: "f32[8, 512, 1]", view_122: "f32[128, 512, 512]", view_123: "f32[128, 512, 64]", view_119: "f32[128, 512, 64]", view_120: "f32[128, 64, 512]", rsqrt_10: "f32[8, 512, 1]", rsqrt_9: "f32[8, 512, 1]", view_100: "f32[128, 512, 512]", view_101: "f32[128, 512, 64]", view_97: "f32[128, 512, 64]", view_98: "f32[128, 64, 512]", rsqrt_8: "f32[8, 512, 1]", rsqrt_7: "f32[8, 512, 1]", view_78: "f32[128, 512, 512]", view_79: "f32[128, 512, 64]", view_75: "f32[128, 512, 64]", view_76: "f32[128, 64, 512]", rsqrt_6: "f32[8, 512, 1]", rsqrt_5: "f32[8, 512, 1]", view_56: "f32[128, 512, 512]", view_57: "f32[128, 512, 64]", view_53: "f32[128, 512, 64]", view_54: "f32[128, 64, 512]", rsqrt_4: "f32[8, 512, 1]", rsqrt_3: "f32[8, 512, 1]", view_34: "f32[128, 512, 512]", view_35: "f32[128, 512, 64]", view_31: "f32[128, 512, 64]", view_32: "f32[128, 64, 512]", rsqrt_2: "f32[8, 512, 1]", rsqrt_1: "f32[8, 512, 1]", view_12: "f32[128, 512, 512]", view_13: "f32[128, 512, 64]", view_9: "f32[128, 512, 64]", view_10: "f32[128, 64, 512]", rsqrt: "f32[8, 512, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:465 in forward, code: hidden_states = self.decoder(hidden_states)
        reshape_default: "f32[8, 512, 29056]" = torch.ops.aten.reshape.default(addmm_145, _shape_param_0);  addmm_145 = _shape_param_0 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default: "i64[8, 513]" = torch.ops.aten.constant_pad_nd.default(primals_1, [0, 1], -100.0);  primals_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:61 in ForCausalLMLoss, code: shift_labels = labels[..., 1:].contiguous()
        slice_tensor: "i64[8, 512]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default, 1, 1, 9223372036854775807);  constant_pad_nd_default = None
        clone_default: "i64[8, 512]" = torch.ops.aten.clone.default(slice_tensor, memory_format = torch.contiguous_format);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        reshape_default_1: "f32[4096, 29056]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:65 in ForCausalLMLoss, code: shift_labels = shift_labels.view(-1)
        reshape_default_2: "i64[4096]" = torch.ops.aten.reshape.default(clone_default, [-1]);  clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        amax_default: "f32[4096, 1]" = torch.ops.aten.amax.default(reshape_default_1, [1], True)
        sub_tensor: "f32[4096, 29056]" = torch.ops.aten.sub.Tensor(reshape_default_1, amax_default);  reshape_default_1 = amax_default = None
        exp_default: "f32[4096, 29056]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[4096, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[4096, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[4096, 29056]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        ne_scalar: "b8[4096]" = torch.ops.aten.ne.Scalar(reshape_default_2, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[4096]" = torch.ops.aten.where.self(ne_scalar, reshape_default_2, full_default);  reshape_default_2 = full_default = None
        unsqueeze_default: "i64[4096, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "f32[4096, 1]" = torch.ops.aten.gather.default(sub_tensor_1, 1, unsqueeze_default);  sub_tensor_1 = unsqueeze_default = None
        squeeze_dim: "f32[4096]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "f32[4096]" = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[4096]" = torch.ops.aten.where.self(ne_scalar, neg_default, full_default_1);  neg_default = full_default_1 = None
        sum_default: "i64[]" = torch.ops.aten.sum.default(ne_scalar);  ne_scalar = None
        convert_element_type_default: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default, torch.float32);  sum_default = None
        sum_default_1: "f32[]" = torch.ops.aten.sum.default(where_self_1);  where_self_1 = None
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(sum_default_1, convert_element_type_default);  sum_default_1 = convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:392 in forward, code: hidden_states = self.ln(hidden_states)
        div_tensor_1: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_48, 1024);  rsqrt_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_2: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 1024);  rsqrt_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default: "f32[128, 512, 512]" = torch.ops.aten.permute.default(view_518, [0, 2, 1]);  view_518 = None
        permute_default_1: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_519, [0, 2, 1]);  view_519 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:171 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_2: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_515, [0, 2, 1]);  view_515 = None
        permute_default_3: "f32[128, 512, 64]" = torch.ops.aten.permute.default(view_516, [0, 2, 1]);  view_516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_3: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 1024);  rsqrt_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_4: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 1024);  rsqrt_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default_4: "f32[128, 512, 512]" = torch.ops.aten.permute.default(view_496, [0, 2, 1]);  view_496 = None
        permute_default_5: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_497, [0, 2, 1]);  view_497 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:171 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_6: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_493, [0, 2, 1]);  view_493 = None
        permute_default_7: "f32[128, 512, 64]" = torch.ops.aten.permute.default(view_494, [0, 2, 1]);  view_494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_5: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 1024);  rsqrt_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_6: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 1024);  rsqrt_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default_8: "f32[128, 512, 512]" = torch.ops.aten.permute.default(view_474, [0, 2, 1]);  view_474 = None
        permute_default_9: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_475, [0, 2, 1]);  view_475 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:171 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_10: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_471, [0, 2, 1]);  view_471 = None
        permute_default_11: "f32[128, 512, 64]" = torch.ops.aten.permute.default(view_472, [0, 2, 1]);  view_472 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_7: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 1024);  rsqrt_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_8: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 1024);  rsqrt_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default_12: "f32[128, 512, 512]" = torch.ops.aten.permute.default(view_452, [0, 2, 1]);  view_452 = None
        permute_default_13: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_453, [0, 2, 1]);  view_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:171 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_14: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_449, [0, 2, 1]);  view_449 = None
        permute_default_15: "f32[128, 512, 64]" = torch.ops.aten.permute.default(view_450, [0, 2, 1]);  view_450 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_9: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 1024);  rsqrt_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_10: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 1024);  rsqrt_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default_16: "f32[128, 512, 512]" = torch.ops.aten.permute.default(view_430, [0, 2, 1]);  view_430 = None
        permute_default_17: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_431, [0, 2, 1]);  view_431 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:171 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_18: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_427, [0, 2, 1]);  view_427 = None
        permute_default_19: "f32[128, 512, 64]" = torch.ops.aten.permute.default(view_428, [0, 2, 1]);  view_428 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_11: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 1024);  rsqrt_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_12: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 1024);  rsqrt_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default_20: "f32[128, 512, 512]" = torch.ops.aten.permute.default(view_408, [0, 2, 1]);  view_408 = None
        permute_default_21: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_409, [0, 2, 1]);  view_409 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:171 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_22: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_405, [0, 2, 1]);  view_405 = None
        permute_default_23: "f32[128, 512, 64]" = torch.ops.aten.permute.default(view_406, [0, 2, 1]);  view_406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_13: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 1024);  rsqrt_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_14: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 1024);  rsqrt_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default_24: "f32[128, 512, 512]" = torch.ops.aten.permute.default(view_386, [0, 2, 1]);  view_386 = None
        permute_default_25: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_387, [0, 2, 1]);  view_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:171 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_26: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_383, [0, 2, 1]);  view_383 = None
        permute_default_27: "f32[128, 512, 64]" = torch.ops.aten.permute.default(view_384, [0, 2, 1]);  view_384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_15: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 1024);  rsqrt_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_16: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 1024);  rsqrt_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default_28: "f32[128, 512, 512]" = torch.ops.aten.permute.default(view_364, [0, 2, 1]);  view_364 = None
        permute_default_29: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_365, [0, 2, 1]);  view_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:171 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_30: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_361, [0, 2, 1]);  view_361 = None
        permute_default_31: "f32[128, 512, 64]" = torch.ops.aten.permute.default(view_362, [0, 2, 1]);  view_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_17: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 1024);  rsqrt_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_18: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 1024);  rsqrt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default_32: "f32[128, 512, 512]" = torch.ops.aten.permute.default(view_342, [0, 2, 1]);  view_342 = None
        permute_default_33: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_343, [0, 2, 1]);  view_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:171 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_34: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_339, [0, 2, 1]);  view_339 = None
        permute_default_35: "f32[128, 512, 64]" = torch.ops.aten.permute.default(view_340, [0, 2, 1]);  view_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_19: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 1024);  rsqrt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_20: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 1024);  rsqrt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default_36: "f32[128, 512, 512]" = torch.ops.aten.permute.default(view_320, [0, 2, 1]);  view_320 = None
        permute_default_37: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_321, [0, 2, 1]);  view_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:171 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_38: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_317, [0, 2, 1]);  view_317 = None
        permute_default_39: "f32[128, 512, 64]" = torch.ops.aten.permute.default(view_318, [0, 2, 1]);  view_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_21: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 1024);  rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_22: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 1024);  rsqrt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default_40: "f32[128, 512, 512]" = torch.ops.aten.permute.default(view_298, [0, 2, 1]);  view_298 = None
        permute_default_41: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_299, [0, 2, 1]);  view_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:171 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_42: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_295, [0, 2, 1]);  view_295 = None
        permute_default_43: "f32[128, 512, 64]" = torch.ops.aten.permute.default(view_296, [0, 2, 1]);  view_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_23: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 1024);  rsqrt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_24: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 1024);  rsqrt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default_44: "f32[128, 512, 512]" = torch.ops.aten.permute.default(view_276, [0, 2, 1]);  view_276 = None
        permute_default_45: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_277, [0, 2, 1]);  view_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:171 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_46: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_273, [0, 2, 1]);  view_273 = None
        permute_default_47: "f32[128, 512, 64]" = torch.ops.aten.permute.default(view_274, [0, 2, 1]);  view_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_25: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 1024);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_26: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 1024);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default_48: "f32[128, 512, 512]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
        permute_default_49: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_255, [0, 2, 1]);  view_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:171 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_50: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_251, [0, 2, 1]);  view_251 = None
        permute_default_51: "f32[128, 512, 64]" = torch.ops.aten.permute.default(view_252, [0, 2, 1]);  view_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_27: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 1024);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_28: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 1024);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default_52: "f32[128, 512, 512]" = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
        permute_default_53: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_233, [0, 2, 1]);  view_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:171 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_54: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_229, [0, 2, 1]);  view_229 = None
        permute_default_55: "f32[128, 512, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_29: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 1024);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_30: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 1024);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default_56: "f32[128, 512, 512]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
        permute_default_57: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_211, [0, 2, 1]);  view_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:171 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_58: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_207, [0, 2, 1]);  view_207 = None
        permute_default_59: "f32[128, 512, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1]);  view_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_31: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 1024);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_32: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 1024);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default_60: "f32[128, 512, 512]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
        permute_default_61: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:171 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_62: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_185, [0, 2, 1]);  view_185 = None
        permute_default_63: "f32[128, 512, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1]);  view_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_33: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 1024);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_34: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 1024);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default_64: "f32[128, 512, 512]" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
        permute_default_65: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:171 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_66: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_163, [0, 2, 1]);  view_163 = None
        permute_default_67: "f32[128, 512, 64]" = torch.ops.aten.permute.default(view_164, [0, 2, 1]);  view_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_35: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 1024);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_36: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 1024);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default_68: "f32[128, 512, 512]" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
        permute_default_69: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:171 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_70: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None
        permute_default_71: "f32[128, 512, 64]" = torch.ops.aten.permute.default(view_142, [0, 2, 1]);  view_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_37: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 1024);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_38: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 1024);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default_72: "f32[128, 512, 512]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
        permute_default_73: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:171 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_74: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_119, [0, 2, 1]);  view_119 = None
        permute_default_75: "f32[128, 512, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1]);  view_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_39: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 1024);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_40: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 1024);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default_76: "f32[128, 512, 512]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
        permute_default_77: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_101, [0, 2, 1]);  view_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:171 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_78: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
        permute_default_79: "f32[128, 512, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_41: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 1024);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_42: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 1024);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default_80: "f32[128, 512, 512]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
        permute_default_81: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:171 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_82: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_75, [0, 2, 1]);  view_75 = None
        permute_default_83: "f32[128, 512, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1]);  view_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_43: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 1024);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_44: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 1024);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default_84: "f32[128, 512, 512]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
        permute_default_85: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:171 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_86: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
        permute_default_87: "f32[128, 512, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1]);  view_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_45: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 1024);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_46: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 1024);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default_88: "f32[128, 512, 512]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
        permute_default_89: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:171 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_90: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
        permute_default_91: "f32[128, 512, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_47: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 1024);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_48: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 1024);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default_92: "f32[128, 512, 512]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        permute_default_93: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:171 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_94: "f32[128, 64, 512]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_default_95: "f32[128, 512, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_49: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None
        return (div_tensor, div_tensor_1, div_tensor_2, permute_default, permute_default_1, permute_default_2, permute_default_3, div_tensor_3, div_tensor_4, permute_default_4, permute_default_5, permute_default_6, permute_default_7, div_tensor_5, div_tensor_6, permute_default_8, permute_default_9, permute_default_10, permute_default_11, div_tensor_7, div_tensor_8, permute_default_12, permute_default_13, permute_default_14, permute_default_15, div_tensor_9, div_tensor_10, permute_default_16, permute_default_17, permute_default_18, permute_default_19, div_tensor_11, div_tensor_12, permute_default_20, permute_default_21, permute_default_22, permute_default_23, div_tensor_13, div_tensor_14, permute_default_24, permute_default_25, permute_default_26, permute_default_27, div_tensor_15, div_tensor_16, permute_default_28, permute_default_29, permute_default_30, permute_default_31, div_tensor_17, div_tensor_18, permute_default_32, permute_default_33, permute_default_34, permute_default_35, div_tensor_19, div_tensor_20, permute_default_36, permute_default_37, permute_default_38, permute_default_39, div_tensor_21, div_tensor_22, permute_default_40, permute_default_41, permute_default_42, permute_default_43, div_tensor_23, div_tensor_24, permute_default_44, permute_default_45, permute_default_46, permute_default_47, div_tensor_25, div_tensor_26, permute_default_48, permute_default_49, permute_default_50, permute_default_51, div_tensor_27, div_tensor_28, permute_default_52, permute_default_53, permute_default_54, permute_default_55, div_tensor_29, div_tensor_30, permute_default_56, permute_default_57, permute_default_58, permute_default_59, div_tensor_31, div_tensor_32, permute_default_60, permute_default_61, permute_default_62, permute_default_63, div_tensor_33, div_tensor_34, permute_default_64, permute_default_65, permute_default_66, permute_default_67, div_tensor_35, div_tensor_36, permute_default_68, permute_default_69, permute_default_70, permute_default_71, div_tensor_37, div_tensor_38, permute_default_72, permute_default_73, permute_default_74, permute_default_75, div_tensor_39, div_tensor_40, permute_default_76, permute_default_77, permute_default_78, permute_default_79, div_tensor_41, div_tensor_42, permute_default_80, permute_default_81, permute_default_82, permute_default_83, div_tensor_43, div_tensor_44, permute_default_84, permute_default_85, permute_default_86, permute_default_87, div_tensor_45, div_tensor_46, permute_default_88, permute_default_89, permute_default_90, permute_default_91, div_tensor_47, div_tensor_48, permute_default_92, permute_default_93, permute_default_94, permute_default_95, div_tensor_49)


def _default_make_inputs():
    return [
    torch.randn([4096, 29056], dtype=torch.float32, device='cuda'),
    [8, 512, 29056],  # _shape_param_0
    torch.randint(0, 29056, [8, 512], dtype=torch.int64, device='cuda'),
    [-1, 29056],  # _shape_param_1
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([128, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
