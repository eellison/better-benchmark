"""
Standalone repro captured via capture_hook.
Label: hf_GPT2ForSequenceClassification_train
Pattern hash: 1f2d9fcd0ec8
Shape hash: 4a275495
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([8192, 2], f32), T([8, 1024], i64, max=1024), T([8], i64, max=8), T([8], i64, max=2), T([], f32), T([8, 1024, 1], f32), T([8192, 3072], f32), T([8192, 768], f32), T([8, 1024, 1], f32), T([8192, 768], f32), T([8, 1024, 1], f32), T([8192, 3072], f32), T([8192, 768], f32), T([8, 1024, 1], f32), T([8192, 768], f32), T([8, 1024, 1], f32), T([8192, 3072], f32), T([8192, 768], f32), T([8, 1024, 1], f32), T([8192, 768], f32), T([8, 1024, 1], f32), T([8192, 3072], f32), T([8192, 768], f32), T([8, 1024, 1], f32), T([8192, 768], f32), T([8, 1024, 1], f32), T([8192, 3072], f32), T([8192, 768], f32), T([8, 1024, 1], f32), T([8192, 768], f32), T([8, 1024, 1], f32), T([8192, 3072], f32), T([8192, 768], f32), T([8, 1024, 1], f32), T([8192, 768], f32), T([8, 1024, 1], f32), T([8192, 3072], f32), T([8192, 768], f32), T([8, 1024, 1], f32), T([8192, 768], f32), T([8, 1024, 1], f32), T([8192, 3072], f32), T([8192, 768], f32), T([8, 1024, 1], f32), T([8192, 768], f32), T([8, 1024, 1], f32), T([8192, 3072], f32), T([8192, 768], f32), T([8, 1024, 1], f32), T([8192, 768], f32), T([8, 1024, 1], f32), T([8192, 3072], f32), T([8192, 768], f32), T([8, 1024, 1], f32), T([8192, 768], f32), T([8, 1024, 1], f32), T([8192, 3072], f32), T([8192, 768], f32), T([8, 1024, 1], f32), T([8192, 768], f32), T([8, 1024, 1], f32), T([8192, 3072], f32), T([8192, 768], f32), T([8, 1024, 1], f32), T([8192, 768], f32), S([8, 1024, 2]), S([1, 2]), S([8, 2]))"

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[8192, 2]", primals_1: "i64[8, 1024]", iota_1: "i64[8]", primals_151: "i64[8]", full_default_2: "f32[]", rsqrt_24: "f32[8, 1024, 1]", view_143: "f32[8192, 3072]", view_141: "f32[8192, 768]", rsqrt_23: "f32[8, 1024, 1]", view_133: "f32[8192, 768]", rsqrt_22: "f32[8, 1024, 1]", view_131: "f32[8192, 3072]", view_129: "f32[8192, 768]", rsqrt_21: "f32[8, 1024, 1]", view_121: "f32[8192, 768]", rsqrt_20: "f32[8, 1024, 1]", view_119: "f32[8192, 3072]", view_117: "f32[8192, 768]", rsqrt_19: "f32[8, 1024, 1]", view_109: "f32[8192, 768]", rsqrt_18: "f32[8, 1024, 1]", view_107: "f32[8192, 3072]", view_105: "f32[8192, 768]", rsqrt_17: "f32[8, 1024, 1]", view_97: "f32[8192, 768]", rsqrt_16: "f32[8, 1024, 1]", view_95: "f32[8192, 3072]", view_93: "f32[8192, 768]", rsqrt_15: "f32[8, 1024, 1]", view_85: "f32[8192, 768]", rsqrt_14: "f32[8, 1024, 1]", view_83: "f32[8192, 3072]", view_81: "f32[8192, 768]", rsqrt_13: "f32[8, 1024, 1]", view_73: "f32[8192, 768]", rsqrt_12: "f32[8, 1024, 1]", view_71: "f32[8192, 3072]", view_69: "f32[8192, 768]", rsqrt_11: "f32[8, 1024, 1]", view_61: "f32[8192, 768]", rsqrt_10: "f32[8, 1024, 1]", view_59: "f32[8192, 3072]", view_57: "f32[8192, 768]", rsqrt_9: "f32[8, 1024, 1]", view_49: "f32[8192, 768]", rsqrt_8: "f32[8, 1024, 1]", view_47: "f32[8192, 3072]", view_45: "f32[8192, 768]", rsqrt_7: "f32[8, 1024, 1]", view_37: "f32[8192, 768]", rsqrt_6: "f32[8, 1024, 1]", view_35: "f32[8192, 3072]", view_33: "f32[8192, 768]", rsqrt_5: "f32[8, 1024, 1]", view_25: "f32[8192, 768]", rsqrt_4: "f32[8, 1024, 1]", view_23: "f32[8192, 3072]", view_21: "f32[8192, 768]", rsqrt_3: "f32[8, 1024, 1]", view_13: "f32[8192, 768]", rsqrt_2: "f32[8, 1024, 1]", view_11: "f32[8192, 3072]", view_9: "f32[8192, 768]", rsqrt_1: "f32[8, 1024, 1]", view_1: "f32[8192, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:920 in forward, code: logits = self.score(hidden_states)
        reshape_default: "f32[8, 1024, 2]" = torch.ops.aten.reshape.default(mm, _shape_param_0);  mm = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:933 in forward, code: non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
        ne_scalar: "b8[8, 1024]" = torch.ops.aten.ne.Scalar(primals_1, 0);  primals_1 = None
        convert_element_type_default: "i32[8, 1024]" = torch.ops.prims.convert_element_type.default(ne_scalar, torch.int32);  ne_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:934 in forward, code: token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
        iota_default: "i32[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:935 in forward, code: last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        mul_tensor: "i32[8, 1024]" = torch.ops.aten.mul.Tensor(iota_default, convert_element_type_default);  iota_default = convert_element_type_default = None
        argmax_default: "i64[8]" = torch.ops.aten.argmax.default(mul_tensor, -1);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:943 in forward, code: pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]
        index_tensor: "f32[8, 2]" = torch.ops.aten.index.Tensor(reshape_default, [iota_1, argmax_default]);  reshape_default = iota_1 = argmax_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:963 in forward, code: loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
        amax_default: "f32[8, 1]" = torch.ops.aten.amax.default(index_tensor, [1], True)
        sub_tensor: "f32[8, 2]" = torch.ops.aten.sub.Tensor(index_tensor, amax_default);  index_tensor = amax_default = None
        exp_default: "f32[8, 2]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[8, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[8, 2]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        ne_scalar_1: "b8[8]" = torch.ops.aten.ne.Scalar(primals_151, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[8]" = torch.ops.aten.where.self(ne_scalar_1, primals_151, full_default)
        unsqueeze_default: "i64[8, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "f32[8, 1]" = torch.ops.aten.gather.default(sub_tensor_1, 1, unsqueeze_default);  sub_tensor_1 = unsqueeze_default = None
        squeeze_dim: "f32[8]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "f32[8]" = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        where_self_1: "f32[8]" = torch.ops.aten.where.self(ne_scalar_1, neg_default, full_default_2);  neg_default = full_default_2 = None
        sum_default: "i64[]" = torch.ops.aten.sum.default(ne_scalar_1);  ne_scalar_1 = None
        convert_element_type_default_1: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default, torch.float32);  sum_default = None
        sum_default_1: "f32[]" = torch.ops.aten.sum.default(where_self_1);  where_self_1 = None
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(sum_default_1, convert_element_type_default_1);  sum_default_1 = convert_element_type_default_1 = None
        unsqueeze_default_1: "i64[8, 1]" = torch.ops.aten.unsqueeze.default(primals_151, 1);  primals_151 = None
        ne_scalar_2: "b8[8, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_default_1, -100)
        where_self_2: "i64[8, 1]" = torch.ops.aten.where.self(ne_scalar_2, unsqueeze_default_1, full_default);  ne_scalar_2 = unsqueeze_default_1 = full_default = None

        # No stacktrace found for following nodes
        iota_default_1: "i64[2]" = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        reshape_default_1: "i64[1, 2]" = torch.ops.aten.reshape.default(iota_default_1, _shape_param_1);  iota_default_1 = _shape_param_1 = None
        expand_default: "i64[8, 2]" = torch.ops.aten.expand.default(where_self_2, _shape_param_2);  where_self_2 = _shape_param_2 = None
        eq_tensor: "b8[8, 2]" = torch.ops.aten.eq.Tensor(expand_default, reshape_default_1);  expand_default = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:628 in forward, code: hidden_states = self.ln_f(hidden_states)
        div_tensor_1: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_143, [1, 0]);  view_143 = None
        permute_default_1: "f32[768, 8192]" = torch.ops.aten.permute.default(view_141, [1, 0]);  view_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_2: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_2: "f32[768, 8192]" = torch.ops.aten.permute.default(view_133, [1, 0]);  view_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_3: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_3: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_131, [1, 0]);  view_131 = None
        permute_default_4: "f32[768, 8192]" = torch.ops.aten.permute.default(view_129, [1, 0]);  view_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_4: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_5: "f32[768, 8192]" = torch.ops.aten.permute.default(view_121, [1, 0]);  view_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_5: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_6: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_119, [1, 0]);  view_119 = None
        permute_default_7: "f32[768, 8192]" = torch.ops.aten.permute.default(view_117, [1, 0]);  view_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_6: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_8: "f32[768, 8192]" = torch.ops.aten.permute.default(view_109, [1, 0]);  view_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_7: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_9: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_107, [1, 0]);  view_107 = None
        permute_default_10: "f32[768, 8192]" = torch.ops.aten.permute.default(view_105, [1, 0]);  view_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_8: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_11: "f32[768, 8192]" = torch.ops.aten.permute.default(view_97, [1, 0]);  view_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_9: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_12: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_95, [1, 0]);  view_95 = None
        permute_default_13: "f32[768, 8192]" = torch.ops.aten.permute.default(view_93, [1, 0]);  view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_10: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_14: "f32[768, 8192]" = torch.ops.aten.permute.default(view_85, [1, 0]);  view_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_11: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_15: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_83, [1, 0]);  view_83 = None
        permute_default_16: "f32[768, 8192]" = torch.ops.aten.permute.default(view_81, [1, 0]);  view_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_12: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_17: "f32[768, 8192]" = torch.ops.aten.permute.default(view_73, [1, 0]);  view_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_13: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_18: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_71, [1, 0]);  view_71 = None
        permute_default_19: "f32[768, 8192]" = torch.ops.aten.permute.default(view_69, [1, 0]);  view_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_14: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_20: "f32[768, 8192]" = torch.ops.aten.permute.default(view_61, [1, 0]);  view_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_15: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_21: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_59, [1, 0]);  view_59 = None
        permute_default_22: "f32[768, 8192]" = torch.ops.aten.permute.default(view_57, [1, 0]);  view_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_16: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_23: "f32[768, 8192]" = torch.ops.aten.permute.default(view_49, [1, 0]);  view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_17: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_24: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_47, [1, 0]);  view_47 = None
        permute_default_25: "f32[768, 8192]" = torch.ops.aten.permute.default(view_45, [1, 0]);  view_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_18: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_26: "f32[768, 8192]" = torch.ops.aten.permute.default(view_37, [1, 0]);  view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_19: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_27: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_35, [1, 0]);  view_35 = None
        permute_default_28: "f32[768, 8192]" = torch.ops.aten.permute.default(view_33, [1, 0]);  view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_20: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_29: "f32[768, 8192]" = torch.ops.aten.permute.default(view_25, [1, 0]);  view_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_21: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_30: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_23, [1, 0]);  view_23 = None
        permute_default_31: "f32[768, 8192]" = torch.ops.aten.permute.default(view_21, [1, 0]);  view_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_22: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_32: "f32[768, 8192]" = torch.ops.aten.permute.default(view_13, [1, 0]);  view_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_23: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_33: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_11, [1, 0]);  view_11 = None
        permute_default_34: "f32[768, 8192]" = torch.ops.aten.permute.default(view_9, [1, 0]);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        div_tensor_24: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_default_35: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1, [1, 0]);  view_1 = None
        return (div_tensor, eq_tensor, div_tensor_1, permute_default, permute_default_1, div_tensor_2, permute_default_2, div_tensor_3, permute_default_3, permute_default_4, div_tensor_4, permute_default_5, div_tensor_5, permute_default_6, permute_default_7, div_tensor_6, permute_default_8, div_tensor_7, permute_default_9, permute_default_10, div_tensor_8, permute_default_11, div_tensor_9, permute_default_12, permute_default_13, div_tensor_10, permute_default_14, div_tensor_11, permute_default_15, permute_default_16, div_tensor_12, permute_default_17, div_tensor_13, permute_default_18, permute_default_19, div_tensor_14, permute_default_20, div_tensor_15, permute_default_21, permute_default_22, div_tensor_16, permute_default_23, div_tensor_17, permute_default_24, permute_default_25, div_tensor_18, permute_default_26, div_tensor_19, permute_default_27, permute_default_28, div_tensor_20, permute_default_29, div_tensor_21, permute_default_30, permute_default_31, div_tensor_22, permute_default_32, div_tensor_23, permute_default_33, permute_default_34, div_tensor_24, permute_default_35)


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
