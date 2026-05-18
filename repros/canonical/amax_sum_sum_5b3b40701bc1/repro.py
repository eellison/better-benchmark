"""
Standalone repro captured via capture_hook.
Label: hf_YituTechConvBert_training
Pattern hash: 5b3b40701bc1
Shape hash: d6707607
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
    def forward(self, addmm_85: "f32[4096, 30522]", _shape_param_0, _shape_param_1, primals_290: "i64[8, 512]", rsqrt_24: "f32[8, 512, 1]", rsqrt_23: "f32[8, 512, 1]", view_421: "f32[48, 512, 512]", view_422: "f32[48, 512, 64]", view_418: "f32[48, 512, 64]", view_419: "f32[48, 64, 512]", expand_67: "f32[24576, 64, 9]", rsqrt_22: "f32[8, 512, 1]", rsqrt_21: "f32[8, 512, 1]", view_385: "f32[48, 512, 512]", view_386: "f32[48, 512, 64]", view_382: "f32[48, 512, 64]", view_383: "f32[48, 64, 512]", expand_61: "f32[24576, 64, 9]", rsqrt_20: "f32[8, 512, 1]", rsqrt_19: "f32[8, 512, 1]", view_349: "f32[48, 512, 512]", view_350: "f32[48, 512, 64]", view_346: "f32[48, 512, 64]", view_347: "f32[48, 64, 512]", expand_55: "f32[24576, 64, 9]", rsqrt_18: "f32[8, 512, 1]", rsqrt_17: "f32[8, 512, 1]", view_313: "f32[48, 512, 512]", view_314: "f32[48, 512, 64]", view_310: "f32[48, 512, 64]", view_311: "f32[48, 64, 512]", expand_49: "f32[24576, 64, 9]", rsqrt_16: "f32[8, 512, 1]", rsqrt_15: "f32[8, 512, 1]", view_277: "f32[48, 512, 512]", view_278: "f32[48, 512, 64]", view_274: "f32[48, 512, 64]", view_275: "f32[48, 64, 512]", expand_43: "f32[24576, 64, 9]", rsqrt_14: "f32[8, 512, 1]", rsqrt_13: "f32[8, 512, 1]", view_241: "f32[48, 512, 512]", view_242: "f32[48, 512, 64]", view_238: "f32[48, 512, 64]", view_239: "f32[48, 64, 512]", expand_37: "f32[24576, 64, 9]", rsqrt_12: "f32[8, 512, 1]", rsqrt_11: "f32[8, 512, 1]", view_205: "f32[48, 512, 512]", view_206: "f32[48, 512, 64]", view_202: "f32[48, 512, 64]", view_203: "f32[48, 64, 512]", expand_31: "f32[24576, 64, 9]", rsqrt_10: "f32[8, 512, 1]", rsqrt_9: "f32[8, 512, 1]", view_169: "f32[48, 512, 512]", view_170: "f32[48, 512, 64]", view_166: "f32[48, 512, 64]", view_167: "f32[48, 64, 512]", expand_25: "f32[24576, 64, 9]", rsqrt_8: "f32[8, 512, 1]", rsqrt_7: "f32[8, 512, 1]", view_133: "f32[48, 512, 512]", view_134: "f32[48, 512, 64]", view_130: "f32[48, 512, 64]", view_131: "f32[48, 64, 512]", expand_19: "f32[24576, 64, 9]", rsqrt_6: "f32[8, 512, 1]", rsqrt_5: "f32[8, 512, 1]", view_97: "f32[48, 512, 512]", view_98: "f32[48, 512, 64]", view_94: "f32[48, 512, 64]", view_95: "f32[48, 64, 512]", expand_13: "f32[24576, 64, 9]", rsqrt_4: "f32[8, 512, 1]", rsqrt_3: "f32[8, 512, 1]", view_61: "f32[48, 512, 512]", view_62: "f32[48, 512, 64]", view_58: "f32[48, 512, 64]", view_59: "f32[48, 64, 512]", expand_7: "f32[24576, 64, 9]", rsqrt_2: "f32[8, 512, 1]", rsqrt_1: "f32[8, 512, 1]", view_25: "f32[48, 512, 512]", view_26: "f32[48, 512, 64]", view_22: "f32[48, 512, 64]", view_23: "f32[48, 64, 512]", expand_1: "f32[24576, 64, 9]", rsqrt: "f32[8, 512, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:718 in forward, code: prediction_scores = self.generator_lm_head(prediction_scores)
        reshape_default: "f32[8, 512, 30522]" = torch.ops.aten.reshape.default(addmm_85, _shape_param_0);  addmm_85 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:724 in forward, code: loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        reshape_default_1: "f32[4096, 30522]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        reshape_default_2: "i64[4096]" = torch.ops.aten.reshape.default(primals_290, [-1]);  primals_290 = None
        amax_default: "f32[4096, 1]" = torch.ops.aten.amax.default(reshape_default_1, [1], True)
        sub_tensor: "f32[4096, 30522]" = torch.ops.aten.sub.Tensor(reshape_default_1, amax_default);  reshape_default_1 = amax_default = None
        exp_default: "f32[4096, 30522]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[4096, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[4096, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[4096, 30522]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
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

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_1: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_2: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_421, [0, 2, 1]);  view_421 = None
        permute_default_1: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_422, [0, 2, 1]);  view_422 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:227 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_2: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_418, [0, 2, 1]);  view_418 = None
        permute_default_3: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_419, [0, 2, 1]);  view_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        permute_default_4: "f32[24576, 9, 64]" = torch.ops.aten.permute.default(expand_67, [0, 2, 1]);  expand_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_3: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_4: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default_5: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_385, [0, 2, 1]);  view_385 = None
        permute_default_6: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_386, [0, 2, 1]);  view_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:227 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_7: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_382, [0, 2, 1]);  view_382 = None
        permute_default_8: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_383, [0, 2, 1]);  view_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        permute_default_9: "f32[24576, 9, 64]" = torch.ops.aten.permute.default(expand_61, [0, 2, 1]);  expand_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_5: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_6: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default_10: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_349, [0, 2, 1]);  view_349 = None
        permute_default_11: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_350, [0, 2, 1]);  view_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:227 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_12: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_346, [0, 2, 1]);  view_346 = None
        permute_default_13: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_347, [0, 2, 1]);  view_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        permute_default_14: "f32[24576, 9, 64]" = torch.ops.aten.permute.default(expand_55, [0, 2, 1]);  expand_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_7: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_8: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default_15: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_313, [0, 2, 1]);  view_313 = None
        permute_default_16: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_314, [0, 2, 1]);  view_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:227 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_17: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_310, [0, 2, 1]);  view_310 = None
        permute_default_18: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_311, [0, 2, 1]);  view_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        permute_default_19: "f32[24576, 9, 64]" = torch.ops.aten.permute.default(expand_49, [0, 2, 1]);  expand_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_9: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_10: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default_20: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_277, [0, 2, 1]);  view_277 = None
        permute_default_21: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_278, [0, 2, 1]);  view_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:227 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_22: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_274, [0, 2, 1]);  view_274 = None
        permute_default_23: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_275, [0, 2, 1]);  view_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        permute_default_24: "f32[24576, 9, 64]" = torch.ops.aten.permute.default(expand_43, [0, 2, 1]);  expand_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_11: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_12: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default_25: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_241, [0, 2, 1]);  view_241 = None
        permute_default_26: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_242, [0, 2, 1]);  view_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:227 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_27: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_238, [0, 2, 1]);  view_238 = None
        permute_default_28: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_239, [0, 2, 1]);  view_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        permute_default_29: "f32[24576, 9, 64]" = torch.ops.aten.permute.default(expand_37, [0, 2, 1]);  expand_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_13: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_14: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default_30: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_205, [0, 2, 1]);  view_205 = None
        permute_default_31: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_206, [0, 2, 1]);  view_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:227 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_32: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_202, [0, 2, 1]);  view_202 = None
        permute_default_33: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_203, [0, 2, 1]);  view_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        permute_default_34: "f32[24576, 9, 64]" = torch.ops.aten.permute.default(expand_31, [0, 2, 1]);  expand_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_15: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_16: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default_35: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_169, [0, 2, 1]);  view_169 = None
        permute_default_36: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_170, [0, 2, 1]);  view_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:227 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_37: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
        permute_default_38: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        permute_default_39: "f32[24576, 9, 64]" = torch.ops.aten.permute.default(expand_25, [0, 2, 1]);  expand_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_17: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_18: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default_40: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_133, [0, 2, 1]);  view_133 = None
        permute_default_41: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_134, [0, 2, 1]);  view_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:227 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_42: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_130, [0, 2, 1]);  view_130 = None
        permute_default_43: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_131, [0, 2, 1]);  view_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        permute_default_44: "f32[24576, 9, 64]" = torch.ops.aten.permute.default(expand_19, [0, 2, 1]);  expand_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_19: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_20: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default_45: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
        permute_default_46: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:227 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_47: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_94, [0, 2, 1]);  view_94 = None
        permute_default_48: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_95, [0, 2, 1]);  view_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        permute_default_49: "f32[24576, 9, 64]" = torch.ops.aten.permute.default(expand_13, [0, 2, 1]);  expand_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_21: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_22: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default_50: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_61, [0, 2, 1]);  view_61 = None
        permute_default_51: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_62, [0, 2, 1]);  view_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:227 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_52: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_58, [0, 2, 1]);  view_58 = None
        permute_default_53: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_59, [0, 2, 1]);  view_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        permute_default_54: "f32[24576, 9, 64]" = torch.ops.aten.permute.default(expand_7, [0, 2, 1]);  expand_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_23: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_24: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        permute_default_55: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_25, [0, 2, 1]);  view_25 = None
        permute_default_56: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_26, [0, 2, 1]);  view_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:227 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        permute_default_57: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_22, [0, 2, 1]);  view_22 = None
        permute_default_58: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_23, [0, 2, 1]);  view_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        permute_default_59: "f32[24576, 9, 64]" = torch.ops.aten.permute.default(expand_1, [0, 2, 1]);  expand_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:104 in forward, code: embeddings = self.LayerNorm(embeddings)
        div_tensor_25: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        return (div_tensor, div_tensor_1, div_tensor_2, permute_default, permute_default_1, permute_default_2, permute_default_3, permute_default_4, div_tensor_3, div_tensor_4, permute_default_5, permute_default_6, permute_default_7, permute_default_8, permute_default_9, div_tensor_5, div_tensor_6, permute_default_10, permute_default_11, permute_default_12, permute_default_13, permute_default_14, div_tensor_7, div_tensor_8, permute_default_15, permute_default_16, permute_default_17, permute_default_18, permute_default_19, div_tensor_9, div_tensor_10, permute_default_20, permute_default_21, permute_default_22, permute_default_23, permute_default_24, div_tensor_11, div_tensor_12, permute_default_25, permute_default_26, permute_default_27, permute_default_28, permute_default_29, div_tensor_13, div_tensor_14, permute_default_30, permute_default_31, permute_default_32, permute_default_33, permute_default_34, div_tensor_15, div_tensor_16, permute_default_35, permute_default_36, permute_default_37, permute_default_38, permute_default_39, div_tensor_17, div_tensor_18, permute_default_40, permute_default_41, permute_default_42, permute_default_43, permute_default_44, div_tensor_19, div_tensor_20, permute_default_45, permute_default_46, permute_default_47, permute_default_48, permute_default_49, div_tensor_21, div_tensor_22, permute_default_50, permute_default_51, permute_default_52, permute_default_53, permute_default_54, div_tensor_23, div_tensor_24, permute_default_55, permute_default_56, permute_default_57, permute_default_58, permute_default_59, div_tensor_25)


def _default_make_inputs():
    return [
    torch.randn([4096, 30522], dtype=torch.float32, device='cuda'),
    [8, 512, 30522],  # _shape_param_0
    [-1, 30522],  # _shape_param_1
    torch.randint(0, 30522, [8, 512], dtype=torch.int64, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([24576, 64, 9], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([24576, 64, 9], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([24576, 64, 9], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([24576, 64, 9], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([24576, 64, 9], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([24576, 64, 9], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([24576, 64, 9], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([24576, 64, 9], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([24576, 64, 9], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([24576, 64, 9], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([24576, 64, 9], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([48, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([24576, 64, 9], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
