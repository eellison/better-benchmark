"""
Standalone repro captured via capture_hook.
Label: hf_DebertaV2ForMaskedLM_train
Pattern hash: ef60757a8fbe
Shape hash: 25bce691
"""
_shapes_config = "(T([4096, 128100], f32), T([8, 512], i64), T([8, 512, 1], f32), T([8, 512, 1], f32), T([192, 512, 512], f32), T([192, 512, 64], f32), T([192, 512, 64], f32), T([192, 64, 512], f32, stride=(32768, 1, 64)), T([8, 512, 1], f32), T([8, 512, 1], f32), T([192, 512, 512], f32), T([192, 512, 64], f32), T([192, 512, 64], f32), T([192, 64, 512], f32, stride=(32768, 1, 64)), T([8, 512, 1], f32), T([8, 512, 1], f32), T([192, 512, 512], f32), T([192, 512, 64], f32), T([192, 512, 64], f32), T([192, 64, 512], f32, stride=(32768, 1, 64)), T([8, 512, 1], f32), T([8, 512, 1], f32), T([192, 512, 512], f32), T([192, 512, 64], f32), T([192, 512, 64], f32), T([192, 64, 512], f32, stride=(32768, 1, 64)), T([8, 512, 1], f32), T([8, 512, 1], f32), T([192, 512, 512], f32), T([192, 512, 64], f32), T([192, 512, 64], f32), T([192, 64, 512], f32, stride=(32768, 1, 64)), T([8, 512, 1], f32), T([8, 512, 1], f32), T([192, 512, 512], f32), T([192, 512, 64], f32), T([192, 512, 64], f32), T([192, 64, 512], f32, stride=(32768, 1, 64)), T([8, 512, 1], f32), T([8, 512, 1], f32), T([192, 512, 512], f32), T([192, 512, 64], f32), T([192, 512, 64], f32), T([192, 64, 512], f32, stride=(32768, 1, 64)), T([8, 512, 1], f32), T([8, 512, 1], f32), T([192, 512, 512], f32), T([192, 512, 64], f32), T([192, 512, 64], f32), T([192, 64, 512], f32, stride=(32768, 1, 64)), T([8, 512, 1], f32), T([8, 512, 1], f32), T([192, 512, 512], f32), T([192, 512, 64], f32), T([192, 512, 64], f32), T([192, 64, 512], f32, stride=(32768, 1, 64)), T([8, 512, 1], f32), T([8, 512, 1], f32), T([192, 512, 512], f32), T([192, 512, 64], f32), T([192, 512, 64], f32), T([192, 64, 512], f32, stride=(32768, 1, 64)), T([8, 512, 1], f32), T([8, 512, 1], f32), T([192, 512, 512], f32), T([192, 512, 64], f32), T([192, 512, 64], f32), T([192, 64, 512], f32, stride=(32768, 1, 64)), T([8, 512, 1], f32), T([8, 512, 1], f32), T([192, 512, 512], f32), T([192, 512, 64], f32), T([192, 512, 64], f32), T([192, 64, 512], f32, stride=(32768, 1, 64)), T([8, 512, 1], f32), T([8, 512, 1], f32), T([192, 512, 512], f32), T([192, 512, 64], f32), T([192, 512, 64], f32), T([192, 64, 512], f32, stride=(32768, 1, 64)), T([8, 512, 1], f32), T([8, 512, 1], f32), T([192, 512, 512], f32), T([192, 512, 64], f32), T([192, 512, 64], f32), T([192, 64, 512], f32, stride=(32768, 1, 64)), T([8, 512, 1], f32), T([8, 512, 1], f32), T([192, 512, 512], f32), T([192, 512, 64], f32), T([192, 512, 64], f32), T([192, 64, 512], f32, stride=(32768, 1, 64)), T([8, 512, 1], f32), T([8, 512, 1], f32), T([192, 512, 512], f32), T([192, 512, 64], f32), T([192, 512, 64], f32), T([192, 64, 512], f32, stride=(32768, 1, 64)), T([8, 512, 1], f32), T([8, 512, 1], f32), T([192, 512, 512], f32), T([192, 512, 64], f32), T([192, 512, 64], f32), T([192, 64, 512], f32, stride=(32768, 1, 64)), T([8, 512, 1], f32), T([8, 512, 1], f32), T([192, 512, 512], f32), T([192, 512, 64], f32), T([192, 512, 64], f32), T([192, 64, 512], f32, stride=(32768, 1, 64)), T([8, 512, 1], f32), T([8, 512, 1], f32), T([192, 512, 512], f32), T([192, 512, 64], f32), T([192, 512, 64], f32), T([192, 64, 512], f32, stride=(32768, 1, 64)), T([8, 512, 1], f32), T([8, 512, 1], f32), T([192, 512, 512], f32), T([192, 512, 64], f32), T([192, 512, 64], f32), T([192, 64, 512], f32, stride=(32768, 1, 64)), T([8, 512, 1], f32), T([8, 512, 1], f32), T([192, 512, 512], f32), T([192, 512, 64], f32), T([192, 512, 64], f32), T([192, 64, 512], f32, stride=(32768, 1, 64)), T([8, 512, 1], f32), T([8, 512, 1], f32), T([192, 512, 512], f32), T([192, 512, 64], f32), T([192, 512, 64], f32), T([192, 64, 512], f32, stride=(32768, 1, 64)), T([8, 512, 1], f32), T([8, 512, 1], f32), T([192, 512, 512], f32), T([192, 512, 64], f32), T([192, 512, 64], f32), T([192, 64, 512], f32, stride=(32768, 1, 64)), T([8, 512, 1], f32), T([8, 512, 1], f32), T([192, 512, 512], f32), T([192, 512, 64], f32), T([192, 512, 64], f32), T([192, 64, 512], f32, stride=(32768, 1, 64)), S([8, 512, 128100]), S([-1, 128100]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_145: "f32[4096, 128100]", primals_396: "i64[8, 512]", rsqrt_48: "f32[8, 512, 1]", rsqrt_47: "f32[8, 512, 1]", view_519: "f32[192, 512, 512]", view_517: "f32[192, 512, 64]", view_509: "f32[192, 512, 64]", div_46: "f32[192, 64, 512]", rsqrt_46: "f32[8, 512, 1]", rsqrt_45: "f32[8, 512, 1]", view_497: "f32[192, 512, 512]", view_495: "f32[192, 512, 64]", view_487: "f32[192, 512, 64]", div_44: "f32[192, 64, 512]", rsqrt_44: "f32[8, 512, 1]", rsqrt_43: "f32[8, 512, 1]", view_475: "f32[192, 512, 512]", view_473: "f32[192, 512, 64]", view_465: "f32[192, 512, 64]", div_42: "f32[192, 64, 512]", rsqrt_42: "f32[8, 512, 1]", rsqrt_41: "f32[8, 512, 1]", view_453: "f32[192, 512, 512]", view_451: "f32[192, 512, 64]", view_443: "f32[192, 512, 64]", div_40: "f32[192, 64, 512]", rsqrt_40: "f32[8, 512, 1]", rsqrt_39: "f32[8, 512, 1]", view_431: "f32[192, 512, 512]", view_429: "f32[192, 512, 64]", view_421: "f32[192, 512, 64]", div_38: "f32[192, 64, 512]", rsqrt_38: "f32[8, 512, 1]", rsqrt_37: "f32[8, 512, 1]", view_409: "f32[192, 512, 512]", view_407: "f32[192, 512, 64]", view_399: "f32[192, 512, 64]", div_36: "f32[192, 64, 512]", rsqrt_36: "f32[8, 512, 1]", rsqrt_35: "f32[8, 512, 1]", view_387: "f32[192, 512, 512]", view_385: "f32[192, 512, 64]", view_377: "f32[192, 512, 64]", div_34: "f32[192, 64, 512]", rsqrt_34: "f32[8, 512, 1]", rsqrt_33: "f32[8, 512, 1]", view_365: "f32[192, 512, 512]", view_363: "f32[192, 512, 64]", view_355: "f32[192, 512, 64]", div_32: "f32[192, 64, 512]", rsqrt_32: "f32[8, 512, 1]", rsqrt_31: "f32[8, 512, 1]", view_343: "f32[192, 512, 512]", view_341: "f32[192, 512, 64]", view_333: "f32[192, 512, 64]", div_30: "f32[192, 64, 512]", rsqrt_30: "f32[8, 512, 1]", rsqrt_29: "f32[8, 512, 1]", view_321: "f32[192, 512, 512]", view_319: "f32[192, 512, 64]", view_311: "f32[192, 512, 64]", div_28: "f32[192, 64, 512]", rsqrt_28: "f32[8, 512, 1]", rsqrt_27: "f32[8, 512, 1]", view_299: "f32[192, 512, 512]", view_297: "f32[192, 512, 64]", view_289: "f32[192, 512, 64]", div_26: "f32[192, 64, 512]", rsqrt_26: "f32[8, 512, 1]", rsqrt_25: "f32[8, 512, 1]", view_277: "f32[192, 512, 512]", view_275: "f32[192, 512, 64]", view_267: "f32[192, 512, 64]", div_24: "f32[192, 64, 512]", rsqrt_24: "f32[8, 512, 1]", rsqrt_23: "f32[8, 512, 1]", view_255: "f32[192, 512, 512]", view_253: "f32[192, 512, 64]", view_245: "f32[192, 512, 64]", div_22: "f32[192, 64, 512]", rsqrt_22: "f32[8, 512, 1]", rsqrt_21: "f32[8, 512, 1]", view_233: "f32[192, 512, 512]", view_231: "f32[192, 512, 64]", view_223: "f32[192, 512, 64]", div_20: "f32[192, 64, 512]", rsqrt_20: "f32[8, 512, 1]", rsqrt_19: "f32[8, 512, 1]", view_211: "f32[192, 512, 512]", view_209: "f32[192, 512, 64]", view_201: "f32[192, 512, 64]", div_18: "f32[192, 64, 512]", rsqrt_18: "f32[8, 512, 1]", rsqrt_17: "f32[8, 512, 1]", view_189: "f32[192, 512, 512]", view_187: "f32[192, 512, 64]", view_179: "f32[192, 512, 64]", div_16: "f32[192, 64, 512]", rsqrt_16: "f32[8, 512, 1]", rsqrt_15: "f32[8, 512, 1]", view_167: "f32[192, 512, 512]", view_165: "f32[192, 512, 64]", view_157: "f32[192, 512, 64]", div_14: "f32[192, 64, 512]", rsqrt_14: "f32[8, 512, 1]", rsqrt_13: "f32[8, 512, 1]", view_145: "f32[192, 512, 512]", view_143: "f32[192, 512, 64]", view_135: "f32[192, 512, 64]", div_12: "f32[192, 64, 512]", rsqrt_12: "f32[8, 512, 1]", rsqrt_11: "f32[8, 512, 1]", view_123: "f32[192, 512, 512]", view_121: "f32[192, 512, 64]", view_113: "f32[192, 512, 64]", div_10: "f32[192, 64, 512]", rsqrt_10: "f32[8, 512, 1]", rsqrt_9: "f32[8, 512, 1]", view_101: "f32[192, 512, 512]", view_99: "f32[192, 512, 64]", view_91: "f32[192, 512, 64]", div_8: "f32[192, 64, 512]", rsqrt_8: "f32[8, 512, 1]", rsqrt_7: "f32[8, 512, 1]", view_79: "f32[192, 512, 512]", view_77: "f32[192, 512, 64]", view_69: "f32[192, 512, 64]", div_6: "f32[192, 64, 512]", rsqrt_6: "f32[8, 512, 1]", rsqrt_5: "f32[8, 512, 1]", view_57: "f32[192, 512, 512]", view_55: "f32[192, 512, 64]", view_47: "f32[192, 512, 64]", div_4: "f32[192, 64, 512]", rsqrt_4: "f32[8, 512, 1]", rsqrt_3: "f32[8, 512, 1]", view_35: "f32[192, 512, 512]", view_33: "f32[192, 512, 64]", view_25: "f32[192, 512, 64]", div_2: "f32[192, 64, 512]", rsqrt_2: "f32[8, 512, 1]", rsqrt_1: "f32[8, 512, 1]", view_13: "f32[192, 512, 512]", view_11: "f32[192, 512, 64]", view_3: "f32[192, 512, 64]", div: "f32[192, 64, 512]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:838 in forward, code: hidden_states = self.decoder(hidden_states)
        reshape_default: "f32[8, 512, 128100]" = torch.ops.aten.reshape.default(addmm_145, _shape_param_0);  addmm_145 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:968 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        reshape_default_1: "f32[4096, 128100]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        reshape_default_2: "i64[4096]" = torch.ops.aten.reshape.default(primals_396, [-1]);  primals_396 = None
        amax_default: "f32[4096, 1]" = torch.ops.aten.amax.default(reshape_default_1, [1], True)
        sub_tensor: "f32[4096, 128100]" = torch.ops.aten.sub.Tensor(reshape_default_1, amax_default);  reshape_default_1 = amax_default = None
        exp_default: "f32[4096, 128100]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[4096, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[4096, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[4096, 128100]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
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

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_1: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_48, 1536);  rsqrt_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_2: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 1536);  rsqrt_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_default: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_519, [0, 2, 1]);  view_519 = None
        permute_default_1: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_517, [0, 2, 1]);  view_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_default_2: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_509, [0, 2, 1]);  view_509 = None
        permute_default_3: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_46, [0, 2, 1]);  div_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_3: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 1536);  rsqrt_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_4: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 1536);  rsqrt_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_default_4: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_497, [0, 2, 1]);  view_497 = None
        permute_default_5: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_495, [0, 2, 1]);  view_495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_default_6: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_487, [0, 2, 1]);  view_487 = None
        permute_default_7: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_44, [0, 2, 1]);  div_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_5: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 1536);  rsqrt_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_6: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 1536);  rsqrt_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_default_8: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_475, [0, 2, 1]);  view_475 = None
        permute_default_9: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_473, [0, 2, 1]);  view_473 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_default_10: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_465, [0, 2, 1]);  view_465 = None
        permute_default_11: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_42, [0, 2, 1]);  div_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_7: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 1536);  rsqrt_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_8: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 1536);  rsqrt_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_default_12: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_453, [0, 2, 1]);  view_453 = None
        permute_default_13: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_451, [0, 2, 1]);  view_451 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_default_14: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_443, [0, 2, 1]);  view_443 = None
        permute_default_15: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_40, [0, 2, 1]);  div_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_9: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 1536);  rsqrt_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_10: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 1536);  rsqrt_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_default_16: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_431, [0, 2, 1]);  view_431 = None
        permute_default_17: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_429, [0, 2, 1]);  view_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_default_18: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_421, [0, 2, 1]);  view_421 = None
        permute_default_19: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_38, [0, 2, 1]);  div_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_11: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 1536);  rsqrt_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_12: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 1536);  rsqrt_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_default_20: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_409, [0, 2, 1]);  view_409 = None
        permute_default_21: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_407, [0, 2, 1]);  view_407 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_default_22: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_399, [0, 2, 1]);  view_399 = None
        permute_default_23: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_36, [0, 2, 1]);  div_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_13: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 1536);  rsqrt_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_14: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 1536);  rsqrt_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_default_24: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_387, [0, 2, 1]);  view_387 = None
        permute_default_25: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_385, [0, 2, 1]);  view_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_default_26: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_377, [0, 2, 1]);  view_377 = None
        permute_default_27: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_34, [0, 2, 1]);  div_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_15: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 1536);  rsqrt_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_16: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 1536);  rsqrt_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_default_28: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_365, [0, 2, 1]);  view_365 = None
        permute_default_29: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_363, [0, 2, 1]);  view_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_default_30: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_355, [0, 2, 1]);  view_355 = None
        permute_default_31: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_32, [0, 2, 1]);  div_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_17: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 1536);  rsqrt_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_18: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 1536);  rsqrt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_default_32: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_343, [0, 2, 1]);  view_343 = None
        permute_default_33: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_341, [0, 2, 1]);  view_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_default_34: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_333, [0, 2, 1]);  view_333 = None
        permute_default_35: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_30, [0, 2, 1]);  div_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_19: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 1536);  rsqrt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_20: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 1536);  rsqrt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_default_36: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_321, [0, 2, 1]);  view_321 = None
        permute_default_37: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_319, [0, 2, 1]);  view_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_default_38: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_311, [0, 2, 1]);  view_311 = None
        permute_default_39: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_28, [0, 2, 1]);  div_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_21: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 1536);  rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_22: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 1536);  rsqrt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_default_40: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_299, [0, 2, 1]);  view_299 = None
        permute_default_41: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_297, [0, 2, 1]);  view_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_default_42: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_289, [0, 2, 1]);  view_289 = None
        permute_default_43: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_26, [0, 2, 1]);  div_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_23: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 1536);  rsqrt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_24: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 1536);  rsqrt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_default_44: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_277, [0, 2, 1]);  view_277 = None
        permute_default_45: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_275, [0, 2, 1]);  view_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_default_46: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_267, [0, 2, 1]);  view_267 = None
        permute_default_47: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_24, [0, 2, 1]);  div_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_25: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 1536);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_26: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 1536);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_default_48: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_255, [0, 2, 1]);  view_255 = None
        permute_default_49: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_253, [0, 2, 1]);  view_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_default_50: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_245, [0, 2, 1]);  view_245 = None
        permute_default_51: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_22, [0, 2, 1]);  div_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_27: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 1536);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_28: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 1536);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_default_52: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_233, [0, 2, 1]);  view_233 = None
        permute_default_53: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_231, [0, 2, 1]);  view_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_default_54: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_223, [0, 2, 1]);  view_223 = None
        permute_default_55: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_20, [0, 2, 1]);  div_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_29: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 1536);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_30: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 1536);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_default_56: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_211, [0, 2, 1]);  view_211 = None
        permute_default_57: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_209, [0, 2, 1]);  view_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_default_58: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_201, [0, 2, 1]);  view_201 = None
        permute_default_59: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_18, [0, 2, 1]);  div_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_31: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 1536);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_32: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 1536);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_default_60: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None
        permute_default_61: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_187, [0, 2, 1]);  view_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_default_62: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_179, [0, 2, 1]);  view_179 = None
        permute_default_63: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_16, [0, 2, 1]);  div_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_33: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 1536);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_34: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 1536);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_default_64: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None
        permute_default_65: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_165, [0, 2, 1]);  view_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_default_66: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_157, [0, 2, 1]);  view_157 = None
        permute_default_67: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_14, [0, 2, 1]);  div_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_35: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 1536);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_36: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 1536);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_default_68: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None
        permute_default_69: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_143, [0, 2, 1]);  view_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_default_70: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_135, [0, 2, 1]);  view_135 = None
        permute_default_71: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_12, [0, 2, 1]);  div_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_37: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 1536);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_38: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 1536);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_default_72: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
        permute_default_73: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_121, [0, 2, 1]);  view_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_default_74: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_113, [0, 2, 1]);  view_113 = None
        permute_default_75: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_10, [0, 2, 1]);  div_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_39: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 1536);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_40: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 1536);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_default_76: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_101, [0, 2, 1]);  view_101 = None
        permute_default_77: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_default_78: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_91, [0, 2, 1]);  view_91 = None
        permute_default_79: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_8, [0, 2, 1]);  div_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_41: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 1536);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_42: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 1536);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_default_80: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
        permute_default_81: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_77, [0, 2, 1]);  view_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_default_82: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_69, [0, 2, 1]);  view_69 = None
        permute_default_83: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_6, [0, 2, 1]);  div_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_43: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 1536);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_44: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 1536);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_default_84: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None
        permute_default_85: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_55, [0, 2, 1]);  view_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_default_86: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_47, [0, 2, 1]);  view_47 = None
        permute_default_87: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_4, [0, 2, 1]);  div_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_45: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 1536);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_46: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 1536);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_default_88: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None
        permute_default_89: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_default_90: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_25, [0, 2, 1]);  view_25 = None
        permute_default_91: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_2, [0, 2, 1]);  div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_47: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 1536);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_48: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 1536);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        permute_default_92: "f32[192, 512, 512]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
        permute_default_93: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_default_94: "f32[192, 64, 512]" = torch.ops.aten.permute.default(view_3, [0, 2, 1]);  view_3 = None
        permute_default_95: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div, [0, 2, 1]);  div = None
        return (div_tensor, div_tensor_1, div_tensor_2, permute_default, permute_default_1, permute_default_2, permute_default_3, div_tensor_3, div_tensor_4, permute_default_4, permute_default_5, permute_default_6, permute_default_7, div_tensor_5, div_tensor_6, permute_default_8, permute_default_9, permute_default_10, permute_default_11, div_tensor_7, div_tensor_8, permute_default_12, permute_default_13, permute_default_14, permute_default_15, div_tensor_9, div_tensor_10, permute_default_16, permute_default_17, permute_default_18, permute_default_19, div_tensor_11, div_tensor_12, permute_default_20, permute_default_21, permute_default_22, permute_default_23, div_tensor_13, div_tensor_14, permute_default_24, permute_default_25, permute_default_26, permute_default_27, div_tensor_15, div_tensor_16, permute_default_28, permute_default_29, permute_default_30, permute_default_31, div_tensor_17, div_tensor_18, permute_default_32, permute_default_33, permute_default_34, permute_default_35, div_tensor_19, div_tensor_20, permute_default_36, permute_default_37, permute_default_38, permute_default_39, div_tensor_21, div_tensor_22, permute_default_40, permute_default_41, permute_default_42, permute_default_43, div_tensor_23, div_tensor_24, permute_default_44, permute_default_45, permute_default_46, permute_default_47, div_tensor_25, div_tensor_26, permute_default_48, permute_default_49, permute_default_50, permute_default_51, div_tensor_27, div_tensor_28, permute_default_52, permute_default_53, permute_default_54, permute_default_55, div_tensor_29, div_tensor_30, permute_default_56, permute_default_57, permute_default_58, permute_default_59, div_tensor_31, div_tensor_32, permute_default_60, permute_default_61, permute_default_62, permute_default_63, div_tensor_33, div_tensor_34, permute_default_64, permute_default_65, permute_default_66, permute_default_67, div_tensor_35, div_tensor_36, permute_default_68, permute_default_69, permute_default_70, permute_default_71, div_tensor_37, div_tensor_38, permute_default_72, permute_default_73, permute_default_74, permute_default_75, div_tensor_39, div_tensor_40, permute_default_76, permute_default_77, permute_default_78, permute_default_79, div_tensor_41, div_tensor_42, permute_default_80, permute_default_81, permute_default_82, permute_default_83, div_tensor_43, div_tensor_44, permute_default_84, permute_default_85, permute_default_86, permute_default_87, div_tensor_45, div_tensor_46, permute_default_88, permute_default_89, permute_default_90, permute_default_91, div_tensor_47, div_tensor_48, permute_default_92, permute_default_93, permute_default_94, permute_default_95)



def make_inputs():
    return [
    torch.randn([4096, 128100], dtype=torch.float32, device='cuda'),
    torch.randint(0, 512, [8, 512], dtype=torch.int64, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([192, 64, 512], [32768, 1, 64]),  # div_46
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([192, 64, 512], [32768, 1, 64]),  # div_44
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([192, 64, 512], [32768, 1, 64]),  # div_42
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([192, 64, 512], [32768, 1, 64]),  # div_40
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([192, 64, 512], [32768, 1, 64]),  # div_38
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([192, 64, 512], [32768, 1, 64]),  # div_36
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([192, 64, 512], [32768, 1, 64]),  # div_34
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([192, 64, 512], [32768, 1, 64]),  # div_32
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([192, 64, 512], [32768, 1, 64]),  # div_30
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([192, 64, 512], [32768, 1, 64]),  # div_28
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([192, 64, 512], [32768, 1, 64]),  # div_26
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([192, 64, 512], [32768, 1, 64]),  # div_24
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([192, 64, 512], [32768, 1, 64]),  # div_22
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([192, 64, 512], [32768, 1, 64]),  # div_20
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([192, 64, 512], [32768, 1, 64]),  # div_18
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([192, 64, 512], [32768, 1, 64]),  # div_16
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([192, 64, 512], [32768, 1, 64]),  # div_14
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([192, 64, 512], [32768, 1, 64]),  # div_12
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([192, 64, 512], [32768, 1, 64]),  # div_10
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([192, 64, 512], [32768, 1, 64]),  # div_8
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([192, 64, 512], [32768, 1, 64]),  # div_6
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([192, 64, 512], [32768, 1, 64]),  # div_4
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([192, 64, 512], [32768, 1, 64]),  # div_2
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([192, 64, 512], [32768, 1, 64]),  # div
    [8, 512, 128100],  # _shape_param_0
    [-1, 128100],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
