"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_training
Pattern hash: 68f22269ff50
Shape hash: 5c088640
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[1024, 30522]", _shape_param_0, primals_1119: "f32[30522]", _shape_param_1, _shape_param_2, primals_1120: "i64[8, 128]", full_default_2: "f32[]", view_964: "f32[1024, 512]", cat_1: "f32[512, 30522]", relu_95: "f32[8, 128, 512]", relu_94: "f32[8, 128, 512]", relu_93: "f32[8, 128, 512]", relu_92: "f32[8, 128, 512]", view_938: "f32[32, 128, 128]", view_939: "f32[32, 128, 32]", view_935: "f32[32, 128, 32]", view_936: "f32[32, 32, 128]", relu_91: "f32[8, 128, 512]", relu_90: "f32[8, 128, 512]", relu_89: "f32[8, 128, 512]", relu_88: "f32[8, 128, 512]", view_898: "f32[32, 128, 128]", view_899: "f32[32, 128, 32]", view_895: "f32[32, 128, 32]", view_896: "f32[32, 32, 128]", relu_87: "f32[8, 128, 512]", relu_86: "f32[8, 128, 512]", relu_85: "f32[8, 128, 512]", relu_84: "f32[8, 128, 512]", view_858: "f32[32, 128, 128]", view_859: "f32[32, 128, 32]", view_855: "f32[32, 128, 32]", view_856: "f32[32, 32, 128]", relu_83: "f32[8, 128, 512]", relu_82: "f32[8, 128, 512]", relu_81: "f32[8, 128, 512]", relu_80: "f32[8, 128, 512]", view_818: "f32[32, 128, 128]", view_819: "f32[32, 128, 32]", view_815: "f32[32, 128, 32]", view_816: "f32[32, 32, 128]", relu_79: "f32[8, 128, 512]", relu_78: "f32[8, 128, 512]", relu_77: "f32[8, 128, 512]", relu_76: "f32[8, 128, 512]", view_778: "f32[32, 128, 128]", view_779: "f32[32, 128, 32]", view_775: "f32[32, 128, 32]", view_776: "f32[32, 32, 128]", relu_75: "f32[8, 128, 512]", relu_74: "f32[8, 128, 512]", relu_73: "f32[8, 128, 512]", relu_72: "f32[8, 128, 512]", view_738: "f32[32, 128, 128]", view_739: "f32[32, 128, 32]", view_735: "f32[32, 128, 32]", view_736: "f32[32, 32, 128]", relu_71: "f32[8, 128, 512]", relu_70: "f32[8, 128, 512]", relu_69: "f32[8, 128, 512]", relu_68: "f32[8, 128, 512]", view_698: "f32[32, 128, 128]", view_699: "f32[32, 128, 32]", view_695: "f32[32, 128, 32]", view_696: "f32[32, 32, 128]", relu_67: "f32[8, 128, 512]", relu_66: "f32[8, 128, 512]", relu_65: "f32[8, 128, 512]", relu_64: "f32[8, 128, 512]", view_658: "f32[32, 128, 128]", view_659: "f32[32, 128, 32]", view_655: "f32[32, 128, 32]", view_656: "f32[32, 32, 128]", relu_63: "f32[8, 128, 512]", relu_62: "f32[8, 128, 512]", relu_61: "f32[8, 128, 512]", relu_60: "f32[8, 128, 512]", view_618: "f32[32, 128, 128]", view_619: "f32[32, 128, 32]", view_615: "f32[32, 128, 32]", view_616: "f32[32, 32, 128]", relu_59: "f32[8, 128, 512]", relu_58: "f32[8, 128, 512]", relu_57: "f32[8, 128, 512]", relu_56: "f32[8, 128, 512]", view_578: "f32[32, 128, 128]", view_579: "f32[32, 128, 32]", view_575: "f32[32, 128, 32]", view_576: "f32[32, 32, 128]", relu_55: "f32[8, 128, 512]", relu_54: "f32[8, 128, 512]", relu_53: "f32[8, 128, 512]", relu_52: "f32[8, 128, 512]", view_538: "f32[32, 128, 128]", view_539: "f32[32, 128, 32]", view_535: "f32[32, 128, 32]", view_536: "f32[32, 32, 128]", relu_51: "f32[8, 128, 512]", relu_50: "f32[8, 128, 512]", relu_49: "f32[8, 128, 512]", relu_48: "f32[8, 128, 512]", view_498: "f32[32, 128, 128]", view_499: "f32[32, 128, 32]", view_495: "f32[32, 128, 32]", view_496: "f32[32, 32, 128]", relu_47: "f32[8, 128, 512]", relu_46: "f32[8, 128, 512]", relu_45: "f32[8, 128, 512]", relu_44: "f32[8, 128, 512]", view_458: "f32[32, 128, 128]", view_459: "f32[32, 128, 32]", view_455: "f32[32, 128, 32]", view_456: "f32[32, 32, 128]", relu_43: "f32[8, 128, 512]", relu_42: "f32[8, 128, 512]", relu_41: "f32[8, 128, 512]", relu_40: "f32[8, 128, 512]", view_418: "f32[32, 128, 128]", view_419: "f32[32, 128, 32]", view_415: "f32[32, 128, 32]", view_416: "f32[32, 32, 128]", relu_39: "f32[8, 128, 512]", relu_38: "f32[8, 128, 512]", relu_37: "f32[8, 128, 512]", relu_36: "f32[8, 128, 512]", view_378: "f32[32, 128, 128]", view_379: "f32[32, 128, 32]", view_375: "f32[32, 128, 32]", view_376: "f32[32, 32, 128]", relu_35: "f32[8, 128, 512]", relu_34: "f32[8, 128, 512]", relu_33: "f32[8, 128, 512]", relu_32: "f32[8, 128, 512]", view_338: "f32[32, 128, 128]", view_339: "f32[32, 128, 32]", view_335: "f32[32, 128, 32]", view_336: "f32[32, 32, 128]", relu_31: "f32[8, 128, 512]", relu_30: "f32[8, 128, 512]", relu_29: "f32[8, 128, 512]", relu_28: "f32[8, 128, 512]", view_298: "f32[32, 128, 128]", view_299: "f32[32, 128, 32]", view_295: "f32[32, 128, 32]", view_296: "f32[32, 32, 128]", relu_27: "f32[8, 128, 512]", relu_26: "f32[8, 128, 512]", relu_25: "f32[8, 128, 512]", relu_24: "f32[8, 128, 512]", view_258: "f32[32, 128, 128]", view_259: "f32[32, 128, 32]", view_255: "f32[32, 128, 32]", view_256: "f32[32, 32, 128]", relu_23: "f32[8, 128, 512]", relu_22: "f32[8, 128, 512]", relu_21: "f32[8, 128, 512]", relu_20: "f32[8, 128, 512]", view_218: "f32[32, 128, 128]", view_219: "f32[32, 128, 32]", view_215: "f32[32, 128, 32]", view_216: "f32[32, 32, 128]", relu_19: "f32[8, 128, 512]", relu_18: "f32[8, 128, 512]", relu_17: "f32[8, 128, 512]", relu_16: "f32[8, 128, 512]", view_178: "f32[32, 128, 128]", view_179: "f32[32, 128, 32]", view_175: "f32[32, 128, 32]", view_176: "f32[32, 32, 128]", relu_15: "f32[8, 128, 512]", relu_14: "f32[8, 128, 512]", relu_13: "f32[8, 128, 512]", relu_12: "f32[8, 128, 512]", view_138: "f32[32, 128, 128]", view_139: "f32[32, 128, 32]", view_135: "f32[32, 128, 32]", view_136: "f32[32, 32, 128]", relu_11: "f32[8, 128, 512]", relu_10: "f32[8, 128, 512]", relu_9: "f32[8, 128, 512]", relu_8: "f32[8, 128, 512]", view_98: "f32[32, 128, 128]", view_99: "f32[32, 128, 32]", view_95: "f32[32, 128, 32]", view_96: "f32[32, 32, 128]", relu_7: "f32[8, 128, 512]", relu_6: "f32[8, 128, 512]", relu_5: "f32[8, 128, 512]", relu_4: "f32[8, 128, 512]", view_58: "f32[32, 128, 128]", view_59: "f32[32, 128, 32]", view_55: "f32[32, 128, 32]", view_56: "f32[32, 32, 128]", relu_3: "f32[8, 128, 512]", relu_2: "f32[8, 128, 512]", relu_1: "f32[8, 128, 512]", relu: "f32[8, 128, 512]", view_18: "f32[32, 128, 128]", view_19: "f32[32, 128, 32]", view_15: "f32[32, 128, 32]", view_16: "f32[32, 32, 128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:507 in forward, code: hidden_states = hidden_states.matmul(torch.cat([self.decoder.weight.t(), self.dense.weight], dim=0))
        reshape_default: "f32[8, 128, 30522]" = torch.ops.aten.reshape.default(mm, _shape_param_0);  mm = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:508 in forward, code: hidden_states += self.decoder.bias
        add_tensor: "f32[8, 128, 30522]" = torch.ops.aten.add.Tensor(reshape_default, primals_1119);  reshape_default = primals_1119 = None
        reshape_default_1: "f32[1024, 30522]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None
        reshape_default_2: "f32[8, 128, 30522]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:825 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        reshape_default_3: "i64[1024]" = torch.ops.aten.reshape.default(primals_1120, [-1]);  primals_1120 = None
        amax_default: "f32[1024, 1]" = torch.ops.aten.amax.default(reshape_default_1, [1], True)
        sub_tensor: "f32[1024, 30522]" = torch.ops.aten.sub.Tensor(reshape_default_1, amax_default);  reshape_default_1 = amax_default = None
        exp_default: "f32[1024, 30522]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[1024, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[1024, 30522]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        ne_scalar: "b8[1024]" = torch.ops.aten.ne.Scalar(reshape_default_3, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[1024]" = torch.ops.aten.where.self(ne_scalar, reshape_default_3, full_default);  reshape_default_3 = full_default = None
        unsqueeze_default: "i64[1024, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "f32[1024, 1]" = torch.ops.aten.gather.default(sub_tensor_1, 1, unsqueeze_default);  unsqueeze_default = None
        squeeze_dim: "f32[1024]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "f32[1024]" = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        where_self_1: "f32[1024]" = torch.ops.aten.where.self(ne_scalar, neg_default, full_default_2);  neg_default = full_default_2 = None
        sum_default: "i64[]" = torch.ops.aten.sum.default(ne_scalar);  ne_scalar = None
        convert_element_type_default: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default, torch.float32);  sum_default = None
        sum_default_1: "f32[]" = torch.ops.aten.sum.default(where_self_1);  where_self_1 = None
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(sum_default_1, convert_element_type_default);  sum_default_1 = convert_element_type_default = None
        exp_default_1: "f32[1024, 30522]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:507 in forward, code: hidden_states = hidden_states.matmul(torch.cat([self.decoder.weight.t(), self.dense.weight], dim=0))
        permute_default: "f32[512, 1024]" = torch.ops.aten.permute.default(view_964, [1, 0]);  view_964 = None
        permute_default_1: "f32[30522, 512]" = torch.ops.aten.permute.default(cat_1, [1, 0]);  cat_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_scalar: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_95, 0);  relu_95 = None
        le_scalar_1: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_94, 0);  relu_94 = None
        le_scalar_2: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_93, 0);  relu_93 = None
        le_scalar_3: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_92, 0);  relu_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_2: "f32[32, 128, 128]" = torch.ops.aten.permute.default(view_938, [0, 2, 1]);  view_938 = None
        permute_default_3: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_939, [0, 2, 1]);  view_939 = None
        permute_default_4: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_935, [0, 2, 1]);  view_935 = None
        permute_default_5: "f32[32, 128, 32]" = torch.ops.aten.permute.default(view_936, [0, 2, 1]);  view_936 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_scalar_4: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_91, 0);  relu_91 = None
        le_scalar_5: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_90, 0);  relu_90 = None
        le_scalar_6: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_89, 0);  relu_89 = None
        le_scalar_7: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_88, 0);  relu_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_6: "f32[32, 128, 128]" = torch.ops.aten.permute.default(view_898, [0, 2, 1]);  view_898 = None
        permute_default_7: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_899, [0, 2, 1]);  view_899 = None
        permute_default_8: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_895, [0, 2, 1]);  view_895 = None
        permute_default_9: "f32[32, 128, 32]" = torch.ops.aten.permute.default(view_896, [0, 2, 1]);  view_896 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_scalar_8: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_87, 0);  relu_87 = None
        le_scalar_9: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_86, 0);  relu_86 = None
        le_scalar_10: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_85, 0);  relu_85 = None
        le_scalar_11: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_84, 0);  relu_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_10: "f32[32, 128, 128]" = torch.ops.aten.permute.default(view_858, [0, 2, 1]);  view_858 = None
        permute_default_11: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_859, [0, 2, 1]);  view_859 = None
        permute_default_12: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_855, [0, 2, 1]);  view_855 = None
        permute_default_13: "f32[32, 128, 32]" = torch.ops.aten.permute.default(view_856, [0, 2, 1]);  view_856 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_scalar_12: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_83, 0);  relu_83 = None
        le_scalar_13: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_82, 0);  relu_82 = None
        le_scalar_14: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_81, 0);  relu_81 = None
        le_scalar_15: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_80, 0);  relu_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_14: "f32[32, 128, 128]" = torch.ops.aten.permute.default(view_818, [0, 2, 1]);  view_818 = None
        permute_default_15: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_819, [0, 2, 1]);  view_819 = None
        permute_default_16: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_815, [0, 2, 1]);  view_815 = None
        permute_default_17: "f32[32, 128, 32]" = torch.ops.aten.permute.default(view_816, [0, 2, 1]);  view_816 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_scalar_16: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_79, 0);  relu_79 = None
        le_scalar_17: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_78, 0);  relu_78 = None
        le_scalar_18: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_77, 0);  relu_77 = None
        le_scalar_19: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_76, 0);  relu_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_18: "f32[32, 128, 128]" = torch.ops.aten.permute.default(view_778, [0, 2, 1]);  view_778 = None
        permute_default_19: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_779, [0, 2, 1]);  view_779 = None
        permute_default_20: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_775, [0, 2, 1]);  view_775 = None
        permute_default_21: "f32[32, 128, 32]" = torch.ops.aten.permute.default(view_776, [0, 2, 1]);  view_776 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_scalar_20: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_75, 0);  relu_75 = None
        le_scalar_21: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_74, 0);  relu_74 = None
        le_scalar_22: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_73, 0);  relu_73 = None
        le_scalar_23: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_72, 0);  relu_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_22: "f32[32, 128, 128]" = torch.ops.aten.permute.default(view_738, [0, 2, 1]);  view_738 = None
        permute_default_23: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_739, [0, 2, 1]);  view_739 = None
        permute_default_24: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_735, [0, 2, 1]);  view_735 = None
        permute_default_25: "f32[32, 128, 32]" = torch.ops.aten.permute.default(view_736, [0, 2, 1]);  view_736 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_scalar_24: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_71, 0);  relu_71 = None
        le_scalar_25: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_70, 0);  relu_70 = None
        le_scalar_26: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_69, 0);  relu_69 = None
        le_scalar_27: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_68, 0);  relu_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_26: "f32[32, 128, 128]" = torch.ops.aten.permute.default(view_698, [0, 2, 1]);  view_698 = None
        permute_default_27: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_699, [0, 2, 1]);  view_699 = None
        permute_default_28: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_695, [0, 2, 1]);  view_695 = None
        permute_default_29: "f32[32, 128, 32]" = torch.ops.aten.permute.default(view_696, [0, 2, 1]);  view_696 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_scalar_28: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_67, 0);  relu_67 = None
        le_scalar_29: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_66, 0);  relu_66 = None
        le_scalar_30: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_65, 0);  relu_65 = None
        le_scalar_31: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_64, 0);  relu_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_30: "f32[32, 128, 128]" = torch.ops.aten.permute.default(view_658, [0, 2, 1]);  view_658 = None
        permute_default_31: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_659, [0, 2, 1]);  view_659 = None
        permute_default_32: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_655, [0, 2, 1]);  view_655 = None
        permute_default_33: "f32[32, 128, 32]" = torch.ops.aten.permute.default(view_656, [0, 2, 1]);  view_656 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_scalar_32: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_63, 0);  relu_63 = None
        le_scalar_33: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_62, 0);  relu_62 = None
        le_scalar_34: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_61, 0);  relu_61 = None
        le_scalar_35: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_60, 0);  relu_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_34: "f32[32, 128, 128]" = torch.ops.aten.permute.default(view_618, [0, 2, 1]);  view_618 = None
        permute_default_35: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_619, [0, 2, 1]);  view_619 = None
        permute_default_36: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_615, [0, 2, 1]);  view_615 = None
        permute_default_37: "f32[32, 128, 32]" = torch.ops.aten.permute.default(view_616, [0, 2, 1]);  view_616 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_scalar_36: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_59, 0);  relu_59 = None
        le_scalar_37: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_58, 0);  relu_58 = None
        le_scalar_38: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_57, 0);  relu_57 = None
        le_scalar_39: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_56, 0);  relu_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_38: "f32[32, 128, 128]" = torch.ops.aten.permute.default(view_578, [0, 2, 1]);  view_578 = None
        permute_default_39: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_579, [0, 2, 1]);  view_579 = None
        permute_default_40: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_575, [0, 2, 1]);  view_575 = None
        permute_default_41: "f32[32, 128, 32]" = torch.ops.aten.permute.default(view_576, [0, 2, 1]);  view_576 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_scalar_40: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_55, 0);  relu_55 = None
        le_scalar_41: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_54, 0);  relu_54 = None
        le_scalar_42: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_53, 0);  relu_53 = None
        le_scalar_43: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_52, 0);  relu_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_42: "f32[32, 128, 128]" = torch.ops.aten.permute.default(view_538, [0, 2, 1]);  view_538 = None
        permute_default_43: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_539, [0, 2, 1]);  view_539 = None
        permute_default_44: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_535, [0, 2, 1]);  view_535 = None
        permute_default_45: "f32[32, 128, 32]" = torch.ops.aten.permute.default(view_536, [0, 2, 1]);  view_536 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_scalar_44: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_51, 0);  relu_51 = None
        le_scalar_45: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_50, 0);  relu_50 = None
        le_scalar_46: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_49, 0);  relu_49 = None
        le_scalar_47: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_48, 0);  relu_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_46: "f32[32, 128, 128]" = torch.ops.aten.permute.default(view_498, [0, 2, 1]);  view_498 = None
        permute_default_47: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_499, [0, 2, 1]);  view_499 = None
        permute_default_48: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_495, [0, 2, 1]);  view_495 = None
        permute_default_49: "f32[32, 128, 32]" = torch.ops.aten.permute.default(view_496, [0, 2, 1]);  view_496 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_scalar_48: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_47, 0);  relu_47 = None
        le_scalar_49: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_46, 0);  relu_46 = None
        le_scalar_50: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_45, 0);  relu_45 = None
        le_scalar_51: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_44, 0);  relu_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_50: "f32[32, 128, 128]" = torch.ops.aten.permute.default(view_458, [0, 2, 1]);  view_458 = None
        permute_default_51: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_459, [0, 2, 1]);  view_459 = None
        permute_default_52: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_455, [0, 2, 1]);  view_455 = None
        permute_default_53: "f32[32, 128, 32]" = torch.ops.aten.permute.default(view_456, [0, 2, 1]);  view_456 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_scalar_52: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_43, 0);  relu_43 = None
        le_scalar_53: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_42, 0);  relu_42 = None
        le_scalar_54: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_41, 0);  relu_41 = None
        le_scalar_55: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_40, 0);  relu_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_54: "f32[32, 128, 128]" = torch.ops.aten.permute.default(view_418, [0, 2, 1]);  view_418 = None
        permute_default_55: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_419, [0, 2, 1]);  view_419 = None
        permute_default_56: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_415, [0, 2, 1]);  view_415 = None
        permute_default_57: "f32[32, 128, 32]" = torch.ops.aten.permute.default(view_416, [0, 2, 1]);  view_416 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_scalar_56: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_39, 0);  relu_39 = None
        le_scalar_57: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_38, 0);  relu_38 = None
        le_scalar_58: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_37, 0);  relu_37 = None
        le_scalar_59: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_36, 0);  relu_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_58: "f32[32, 128, 128]" = torch.ops.aten.permute.default(view_378, [0, 2, 1]);  view_378 = None
        permute_default_59: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_379, [0, 2, 1]);  view_379 = None
        permute_default_60: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_375, [0, 2, 1]);  view_375 = None
        permute_default_61: "f32[32, 128, 32]" = torch.ops.aten.permute.default(view_376, [0, 2, 1]);  view_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_scalar_60: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_35, 0);  relu_35 = None
        le_scalar_61: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_34, 0);  relu_34 = None
        le_scalar_62: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_33, 0);  relu_33 = None
        le_scalar_63: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_32, 0);  relu_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_62: "f32[32, 128, 128]" = torch.ops.aten.permute.default(view_338, [0, 2, 1]);  view_338 = None
        permute_default_63: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_339, [0, 2, 1]);  view_339 = None
        permute_default_64: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_335, [0, 2, 1]);  view_335 = None
        permute_default_65: "f32[32, 128, 32]" = torch.ops.aten.permute.default(view_336, [0, 2, 1]);  view_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_scalar_64: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_31, 0);  relu_31 = None
        le_scalar_65: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_30, 0);  relu_30 = None
        le_scalar_66: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_29, 0);  relu_29 = None
        le_scalar_67: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_28, 0);  relu_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_66: "f32[32, 128, 128]" = torch.ops.aten.permute.default(view_298, [0, 2, 1]);  view_298 = None
        permute_default_67: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_299, [0, 2, 1]);  view_299 = None
        permute_default_68: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_295, [0, 2, 1]);  view_295 = None
        permute_default_69: "f32[32, 128, 32]" = torch.ops.aten.permute.default(view_296, [0, 2, 1]);  view_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_scalar_68: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_27, 0);  relu_27 = None
        le_scalar_69: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_26, 0);  relu_26 = None
        le_scalar_70: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_25, 0);  relu_25 = None
        le_scalar_71: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_24, 0);  relu_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_70: "f32[32, 128, 128]" = torch.ops.aten.permute.default(view_258, [0, 2, 1]);  view_258 = None
        permute_default_71: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_259, [0, 2, 1]);  view_259 = None
        permute_default_72: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_255, [0, 2, 1]);  view_255 = None
        permute_default_73: "f32[32, 128, 32]" = torch.ops.aten.permute.default(view_256, [0, 2, 1]);  view_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_scalar_72: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_23, 0);  relu_23 = None
        le_scalar_73: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_22, 0);  relu_22 = None
        le_scalar_74: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
        le_scalar_75: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_74: "f32[32, 128, 128]" = torch.ops.aten.permute.default(view_218, [0, 2, 1]);  view_218 = None
        permute_default_75: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_219, [0, 2, 1]);  view_219 = None
        permute_default_76: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_215, [0, 2, 1]);  view_215 = None
        permute_default_77: "f32[32, 128, 32]" = torch.ops.aten.permute.default(view_216, [0, 2, 1]);  view_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_scalar_76: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None
        le_scalar_77: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
        le_scalar_78: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
        le_scalar_79: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_78: "f32[32, 128, 128]" = torch.ops.aten.permute.default(view_178, [0, 2, 1]);  view_178 = None
        permute_default_79: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_179, [0, 2, 1]);  view_179 = None
        permute_default_80: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_175, [0, 2, 1]);  view_175 = None
        permute_default_81: "f32[32, 128, 32]" = torch.ops.aten.permute.default(view_176, [0, 2, 1]);  view_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_scalar_80: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
        le_scalar_81: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
        le_scalar_82: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
        le_scalar_83: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_82: "f32[32, 128, 128]" = torch.ops.aten.permute.default(view_138, [0, 2, 1]);  view_138 = None
        permute_default_83: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_139, [0, 2, 1]);  view_139 = None
        permute_default_84: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_135, [0, 2, 1]);  view_135 = None
        permute_default_85: "f32[32, 128, 32]" = torch.ops.aten.permute.default(view_136, [0, 2, 1]);  view_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_scalar_84: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        le_scalar_85: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        le_scalar_86: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        le_scalar_87: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_86: "f32[32, 128, 128]" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None
        permute_default_87: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None
        permute_default_88: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_95, [0, 2, 1]);  view_95 = None
        permute_default_89: "f32[32, 128, 32]" = torch.ops.aten.permute.default(view_96, [0, 2, 1]);  view_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_scalar_88: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        le_scalar_89: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        le_scalar_90: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        le_scalar_91: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_90: "f32[32, 128, 128]" = torch.ops.aten.permute.default(view_58, [0, 2, 1]);  view_58 = None
        permute_default_91: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_59, [0, 2, 1]);  view_59 = None
        permute_default_92: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_55, [0, 2, 1]);  view_55 = None
        permute_default_93: "f32[32, 128, 32]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_scalar_92: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        le_scalar_93: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        le_scalar_94: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        le_scalar_95: "b8[8, 128, 512]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_94: "f32[32, 128, 128]" = torch.ops.aten.permute.default(view_18, [0, 2, 1]);  view_18 = None
        permute_default_95: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_19, [0, 2, 1]);  view_19 = None
        permute_default_96: "f32[32, 32, 128]" = torch.ops.aten.permute.default(view_15, [0, 2, 1]);  view_15 = None
        permute_default_97: "f32[32, 128, 32]" = torch.ops.aten.permute.default(view_16, [0, 2, 1]);  view_16 = None
        return (reshape_default_2, div_tensor, exp_default_1, permute_default, permute_default_1, le_scalar, le_scalar_1, le_scalar_2, le_scalar_3, permute_default_2, permute_default_3, permute_default_4, permute_default_5, le_scalar_4, le_scalar_5, le_scalar_6, le_scalar_7, permute_default_6, permute_default_7, permute_default_8, permute_default_9, le_scalar_8, le_scalar_9, le_scalar_10, le_scalar_11, permute_default_10, permute_default_11, permute_default_12, permute_default_13, le_scalar_12, le_scalar_13, le_scalar_14, le_scalar_15, permute_default_14, permute_default_15, permute_default_16, permute_default_17, le_scalar_16, le_scalar_17, le_scalar_18, le_scalar_19, permute_default_18, permute_default_19, permute_default_20, permute_default_21, le_scalar_20, le_scalar_21, le_scalar_22, le_scalar_23, permute_default_22, permute_default_23, permute_default_24, permute_default_25, le_scalar_24, le_scalar_25, le_scalar_26, le_scalar_27, permute_default_26, permute_default_27, permute_default_28, permute_default_29, le_scalar_28, le_scalar_29, le_scalar_30, le_scalar_31, permute_default_30, permute_default_31, permute_default_32, permute_default_33, le_scalar_32, le_scalar_33, le_scalar_34, le_scalar_35, permute_default_34, permute_default_35, permute_default_36, permute_default_37, le_scalar_36, le_scalar_37, le_scalar_38, le_scalar_39, permute_default_38, permute_default_39, permute_default_40, permute_default_41, le_scalar_40, le_scalar_41, le_scalar_42, le_scalar_43, permute_default_42, permute_default_43, permute_default_44, permute_default_45, le_scalar_44, le_scalar_45, le_scalar_46, le_scalar_47, permute_default_46, permute_default_47, permute_default_48, permute_default_49, le_scalar_48, le_scalar_49, le_scalar_50, le_scalar_51, permute_default_50, permute_default_51, permute_default_52, permute_default_53, le_scalar_52, le_scalar_53, le_scalar_54, le_scalar_55, permute_default_54, permute_default_55, permute_default_56, permute_default_57, le_scalar_56, le_scalar_57, le_scalar_58, le_scalar_59, permute_default_58, permute_default_59, permute_default_60, permute_default_61, le_scalar_60, le_scalar_61, le_scalar_62, le_scalar_63, permute_default_62, permute_default_63, permute_default_64, permute_default_65, le_scalar_64, le_scalar_65, le_scalar_66, le_scalar_67, permute_default_66, permute_default_67, permute_default_68, permute_default_69, le_scalar_68, le_scalar_69, le_scalar_70, le_scalar_71, permute_default_70, permute_default_71, permute_default_72, permute_default_73, le_scalar_72, le_scalar_73, le_scalar_74, le_scalar_75, permute_default_74, permute_default_75, permute_default_76, permute_default_77, le_scalar_76, le_scalar_77, le_scalar_78, le_scalar_79, permute_default_78, permute_default_79, permute_default_80, permute_default_81, le_scalar_80, le_scalar_81, le_scalar_82, le_scalar_83, permute_default_82, permute_default_83, permute_default_84, permute_default_85, le_scalar_84, le_scalar_85, le_scalar_86, le_scalar_87, permute_default_86, permute_default_87, permute_default_88, permute_default_89, le_scalar_88, le_scalar_89, le_scalar_90, le_scalar_91, permute_default_90, permute_default_91, permute_default_92, permute_default_93, le_scalar_92, le_scalar_93, le_scalar_94, le_scalar_95, permute_default_94, permute_default_95, permute_default_96, permute_default_97)


def _default_make_inputs():
    return [
    torch.randn([1024, 30522], dtype=torch.float32, device='cuda'),
    [8, 128, 30522],  # _shape_param_0
    torch.randn([30522], dtype=torch.float32, device='cuda'),
    [1024, 30522],  # _shape_param_1
    [8, 128, 30522],  # _shape_param_2
    torch.randint(0, 2, [8, 128], dtype=torch.int64, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 30522], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 32, 128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 32, 128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 32, 128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 32, 128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 32, 128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 32, 128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 32, 128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 32, 128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 32, 128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 32, 128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 32, 128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 32, 128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 32, 128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 32, 128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 32, 128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 32, 128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 32, 128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 32, 128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 32, 128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 32, 128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 32, 128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 32, 128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 32, 128], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32, 32, 128], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
