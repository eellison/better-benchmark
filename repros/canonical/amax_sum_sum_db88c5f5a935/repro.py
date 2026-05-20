"""
Standalone repro captured via capture_hook.
Label: hf_DistilBertForMaskedLM_train
Pattern hash: db88c5f5a935
Shape hash: 95b21b0e
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([32768, 30524], f32), T([256, 128], i64, gen=Index(256)), T([], f32), T([256, 128, 1], f32), T([256, 128, 1], f32), T([3072, 128, 128], f32), T([3072, 128, 64], f32), T([3072, 128, 64], f32), T([3072, 64, 128], f32), T([256, 128, 1], f32), T([256, 128, 1], f32), T([3072, 128, 128], f32), T([3072, 128, 64], f32), T([3072, 128, 64], f32), T([3072, 64, 128], f32), T([256, 128, 1], f32), T([256, 128, 1], f32), T([3072, 128, 128], f32), T([3072, 128, 64], f32), T([3072, 128, 64], f32), T([3072, 64, 128], f32), T([256, 128, 1], f32), T([256, 128, 1], f32), T([3072, 128, 128], f32), T([3072, 128, 64], f32), T([3072, 128, 64], f32), T([3072, 64, 128], f32), T([256, 128, 1], f32), T([256, 128, 1], f32), T([3072, 128, 128], f32), T([3072, 128, 64], f32), T([3072, 128, 64], f32), T([3072, 64, 128], f32), T([256, 128, 1], f32), T([256, 128, 1], f32), T([3072, 128, 128], f32), T([3072, 128, 64], f32), T([3072, 128, 64], f32), T([3072, 64, 128], f32), S([256, 128, 30522]), S([-1, 30522]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_default: "f32[32768, 30524]", primals_108: "i64[256, 128]", full_default_1: "f32[]", rsqrt_12: "f32[256, 128, 1]", rsqrt_11: "f32[256, 128, 1]", view_122: "f32[3072, 128, 128]", view_123: "f32[3072, 128, 64]", view_119: "f32[3072, 128, 64]", view_120: "f32[3072, 64, 128]", rsqrt_10: "f32[256, 128, 1]", rsqrt_9: "f32[256, 128, 1]", view_100: "f32[3072, 128, 128]", view_101: "f32[3072, 128, 64]", view_97: "f32[3072, 128, 64]", view_98: "f32[3072, 64, 128]", rsqrt_8: "f32[256, 128, 1]", rsqrt_7: "f32[256, 128, 1]", view_78: "f32[3072, 128, 128]", view_79: "f32[3072, 128, 64]", view_75: "f32[3072, 128, 64]", view_76: "f32[3072, 64, 128]", rsqrt_6: "f32[256, 128, 1]", rsqrt_5: "f32[256, 128, 1]", view_56: "f32[3072, 128, 128]", view_57: "f32[3072, 128, 64]", view_53: "f32[3072, 128, 64]", view_54: "f32[3072, 64, 128]", rsqrt_4: "f32[256, 128, 1]", rsqrt_3: "f32[256, 128, 1]", view_34: "f32[3072, 128, 128]", view_35: "f32[3072, 128, 64]", view_31: "f32[3072, 128, 64]", view_32: "f32[3072, 64, 128]", rsqrt_2: "f32[256, 128, 1]", rsqrt_1: "f32[256, 128, 1]", view_12: "f32[3072, 128, 128]", view_13: "f32[3072, 128, 64]", view_9: "f32[3072, 128, 64]", view_10: "f32[3072, 64, 128]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:517 in forward, code: prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
        slice_tensor: "f32[32768, 30522]" = torch.ops.aten.slice.Tensor(addmm_default, 1, 0, -2);  addmm_default = None
        reshape_default: "f32[256, 128, 30522]" = torch.ops.aten.reshape.default(slice_tensor, _shape_param_0);  slice_tensor = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:521 in forward, code: mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))
        reshape_default_1: "f32[32768, 30522]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        reshape_default_2: "i64[32768]" = torch.ops.aten.reshape.default(primals_108, [-1]);  primals_108 = None
        amax_default: "f32[32768, 1]" = torch.ops.aten.amax.default(reshape_default_1, [1], True)
        sub_tensor: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(reshape_default_1, amax_default);  reshape_default_1 = amax_default = None
        exp_default: "f32[32768, 30522]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[32768, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[32768, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        ne_scalar: "b8[32768]" = torch.ops.aten.ne.Scalar(reshape_default_2, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[32768]" = torch.ops.aten.where.self(ne_scalar, reshape_default_2, full_default);  reshape_default_2 = full_default = None
        unsqueeze_default: "i64[32768, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "f32[32768, 1]" = torch.ops.aten.gather.default(sub_tensor_1, 1, unsqueeze_default);  sub_tensor_1 = unsqueeze_default = None
        squeeze_dim: "f32[32768]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "f32[32768]" = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        where_self_1: "f32[32768]" = torch.ops.aten.where.self(ne_scalar, neg_default, full_default_1);  neg_default = full_default_1 = None
        sum_default: "i64[]" = torch.ops.aten.sum.default(ne_scalar);  ne_scalar = None
        convert_element_type_default: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default, torch.float32);  sum_default = None
        sum_default_1: "f32[]" = torch.ops.aten.sum.default(where_self_1);  where_self_1 = None
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(sum_default_1, convert_element_type_default);  sum_default_1 = convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        div_tensor_1: "f32[256, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        div_tensor_2: "f32[256, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default: "f32[3072, 128, 128]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
        permute_default_1: "f32[3072, 64, 128]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
        permute_default_2: "f32[3072, 64, 128]" = torch.ops.aten.permute.default(view_119, [0, 2, 1]);  view_119 = None
        permute_default_3: "f32[3072, 128, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1]);  view_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        div_tensor_3: "f32[256, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        div_tensor_4: "f32[256, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_4: "f32[3072, 128, 128]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
        permute_default_5: "f32[3072, 64, 128]" = torch.ops.aten.permute.default(view_101, [0, 2, 1]);  view_101 = None
        permute_default_6: "f32[3072, 64, 128]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
        permute_default_7: "f32[3072, 128, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        div_tensor_5: "f32[256, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        div_tensor_6: "f32[256, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_8: "f32[3072, 128, 128]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
        permute_default_9: "f32[3072, 64, 128]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
        permute_default_10: "f32[3072, 64, 128]" = torch.ops.aten.permute.default(view_75, [0, 2, 1]);  view_75 = None
        permute_default_11: "f32[3072, 128, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1]);  view_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        div_tensor_7: "f32[256, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        div_tensor_8: "f32[256, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_12: "f32[3072, 128, 128]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
        permute_default_13: "f32[3072, 64, 128]" = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None
        permute_default_14: "f32[3072, 64, 128]" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
        permute_default_15: "f32[3072, 128, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1]);  view_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        div_tensor_9: "f32[256, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        div_tensor_10: "f32[256, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_16: "f32[3072, 128, 128]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
        permute_default_17: "f32[3072, 64, 128]" = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None
        permute_default_18: "f32[3072, 64, 128]" = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
        permute_default_19: "f32[3072, 128, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        div_tensor_11: "f32[256, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        div_tensor_12: "f32[256, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_20: "f32[3072, 128, 128]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        permute_default_21: "f32[3072, 64, 128]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
        permute_default_22: "f32[3072, 64, 128]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_default_23: "f32[3072, 128, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
        return (div_tensor, div_tensor_1, div_tensor_2, permute_default, permute_default_1, permute_default_2, permute_default_3, div_tensor_3, div_tensor_4, permute_default_4, permute_default_5, permute_default_6, permute_default_7, div_tensor_5, div_tensor_6, permute_default_8, permute_default_9, permute_default_10, permute_default_11, div_tensor_7, div_tensor_8, permute_default_12, permute_default_13, permute_default_14, permute_default_15, div_tensor_9, div_tensor_10, permute_default_16, permute_default_17, permute_default_18, permute_default_19, div_tensor_11, div_tensor_12, permute_default_20, permute_default_21, permute_default_22, permute_default_23)


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
