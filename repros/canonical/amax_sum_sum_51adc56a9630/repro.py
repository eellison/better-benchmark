"""
Standalone repro captured via capture_hook.
Label: hf_MegatronBertForCausalLM_train
Pattern hash: 51adc56a9630
Shape hash: 5fad5bf5
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_145: "f32[8192, 29056]", _shape_param_0, primals_1: "i64[16, 512]", _shape_param_1, rsqrt_48: "f32[16, 512, 1]", rsqrt_47: "f32[16, 512, 1]", rsqrt_46: "f32[16, 512, 1]", rsqrt_45: "f32[16, 512, 1]", rsqrt_44: "f32[16, 512, 1]", rsqrt_43: "f32[16, 512, 1]", rsqrt_42: "f32[16, 512, 1]", rsqrt_41: "f32[16, 512, 1]", rsqrt_40: "f32[16, 512, 1]", rsqrt_39: "f32[16, 512, 1]", rsqrt_38: "f32[16, 512, 1]", rsqrt_37: "f32[16, 512, 1]", rsqrt_36: "f32[16, 512, 1]", rsqrt_35: "f32[16, 512, 1]", rsqrt_34: "f32[16, 512, 1]", rsqrt_33: "f32[16, 512, 1]", rsqrt_32: "f32[16, 512, 1]", rsqrt_31: "f32[16, 512, 1]", rsqrt_30: "f32[16, 512, 1]", rsqrt_29: "f32[16, 512, 1]", rsqrt_28: "f32[16, 512, 1]", rsqrt_27: "f32[16, 512, 1]", rsqrt_26: "f32[16, 512, 1]", rsqrt_25: "f32[16, 512, 1]", rsqrt_24: "f32[16, 512, 1]", rsqrt_23: "f32[16, 512, 1]", rsqrt_22: "f32[16, 512, 1]", rsqrt_21: "f32[16, 512, 1]", rsqrt_20: "f32[16, 512, 1]", rsqrt_19: "f32[16, 512, 1]", rsqrt_18: "f32[16, 512, 1]", rsqrt_17: "f32[16, 512, 1]", rsqrt_16: "f32[16, 512, 1]", rsqrt_15: "f32[16, 512, 1]", rsqrt_14: "f32[16, 512, 1]", rsqrt_13: "f32[16, 512, 1]", rsqrt_12: "f32[16, 512, 1]", rsqrt_11: "f32[16, 512, 1]", rsqrt_10: "f32[16, 512, 1]", rsqrt_9: "f32[16, 512, 1]", rsqrt_8: "f32[16, 512, 1]", rsqrt_7: "f32[16, 512, 1]", rsqrt_6: "f32[16, 512, 1]", rsqrt_5: "f32[16, 512, 1]", rsqrt_4: "f32[16, 512, 1]", rsqrt_3: "f32[16, 512, 1]", rsqrt_2: "f32[16, 512, 1]", rsqrt_1: "f32[16, 512, 1]", rsqrt: "f32[16, 512, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:465 in forward, code: hidden_states = self.decoder(hidden_states)
        reshape_default: "f32[16, 512, 29056]" = torch.ops.aten.reshape.default(addmm_145, _shape_param_0);  addmm_145 = _shape_param_0 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default: "i64[16, 513]" = torch.ops.aten.constant_pad_nd.default(primals_1, [0, 1], -100.0);  primals_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:61 in ForCausalLMLoss, code: shift_labels = labels[..., 1:].contiguous()
        slice_tensor: "i64[16, 512]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default, 1, 1, 9223372036854775807);  constant_pad_nd_default = None
        clone_default: "i64[16, 512]" = torch.ops.aten.clone.default(slice_tensor, memory_format = torch.contiguous_format);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        reshape_default_1: "f32[8192, 29056]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:65 in ForCausalLMLoss, code: shift_labels = shift_labels.view(-1)
        reshape_default_2: "i64[8192]" = torch.ops.aten.reshape.default(clone_default, [-1]);  clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        amax_default: "f32[8192, 1]" = torch.ops.aten.amax.default(reshape_default_1, [1], True)
        sub_tensor: "f32[8192, 29056]" = torch.ops.aten.sub.Tensor(reshape_default_1, amax_default);  reshape_default_1 = amax_default = None
        exp_default: "f32[8192, 29056]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[8192, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[8192, 29056]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        ne_scalar: "b8[8192]" = torch.ops.aten.ne.Scalar(reshape_default_2, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[8192]" = torch.ops.aten.where.self(ne_scalar, reshape_default_2, full_default);  reshape_default_2 = full_default = None
        unsqueeze_default: "i64[8192, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "f32[8192, 1]" = torch.ops.aten.gather.default(sub_tensor_1, 1, unsqueeze_default);  sub_tensor_1 = unsqueeze_default = None
        squeeze_dim: "f32[8192]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "f32[8192]" = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[8192]" = torch.ops.aten.where.self(ne_scalar, neg_default, full_default_1);  neg_default = full_default_1 = None
        sum_default: "i64[]" = torch.ops.aten.sum.default(ne_scalar);  ne_scalar = None
        convert_element_type_default: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default, torch.float32);  sum_default = None
        sum_default_1: "f32[]" = torch.ops.aten.sum.default(where_self_1);  where_self_1 = None
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(sum_default_1, convert_element_type_default);  sum_default_1 = convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:392 in forward, code: hidden_states = self.ln(hidden_states)
        div_tensor_1: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_48, 1024);  rsqrt_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_2: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 1024);  rsqrt_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_3: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 1024);  rsqrt_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_4: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 1024);  rsqrt_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_5: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 1024);  rsqrt_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_6: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 1024);  rsqrt_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_7: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 1024);  rsqrt_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_8: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 1024);  rsqrt_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_9: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 1024);  rsqrt_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_10: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 1024);  rsqrt_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_11: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 1024);  rsqrt_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_12: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 1024);  rsqrt_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_13: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 1024);  rsqrt_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_14: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 1024);  rsqrt_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_15: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 1024);  rsqrt_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_16: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 1024);  rsqrt_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_17: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 1024);  rsqrt_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_18: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 1024);  rsqrt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_19: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 1024);  rsqrt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_20: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 1024);  rsqrt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_21: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 1024);  rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_22: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 1024);  rsqrt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_23: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 1024);  rsqrt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_24: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 1024);  rsqrt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_25: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 1024);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_26: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 1024);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_27: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 1024);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_28: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 1024);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_29: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 1024);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_30: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 1024);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_31: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 1024);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_32: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 1024);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_33: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 1024);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_34: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 1024);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_35: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 1024);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_36: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 1024);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_37: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 1024);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_38: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 1024);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_39: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 1024);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_40: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 1024);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_41: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 1024);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_42: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 1024);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_43: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 1024);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_44: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 1024);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_45: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 1024);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_46: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 1024);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_47: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 1024);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_tensor_48: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 1024);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_tensor_49: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None
        return (div_tensor, div_tensor_1, div_tensor_2, div_tensor_3, div_tensor_4, div_tensor_5, div_tensor_6, div_tensor_7, div_tensor_8, div_tensor_9, div_tensor_10, div_tensor_11, div_tensor_12, div_tensor_13, div_tensor_14, div_tensor_15, div_tensor_16, div_tensor_17, div_tensor_18, div_tensor_19, div_tensor_20, div_tensor_21, div_tensor_22, div_tensor_23, div_tensor_24, div_tensor_25, div_tensor_26, div_tensor_27, div_tensor_28, div_tensor_29, div_tensor_30, div_tensor_31, div_tensor_32, div_tensor_33, div_tensor_34, div_tensor_35, div_tensor_36, div_tensor_37, div_tensor_38, div_tensor_39, div_tensor_40, div_tensor_41, div_tensor_42, div_tensor_43, div_tensor_44, div_tensor_45, div_tensor_46, div_tensor_47, div_tensor_48, div_tensor_49)


def _default_make_inputs():
    return [
    torch.randn([8192, 29056], dtype=torch.float32, device='cuda'),
    [16, 512, 29056],  # _shape_param_0
    torch.randint(0, 2, [16, 512], dtype=torch.int64, device='cuda'),
    [-1, 29056],  # _shape_param_1
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
