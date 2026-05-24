import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[4, 512]", primals_3: "f32[64]", primals_4: "bf16[4096]", primals_5: "bf16[4096, 4096]", primals_6: "bf16[1024, 4096]", primals_7: "bf16[1024, 4096]", primals_8: "bf16[4096, 4096]", primals_9: "bf16[4096]", primals_10: "bf16[14336, 4096]", primals_11: "bf16[14336, 4096]", primals_12: "bf16[4096, 14336]", primals_13: "bf16[4096]", primals_14: "bf16[4096, 4096]", primals_15: "bf16[1024, 4096]", primals_16: "bf16[1024, 4096]", primals_17: "bf16[4096, 4096]", primals_18: "bf16[4096]", primals_19: "bf16[14336, 4096]", primals_20: "bf16[14336, 4096]", primals_21: "bf16[4096, 14336]", primals_22: "bf16[4096]", primals_23: "bf16[4096, 4096]", primals_24: "bf16[1024, 4096]", primals_25: "bf16[1024, 4096]", primals_26: "bf16[4096, 4096]", primals_27: "bf16[4096]", primals_28: "bf16[14336, 4096]", primals_29: "bf16[14336, 4096]", primals_30: "bf16[4096, 14336]", primals_31: "bf16[4096]", primals_32: "bf16[4096, 4096]", primals_33: "bf16[1024, 4096]", primals_34: "bf16[1024, 4096]", primals_35: "bf16[4096, 4096]", primals_36: "bf16[4096]", primals_37: "bf16[14336, 4096]", primals_38: "bf16[14336, 4096]", primals_39: "bf16[4096, 14336]", primals_40: "bf16[4096]", primals_41: "bf16[4096, 4096]", primals_42: "bf16[1024, 4096]", primals_43: "bf16[1024, 4096]", primals_44: "bf16[4096, 4096]", primals_45: "bf16[4096]", primals_46: "bf16[14336, 4096]", primals_47: "bf16[14336, 4096]", primals_48: "bf16[4096, 14336]", primals_49: "bf16[4096]", primals_50: "bf16[4096, 4096]", primals_51: "bf16[1024, 4096]", primals_52: "bf16[1024, 4096]", primals_53: "bf16[4096, 4096]", primals_54: "bf16[4096]", primals_55: "bf16[14336, 4096]", primals_56: "bf16[14336, 4096]", primals_57: "bf16[4096, 14336]", primals_58: "bf16[4096]", primals_59: "bf16[4096, 4096]", primals_60: "bf16[1024, 4096]", primals_61: "bf16[1024, 4096]", primals_62: "bf16[4096, 4096]", primals_63: "bf16[4096]", primals_64: "bf16[14336, 4096]", primals_65: "bf16[14336, 4096]", primals_66: "bf16[4096, 14336]", primals_67: "bf16[4096]", primals_68: "bf16[4096, 4096]", primals_69: "bf16[1024, 4096]", primals_70: "bf16[1024, 4096]", primals_71: "bf16[4096, 4096]", primals_72: "bf16[4096]", primals_73: "bf16[14336, 4096]", primals_74: "bf16[14336, 4096]", primals_75: "bf16[4096, 14336]", primals_76: "bf16[4096]", primals_77: "bf16[32768, 4096]", embedding: "bf16[4, 512, 4096]", rsqrt: "f32[4, 512, 1]", view_4: "bf16[2048, 4096]", add_4: "bf16[4, 32, 512, 128]", view_13: "bf16[4, 32, 512, 128]", view_14: "bf16[4, 32, 512, 128]", where: "bf16[4, 1, 512, 512]", getitem: "bf16[4, 32, 512, 128]", getitem_1: "f32[4, 32, 512, 1]", getitem_6: "i64[]", getitem_7: "i64[]", mm_3: "bf16[2048, 4096]", rsqrt_1: "f32[4, 512, 1]", view_18: "bf16[2048, 4096]", mm_4: "bf16[2048, 14336]", mm_5: "bf16[2048, 14336]", view_22: "bf16[2048, 14336]", add_9: "bf16[4, 512, 4096]", rsqrt_2: "f32[4, 512, 1]", view_24: "bf16[2048, 4096]", add_11: "bf16[4, 32, 512, 128]", view_33: "bf16[4, 32, 512, 128]", view_34: "bf16[4, 32, 512, 128]", getitem_9: "bf16[4, 32, 512, 128]", getitem_10: "f32[4, 32, 512, 1]", getitem_15: "i64[]", getitem_16: "i64[]", add_13: "bf16[4, 512, 4096]", rsqrt_3: "f32[4, 512, 1]", view_38: "bf16[2048, 4096]", mm_11: "bf16[2048, 14336]", mm_12: "bf16[2048, 14336]", view_42: "bf16[2048, 14336]", add_16: "bf16[4, 512, 4096]", rsqrt_4: "f32[4, 512, 1]", view_44: "bf16[2048, 4096]", add_18: "bf16[4, 32, 512, 128]", view_53: "bf16[4, 32, 512, 128]", view_54: "bf16[4, 32, 512, 128]", getitem_18: "bf16[4, 32, 512, 128]", getitem_19: "f32[4, 32, 512, 1]", getitem_24: "i64[]", getitem_25: "i64[]", add_20: "bf16[4, 512, 4096]", rsqrt_5: "f32[4, 512, 1]", view_58: "bf16[2048, 4096]", mm_18: "bf16[2048, 14336]", mm_19: "bf16[2048, 14336]", view_62: "bf16[2048, 14336]", add_23: "bf16[4, 512, 4096]", rsqrt_6: "f32[4, 512, 1]", view_64: "bf16[2048, 4096]", add_25: "bf16[4, 32, 512, 128]", view_73: "bf16[4, 32, 512, 128]", view_74: "bf16[4, 32, 512, 128]", getitem_27: "bf16[4, 32, 512, 128]", getitem_28: "f32[4, 32, 512, 1]", getitem_33: "i64[]", getitem_34: "i64[]", add_27: "bf16[4, 512, 4096]", rsqrt_7: "f32[4, 512, 1]", view_78: "bf16[2048, 4096]", mm_25: "bf16[2048, 14336]", mm_26: "bf16[2048, 14336]", view_82: "bf16[2048, 14336]", add_30: "bf16[4, 512, 4096]", rsqrt_8: "f32[4, 512, 1]", view_84: "bf16[2048, 4096]", add_32: "bf16[4, 32, 512, 128]", view_93: "bf16[4, 32, 512, 128]", view_94: "bf16[4, 32, 512, 128]", getitem_36: "bf16[4, 32, 512, 128]", getitem_37: "f32[4, 32, 512, 1]", getitem_42: "i64[]", getitem_43: "i64[]", add_34: "bf16[4, 512, 4096]", rsqrt_9: "f32[4, 512, 1]", view_98: "bf16[2048, 4096]", mm_32: "bf16[2048, 14336]", mm_33: "bf16[2048, 14336]", view_102: "bf16[2048, 14336]", add_37: "bf16[4, 512, 4096]", rsqrt_10: "f32[4, 512, 1]", view_104: "bf16[2048, 4096]", add_39: "bf16[4, 32, 512, 128]", view_113: "bf16[4, 32, 512, 128]", view_114: "bf16[4, 32, 512, 128]", getitem_45: "bf16[4, 32, 512, 128]", getitem_46: "f32[4, 32, 512, 1]", getitem_51: "i64[]", getitem_52: "i64[]", add_41: "bf16[4, 512, 4096]", rsqrt_11: "f32[4, 512, 1]", view_118: "bf16[2048, 4096]", mm_39: "bf16[2048, 14336]", mm_40: "bf16[2048, 14336]", view_122: "bf16[2048, 14336]", add_44: "bf16[4, 512, 4096]", rsqrt_12: "f32[4, 512, 1]", view_124: "bf16[2048, 4096]", add_46: "bf16[4, 32, 512, 128]", view_133: "bf16[4, 32, 512, 128]", view_134: "bf16[4, 32, 512, 128]", getitem_54: "bf16[4, 32, 512, 128]", getitem_55: "f32[4, 32, 512, 1]", getitem_60: "i64[]", getitem_61: "i64[]", add_48: "bf16[4, 512, 4096]", rsqrt_13: "f32[4, 512, 1]", view_138: "bf16[2048, 4096]", mm_46: "bf16[2048, 14336]", mm_47: "bf16[2048, 14336]", view_142: "bf16[2048, 14336]", add_51: "bf16[4, 512, 4096]", rsqrt_14: "f32[4, 512, 1]", view_144: "bf16[2048, 4096]", add_53: "bf16[4, 32, 512, 128]", view_153: "bf16[4, 32, 512, 128]", view_154: "bf16[4, 32, 512, 128]", getitem_63: "bf16[4, 32, 512, 128]", getitem_64: "f32[4, 32, 512, 1]", getitem_69: "i64[]", getitem_70: "i64[]", add_55: "bf16[4, 512, 4096]", rsqrt_15: "f32[4, 512, 1]", view_158: "bf16[2048, 4096]", mm_53: "bf16[2048, 14336]", mm_54: "bf16[2048, 14336]", view_162: "bf16[2048, 14336]", add_58: "bf16[4, 512, 4096]", rsqrt_16: "f32[4, 512, 1]", view_164: "bf16[2048, 4096]", view_165: "bf16[4, 512, 32768]", constant_pad_nd: "i64[4, 513]", amax: "f32[2048, 1]", log: "f32[2048, 1]", convert_element_type_168: "f32[]", tangents_1: "f32[]", tangents_2: "bf16[4, 512, 32768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        div_9: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_168);  tangents_1 = convert_element_type_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:61 in ForCausalLMLoss, code: shift_labels = labels[..., 1:].contiguous()
        slice_36: "i64[4, 512]" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807);  constant_pad_nd = None
        clone_18: "i64[4, 512]" = torch.ops.aten.clone.default(slice_36, memory_format = torch.contiguous_format);  slice_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:65 in ForCausalLMLoss, code: shift_labels = shift_labels.view(-1)
        view_167: "i64[2048]" = torch.ops.aten.reshape.default(clone_18, [-1]);  clone_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        unsqueeze_47: "i64[2048, 1]" = torch.ops.aten.unsqueeze.default(view_167, 1);  view_167 = None
        ne_4: "b8[2048, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_47, -100)
        full_default_17: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_10: "i64[2048, 1]" = torch.ops.aten.where.self(ne_4, unsqueeze_47, full_default_17);  unsqueeze_47 = full_default_17 = None

        # No stacktrace found for following nodes
        iota_default: "i64[32768]" = torch.ops.prims.iota.default(32768, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 32768]" = torch.ops.aten.reshape.default(iota_default, [1, 32768]);  iota_default = None
        expand_default: "i64[2048, 32768]" = torch.ops.aten.expand.default(where_10, [2048, 32768]);  where_10 = None
        eq_tensor: "b8[2048, 32768]" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        where_self: "f32[2048, 32768]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        full_default_18: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_11: "f32[2048, 1]" = torch.ops.aten.where.self(ne_4, div_9, full_default_18);  ne_4 = div_9 = None
        mul_77: "f32[2048, 32768]" = torch.ops.aten.mul.Tensor(where_self, where_11);  where_self = where_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:56 in ForCausalLMLoss, code: logits = logits.float()
        convert_element_type_167: "f32[4, 512, 32768]" = torch.ops.prims.convert_element_type.default(view_165, torch.float32);  view_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_166: "f32[2048, 32768]" = torch.ops.aten.reshape.default(convert_element_type_167, [-1, 32768]);  convert_element_type_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        sub_2: "f32[2048, 32768]" = torch.ops.aten.sub.Tensor(view_166, amax);  view_166 = amax = None
        sub_3: "f32[2048, 32768]" = torch.ops.aten.sub.Tensor(sub_2, log);  sub_2 = log = None
        exp_9: "f32[2048, 32768]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        sum_4: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_77, [1], True)
        mul_78: "f32[2048, 32768]" = torch.ops.aten.mul.Tensor(exp_9, sum_4);  exp_9 = sum_4 = None
        sub_4: "f32[2048, 32768]" = torch.ops.aten.sub.Tensor(mul_77, mul_78);  mul_77 = mul_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_168: "f32[4, 512, 32768]" = torch.ops.aten.reshape.default(sub_4, [4, 512, 32768]);  sub_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:56 in ForCausalLMLoss, code: logits = logits.float()
        convert_element_type_169: "bf16[4, 512, 32768]" = torch.ops.prims.convert_element_type.default(view_168, torch.bfloat16);  view_168 = None
        add_60: "bf16[4, 512, 32768]" = torch.ops.aten.add.Tensor(tangents_2, convert_element_type_169);  tangents_2 = convert_element_type_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:460 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        view_169: "bf16[2048, 32768]" = torch.ops.aten.reshape.default(add_60, [2048, 32768]);  add_60 = None
        permute_90: "bf16[32768, 2048]" = torch.ops.aten.permute.default(view_169, [1, 0])
        mm_57: "bf16[32768, 4096]" = torch.ops.aten.mm.default(permute_90, view_164);  permute_90 = view_164 = None
        permute_89: "bf16[4096, 32768]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
        permute_92: "bf16[32768, 4096]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
        mm_58: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_169, permute_92);  view_169 = permute_92 = None
        view_170: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_58, [4, 512, 4096]);  mm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_79: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_170, primals_76);  primals_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_163: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_58, torch.float32);  add_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_75: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_163, rsqrt_16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_164: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_75, torch.bfloat16);  mul_75 = None
        mul_80: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(view_170, convert_element_type_164);  view_170 = convert_element_type_164 = None
        sum_5: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_80, [0, 1], True);  mul_80 = None
        view_171: "bf16[4096]" = torch.ops.aten.reshape.default(sum_5, [4096]);  sum_5 = None
        convert_element_type_174: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_79, torch.float32);  mul_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_81: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_174, convert_element_type_163)
        mul_82: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_174, rsqrt_16);  convert_element_type_174 = None
        sum_6: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_81, [2], True);  mul_81 = None
        pow_18: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_16, 3);  rsqrt_16 = None
        mul_83: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_6, -0.5);  sum_6 = None
        mul_84: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_83, pow_18);  mul_83 = pow_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_22: "f32[4, 512, 4096]" = torch.ops.aten.expand.default(mul_84, [4, 512, 4096]);  mul_84 = None
        div_10: "f32[4, 512, 4096]" = torch.ops.aten.div.Scalar(expand_22, 4096);  expand_22 = None
        pow_19: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_163, 1.0);  convert_element_type_163 = None
        mul_85: "f32[4, 512, 4096]" = torch.ops.aten.mul.Scalar(pow_19, 2.0);  pow_19 = None
        mul_86: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(div_10, mul_85);  div_10 = mul_85 = None
        add_61: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_82, mul_86);  mul_82 = mul_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_175: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_61, torch.bfloat16);  add_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_172: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(convert_element_type_175, [2048, 4096])
        permute_94: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_172, [1, 0])
        mm_59: "bf16[4096, 14336]" = torch.ops.aten.mm.default(permute_94, view_162);  permute_94 = view_162 = None
        permute_88: "bf16[14336, 4096]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
        permute_96: "bf16[4096, 14336]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
        mm_60: "bf16[2048, 14336]" = torch.ops.aten.mm.default(view_172, permute_96);  view_172 = permute_96 = None
        view_173: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_60, [4, 512, 14336]);  mm_60 = None
        view_159: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_53, [4, 512, 14336]);  mm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_157: "f32[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(view_159, torch.float32);  view_159 = None
        neg_23: "f32[4, 512, 14336]" = torch.ops.aten.neg.default(convert_element_type_157)
        exp_7: "f32[4, 512, 14336]" = torch.ops.aten.exp.default(neg_23);  neg_23 = None
        add_57: "f32[4, 512, 14336]" = torch.ops.aten.add.Tensor(exp_7, 1);  exp_7 = None
        div_7: "f32[4, 512, 14336]" = torch.ops.aten.div.Tensor(convert_element_type_157, add_57)
        convert_element_type_158: "bf16[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_87: "bf16[4, 512, 14336]" = torch.ops.aten.mul.Tensor(view_173, convert_element_type_158);  convert_element_type_158 = None
        view_161: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_54, [4, 512, 14336]);  mm_54 = None
        mul_88: "bf16[4, 512, 14336]" = torch.ops.aten.mul.Tensor(view_173, view_161);  view_173 = view_161 = None
        view_174: "bf16[2048, 14336]" = torch.ops.aten.reshape.default(mul_87, [2048, 14336]);  mul_87 = None
        permute_98: "bf16[14336, 2048]" = torch.ops.aten.permute.default(view_174, [1, 0])
        mm_61: "bf16[14336, 4096]" = torch.ops.aten.mm.default(permute_98, view_158);  permute_98 = None
        permute_87: "bf16[4096, 14336]" = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
        permute_100: "bf16[14336, 4096]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        mm_62: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_174, permute_100);  view_174 = permute_100 = None
        view_175: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_62, [4, 512, 4096]);  mm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_184: "f32[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(mul_88, torch.float32);  mul_88 = None
        reciprocal: "f32[4, 512, 14336]" = torch.ops.aten.reciprocal.default(add_57);  add_57 = None
        mul_89: "f32[4, 512, 14336]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        mul_90: "f32[4, 512, 14336]" = torch.ops.aten.mul.Tensor(convert_element_type_184, mul_89);  convert_element_type_184 = None
        sub_5: "f32[4, 512, 14336]" = torch.ops.aten.sub.Tensor(1, mul_89);  mul_89 = None
        mul_91: "f32[4, 512, 14336]" = torch.ops.aten.mul.Tensor(convert_element_type_157, sub_5);  convert_element_type_157 = sub_5 = None
        add_63: "f32[4, 512, 14336]" = torch.ops.aten.add.Tensor(mul_91, 1);  mul_91 = None
        mul_92: "f32[4, 512, 14336]" = torch.ops.aten.mul.Tensor(mul_90, add_63);  mul_90 = add_63 = None
        convert_element_type_186: "bf16[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(mul_92, torch.bfloat16);  mul_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_176: "bf16[2048, 14336]" = torch.ops.aten.reshape.default(convert_element_type_186, [2048, 14336]);  convert_element_type_186 = None
        permute_102: "bf16[14336, 2048]" = torch.ops.aten.permute.default(view_176, [1, 0])
        mm_63: "bf16[14336, 4096]" = torch.ops.aten.mm.default(permute_102, view_158);  permute_102 = view_158 = None
        permute_86: "bf16[4096, 14336]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
        permute_104: "bf16[14336, 4096]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
        mm_64: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_176, permute_104);  view_176 = permute_104 = None
        view_177: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_64, [4, 512, 4096]);  mm_64 = None
        add_64: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(view_175, view_177);  view_175 = view_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_93: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_64, primals_72);  primals_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_153: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_55, torch.float32);  add_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_72: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_153, rsqrt_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_154: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_72, torch.bfloat16);  mul_72 = None
        mul_94: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_64, convert_element_type_154);  add_64 = convert_element_type_154 = None
        sum_7: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_94, [0, 1], True);  mul_94 = None
        view_178: "bf16[4096]" = torch.ops.aten.reshape.default(sum_7, [4096]);  sum_7 = None
        convert_element_type_191: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_93, torch.float32);  mul_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_95: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_191, convert_element_type_153)
        mul_96: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_191, rsqrt_15);  convert_element_type_191 = None
        sum_8: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_95, [2], True);  mul_95 = None
        pow_20: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_15, 3);  rsqrt_15 = None
        mul_97: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_8, -0.5);  sum_8 = None
        mul_98: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_97, pow_20);  mul_97 = pow_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_23: "f32[4, 512, 4096]" = torch.ops.aten.expand.default(mul_98, [4, 512, 4096]);  mul_98 = None
        div_11: "f32[4, 512, 4096]" = torch.ops.aten.div.Scalar(expand_23, 4096);  expand_23 = None
        pow_21: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_153, 1.0);  convert_element_type_153 = None
        mul_99: "f32[4, 512, 4096]" = torch.ops.aten.mul.Scalar(pow_21, 2.0);  pow_21 = None
        mul_100: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(div_11, mul_99);  div_11 = mul_99 = None
        add_65: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_96, mul_100);  mul_96 = mul_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_192: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_65, torch.bfloat16);  add_65 = None
        add_66: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_175, convert_element_type_192);  convert_element_type_175 = convert_element_type_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_179: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(add_66, [2048, 4096])
        permute_106: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_179, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_84: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_63, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_155: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_84, [4, 512, -1]);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_156: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_155, [2048, 4096]);  view_155 = None
        mm_65: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_106, view_156);  permute_106 = view_156 = None
        permute_85: "bf16[4096, 4096]" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
        permute_108: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        mm_66: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_179, permute_108);  view_179 = permute_108 = None
        view_180: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_66, [4, 512, 4096]);  mm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_181: "bf16[4, 512, 32, 128]" = torch.ops.aten.reshape.default(view_180, [4, 512, 32, 128]);  view_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_110: "bf16[4, 32, 512, 128]" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_110, add_53, view_153, view_154, getitem_63, getitem_64, getitem_69, getitem_70, where, None, None, 512, 512, 0.0, False, scale = 0.08838834764831845);  permute_110 = add_53 = view_153 = view_154 = getitem_63 = getitem_64 = getitem_69 = getitem_70 = None
        getitem_72: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward[0]
        getitem_73: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward[1]
        getitem_74: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward[2];  _scaled_dot_product_cudnn_attention_backward = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_182: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.reshape.default(getitem_74, [4, 8, 4, 512, 128]);  getitem_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_9: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.sum.dim_IntList(view_182, [2], True);  view_182 = None
        squeeze_1: "bf16[4, 8, 512, 128]" = torch.ops.aten.squeeze.dim(sum_9, 2);  sum_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_183: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.reshape.default(getitem_73, [4, 8, 4, 512, 128]);  getitem_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_10: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.sum.dim_IntList(view_183, [2], True);  view_183 = None
        squeeze_2: "bf16[4, 8, 512, 128]" = torch.ops.aten.squeeze.dim(sum_10, 2);  sum_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:369 in forward, code: position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        iota: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[512]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:370 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_5: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:314 in forward, code: inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        unsqueeze_10: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_3, 0);  primals_3 = None
        unsqueeze_11: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 2);  unsqueeze_10 = None
        expand_2: "f32[1, 64, 1]" = torch.ops.aten.expand.default(unsqueeze_11, [1, -1, 1]);  unsqueeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:315 in forward, code: position_ids_expanded = position_ids[:, None, :].float()
        convert_element_type: "f32[1, 1, 512]" = torch.ops.prims.convert_element_type.default(unsqueeze_5, torch.float32);  unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:319 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        expand_3: "f32[1, 64, 1]" = torch.ops.aten.expand.default(expand_2, [1, 64, 1]);  expand_2 = None
        expand_4: "f32[1, 1, 512]" = torch.ops.aten.expand.default(convert_element_type, [1, 1, 512]);  convert_element_type = None
        mul: "f32[1, 64, 512]" = torch.ops.aten.mul.Tensor(expand_3, expand_4);  expand_3 = expand_4 = None
        permute: "f32[1, 512, 64]" = torch.ops.aten.permute.default(mul, [0, 2, 1]);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:320 in forward, code: emb = torch.cat((freqs, freqs), dim=-1)
        unsqueeze_13: "f32[1, 512, 1, 64]" = torch.ops.aten.unsqueeze.default(permute, 2);  permute = None
        expand_5: "f32[1, 512, 2, 64]" = torch.ops.aten.expand.default(unsqueeze_13, [1, 512, 2, 64]);  unsqueeze_13 = None
        clone: "f32[1, 512, 2, 64]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_3: "f32[1, 512, 128]" = torch.ops.aten.reshape.default(clone, [1, 512, 128]);  clone = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:322 in forward, code: sin = emb.sin() * self.attention_scaling
        sin: "f32[1, 512, 128]" = torch.ops.aten.sin.default(view_3)
        mul_2: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sin, 1.0);  sin = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:324 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_2: "bf16[1, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:78 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_15: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1);  convert_element_type_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_101: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(squeeze_2, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_37: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(mul_101, 3, 0, 64)
        slice_38: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(mul_101, 3, 64, 128);  mul_101 = None
        neg_26: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_37);  slice_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        full_default_22: "bf16[4, 8, 512, 128]" = torch.ops.aten.full.default([4, 8, 512, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter: "bf16[4, 8, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_22, neg_26, 3, 64, 9223372036854775807);  neg_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_1: "bf16[4, 8, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_22, slice_38, 3, 0, 64);  slice_38 = None
        add_67: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(slice_scatter, slice_scatter_1);  slice_scatter = slice_scatter_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:321 in forward, code: cos = emb.cos() * self.attention_scaling
        cos: "f32[1, 512, 128]" = torch.ops.aten.cos.default(view_3);  view_3 = None
        mul_1: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(cos, 1.0);  cos = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:324 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_1: "bf16[1, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:77 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_14: "bf16[1, 1, 512, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1);  convert_element_type_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_102: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(squeeze_2, unsqueeze_14);  squeeze_2 = None
        add_68: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(add_67, mul_102);  add_67 = mul_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_103: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(getitem_72, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_39: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(mul_103, 3, 0, 64)
        slice_40: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(mul_103, 3, 64, 128);  mul_103 = None
        neg_27: "bf16[4, 32, 512, 64]" = torch.ops.aten.neg.default(slice_39);  slice_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        full_default_24: "bf16[4, 32, 512, 128]" = torch.ops.aten.full.default([4, 32, 512, 128], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_2: "bf16[4, 32, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_24, neg_27, 3, 64, 9223372036854775807);  neg_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_3: "bf16[4, 32, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_24, slice_40, 3, 0, 64);  slice_40 = None
        add_69: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(slice_scatter_2, slice_scatter_3);  slice_scatter_2 = slice_scatter_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_104: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(getitem_72, unsqueeze_14);  getitem_72 = None
        add_70: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(add_69, mul_104);  add_69 = mul_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_111: "bf16[4, 512, 8, 128]" = torch.ops.aten.permute.default(squeeze_1, [0, 2, 1, 3]);  squeeze_1 = None
        clone_19: "bf16[4, 512, 8, 128]" = torch.ops.aten.clone.default(permute_111, memory_format = torch.contiguous_format);  permute_111 = None
        view_184: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_19, [4, 512, 1024]);  clone_19 = None
        view_185: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(view_184, [2048, 1024]);  view_184 = None
        permute_112: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_185, [1, 0])
        mm_67: "bf16[1024, 4096]" = torch.ops.aten.mm.default(permute_112, view_144);  permute_112 = None
        permute_82: "bf16[4096, 1024]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
        permute_114: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_82, [1, 0]);  permute_82 = None
        mm_68: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_185, permute_114);  view_185 = permute_114 = None
        view_186: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_68, [4, 512, 4096]);  mm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_116: "bf16[4, 512, 8, 128]" = torch.ops.aten.permute.default(add_68, [0, 2, 1, 3]);  add_68 = None
        clone_20: "bf16[4, 512, 8, 128]" = torch.ops.aten.clone.default(permute_116, memory_format = torch.contiguous_format);  permute_116 = None
        view_187: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_20, [4, 512, 1024]);  clone_20 = None
        view_188: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(view_187, [2048, 1024]);  view_187 = None
        permute_117: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_188, [1, 0])
        mm_69: "bf16[1024, 4096]" = torch.ops.aten.mm.default(permute_117, view_144);  permute_117 = None
        permute_80: "bf16[4096, 1024]" = torch.ops.aten.permute.default(primals_69, [1, 0]);  primals_69 = None
        permute_119: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
        mm_70: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_188, permute_119);  view_188 = permute_119 = None
        view_189: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_70, [4, 512, 4096]);  mm_70 = None
        add_71: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(view_186, view_189);  view_186 = view_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_121: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(add_70, [0, 2, 1, 3]);  add_70 = None
        clone_21: "bf16[4, 512, 32, 128]" = torch.ops.aten.clone.default(permute_121, memory_format = torch.contiguous_format);  permute_121 = None
        view_190: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(clone_21, [4, 512, 4096]);  clone_21 = None
        view_191: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_190, [2048, 4096]);  view_190 = None
        permute_122: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_191, [1, 0])
        mm_71: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_122, view_144);  permute_122 = view_144 = None
        permute_78: "bf16[4096, 4096]" = torch.ops.aten.permute.default(primals_68, [1, 0]);  primals_68 = None
        permute_124: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
        mm_72: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_191, permute_124);  view_191 = permute_124 = None
        view_192: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_72, [4, 512, 4096]);  mm_72 = None
        add_72: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_71, view_192);  add_71 = view_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_105: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_72, primals_67);  primals_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_143: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_51, torch.float32);  add_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_66: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_143, rsqrt_14)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_144: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_66, torch.bfloat16);  mul_66 = None
        mul_106: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_72, convert_element_type_144);  add_72 = convert_element_type_144 = None
        sum_11: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_106, [0, 1], True);  mul_106 = None
        view_193: "bf16[4096]" = torch.ops.aten.reshape.default(sum_11, [4096]);  sum_11 = None
        convert_element_type_209: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_105, torch.float32);  mul_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_107: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_209, convert_element_type_143)
        mul_108: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_209, rsqrt_14);  convert_element_type_209 = None
        sum_12: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_107, [2], True);  mul_107 = None
        pow_22: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_14, 3);  rsqrt_14 = None
        mul_109: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_12, -0.5);  sum_12 = None
        mul_110: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_109, pow_22);  mul_109 = pow_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_24: "f32[4, 512, 4096]" = torch.ops.aten.expand.default(mul_110, [4, 512, 4096]);  mul_110 = None
        div_12: "f32[4, 512, 4096]" = torch.ops.aten.div.Scalar(expand_24, 4096);  expand_24 = None
        pow_23: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_143, 1.0);  convert_element_type_143 = None
        mul_111: "f32[4, 512, 4096]" = torch.ops.aten.mul.Scalar(pow_23, 2.0);  pow_23 = None
        mul_112: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(div_12, mul_111);  div_12 = mul_111 = None
        add_73: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_108, mul_112);  mul_108 = mul_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_210: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_73, torch.bfloat16);  add_73 = None
        add_74: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_66, convert_element_type_210);  add_66 = convert_element_type_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_194: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(add_74, [2048, 4096])
        permute_126: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_194, [1, 0])
        mm_73: "bf16[4096, 14336]" = torch.ops.aten.mm.default(permute_126, view_142);  permute_126 = view_142 = None
        permute_77: "bf16[14336, 4096]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
        permute_128: "bf16[4096, 14336]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        mm_74: "bf16[2048, 14336]" = torch.ops.aten.mm.default(view_194, permute_128);  view_194 = permute_128 = None
        view_195: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_74, [4, 512, 14336]);  mm_74 = None
        view_139: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_46, [4, 512, 14336]);  mm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_137: "f32[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(view_139, torch.float32);  view_139 = None
        neg_20: "f32[4, 512, 14336]" = torch.ops.aten.neg.default(convert_element_type_137)
        exp_6: "f32[4, 512, 14336]" = torch.ops.aten.exp.default(neg_20);  neg_20 = None
        add_50: "f32[4, 512, 14336]" = torch.ops.aten.add.Tensor(exp_6, 1);  exp_6 = None
        div_6: "f32[4, 512, 14336]" = torch.ops.aten.div.Tensor(convert_element_type_137, add_50)
        convert_element_type_138: "bf16[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_113: "bf16[4, 512, 14336]" = torch.ops.aten.mul.Tensor(view_195, convert_element_type_138);  convert_element_type_138 = None
        view_141: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_47, [4, 512, 14336]);  mm_47 = None
        mul_114: "bf16[4, 512, 14336]" = torch.ops.aten.mul.Tensor(view_195, view_141);  view_195 = view_141 = None
        view_196: "bf16[2048, 14336]" = torch.ops.aten.reshape.default(mul_113, [2048, 14336]);  mul_113 = None
        permute_130: "bf16[14336, 2048]" = torch.ops.aten.permute.default(view_196, [1, 0])
        mm_75: "bf16[14336, 4096]" = torch.ops.aten.mm.default(permute_130, view_138);  permute_130 = None
        permute_76: "bf16[4096, 14336]" = torch.ops.aten.permute.default(primals_65, [1, 0]);  primals_65 = None
        permute_132: "bf16[14336, 4096]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        mm_76: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_196, permute_132);  view_196 = permute_132 = None
        view_197: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_76, [4, 512, 4096]);  mm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_219: "f32[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(mul_114, torch.float32);  mul_114 = None
        reciprocal_1: "f32[4, 512, 14336]" = torch.ops.aten.reciprocal.default(add_50);  add_50 = None
        mul_115: "f32[4, 512, 14336]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
        mul_116: "f32[4, 512, 14336]" = torch.ops.aten.mul.Tensor(convert_element_type_219, mul_115);  convert_element_type_219 = None
        sub_6: "f32[4, 512, 14336]" = torch.ops.aten.sub.Tensor(1, mul_115);  mul_115 = None
        mul_117: "f32[4, 512, 14336]" = torch.ops.aten.mul.Tensor(convert_element_type_137, sub_6);  convert_element_type_137 = sub_6 = None
        add_76: "f32[4, 512, 14336]" = torch.ops.aten.add.Tensor(mul_117, 1);  mul_117 = None
        mul_118: "f32[4, 512, 14336]" = torch.ops.aten.mul.Tensor(mul_116, add_76);  mul_116 = add_76 = None
        convert_element_type_221: "bf16[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(mul_118, torch.bfloat16);  mul_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_198: "bf16[2048, 14336]" = torch.ops.aten.reshape.default(convert_element_type_221, [2048, 14336]);  convert_element_type_221 = None
        permute_134: "bf16[14336, 2048]" = torch.ops.aten.permute.default(view_198, [1, 0])
        mm_77: "bf16[14336, 4096]" = torch.ops.aten.mm.default(permute_134, view_138);  permute_134 = view_138 = None
        permute_75: "bf16[4096, 14336]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
        permute_136: "bf16[14336, 4096]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
        mm_78: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_198, permute_136);  view_198 = permute_136 = None
        view_199: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_78, [4, 512, 4096]);  mm_78 = None
        add_77: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(view_197, view_199);  view_197 = view_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_119: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_77, primals_63);  primals_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_133: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_48, torch.float32);  add_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_63: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_133, rsqrt_13)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_134: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_63, torch.bfloat16);  mul_63 = None
        mul_120: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_77, convert_element_type_134);  add_77 = convert_element_type_134 = None
        sum_13: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_120, [0, 1], True);  mul_120 = None
        view_200: "bf16[4096]" = torch.ops.aten.reshape.default(sum_13, [4096]);  sum_13 = None
        convert_element_type_226: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_119, torch.float32);  mul_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_121: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_226, convert_element_type_133)
        mul_122: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_226, rsqrt_13);  convert_element_type_226 = None
        sum_14: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_121, [2], True);  mul_121 = None
        pow_24: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_13, 3);  rsqrt_13 = None
        mul_123: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_14, -0.5);  sum_14 = None
        mul_124: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_123, pow_24);  mul_123 = pow_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_25: "f32[4, 512, 4096]" = torch.ops.aten.expand.default(mul_124, [4, 512, 4096]);  mul_124 = None
        div_13: "f32[4, 512, 4096]" = torch.ops.aten.div.Scalar(expand_25, 4096);  expand_25 = None
        pow_25: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_133, 1.0);  convert_element_type_133 = None
        mul_125: "f32[4, 512, 4096]" = torch.ops.aten.mul.Scalar(pow_25, 2.0);  pow_25 = None
        mul_126: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(div_13, mul_125);  div_13 = mul_125 = None
        add_78: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_122, mul_126);  mul_122 = mul_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_227: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_78, torch.bfloat16);  add_78 = None
        add_79: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_74, convert_element_type_227);  add_74 = convert_element_type_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_201: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(add_79, [2048, 4096])
        permute_138: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_201, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_73: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_54, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_135: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_73, [4, 512, -1]);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_136: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_135, [2048, 4096]);  view_135 = None
        mm_79: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_138, view_136);  permute_138 = view_136 = None
        permute_74: "bf16[4096, 4096]" = torch.ops.aten.permute.default(primals_62, [1, 0]);  primals_62 = None
        permute_140: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
        mm_80: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_201, permute_140);  view_201 = permute_140 = None
        view_202: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_80, [4, 512, 4096]);  mm_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_203: "bf16[4, 512, 32, 128]" = torch.ops.aten.reshape.default(view_202, [4, 512, 32, 128]);  view_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_142: "bf16[4, 32, 512, 128]" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_1 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_142, add_46, view_133, view_134, getitem_54, getitem_55, getitem_60, getitem_61, where, None, None, 512, 512, 0.0, False, scale = 0.08838834764831845);  permute_142 = add_46 = view_133 = view_134 = getitem_54 = getitem_55 = getitem_60 = getitem_61 = None
        getitem_75: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward_1[0]
        getitem_76: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward_1[1]
        getitem_77: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward_1[2];  _scaled_dot_product_cudnn_attention_backward_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_204: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.reshape.default(getitem_77, [4, 8, 4, 512, 128]);  getitem_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_15: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.sum.dim_IntList(view_204, [2], True);  view_204 = None
        squeeze_3: "bf16[4, 8, 512, 128]" = torch.ops.aten.squeeze.dim(sum_15, 2);  sum_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_205: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.reshape.default(getitem_76, [4, 8, 4, 512, 128]);  getitem_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_16: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.sum.dim_IntList(view_205, [2], True);  view_205 = None
        squeeze_4: "bf16[4, 8, 512, 128]" = torch.ops.aten.squeeze.dim(sum_16, 2);  sum_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_127: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(squeeze_4, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_41: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(mul_127, 3, 0, 64)
        slice_42: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(mul_127, 3, 64, 128);  mul_127 = None
        neg_29: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_41);  slice_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_4: "bf16[4, 8, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_22, neg_29, 3, 64, 9223372036854775807);  neg_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_5: "bf16[4, 8, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_22, slice_42, 3, 0, 64);  slice_42 = None
        add_80: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(slice_scatter_4, slice_scatter_5);  slice_scatter_4 = slice_scatter_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_128: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(squeeze_4, unsqueeze_14);  squeeze_4 = None
        add_81: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(add_80, mul_128);  add_80 = mul_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_129: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(getitem_75, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_43: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(mul_129, 3, 0, 64)
        slice_44: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(mul_129, 3, 64, 128);  mul_129 = None
        neg_30: "bf16[4, 32, 512, 64]" = torch.ops.aten.neg.default(slice_43);  slice_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_6: "bf16[4, 32, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_24, neg_30, 3, 64, 9223372036854775807);  neg_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_7: "bf16[4, 32, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_24, slice_44, 3, 0, 64);  slice_44 = None
        add_82: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(slice_scatter_6, slice_scatter_7);  slice_scatter_6 = slice_scatter_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_130: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(getitem_75, unsqueeze_14);  getitem_75 = None
        add_83: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(add_82, mul_130);  add_82 = mul_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_143: "bf16[4, 512, 8, 128]" = torch.ops.aten.permute.default(squeeze_3, [0, 2, 1, 3]);  squeeze_3 = None
        clone_22: "bf16[4, 512, 8, 128]" = torch.ops.aten.clone.default(permute_143, memory_format = torch.contiguous_format);  permute_143 = None
        view_206: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_22, [4, 512, 1024]);  clone_22 = None
        view_207: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(view_206, [2048, 1024]);  view_206 = None
        permute_144: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_207, [1, 0])
        mm_81: "bf16[1024, 4096]" = torch.ops.aten.mm.default(permute_144, view_124);  permute_144 = None
        permute_71: "bf16[4096, 1024]" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
        permute_146: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
        mm_82: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_207, permute_146);  view_207 = permute_146 = None
        view_208: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_82, [4, 512, 4096]);  mm_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_148: "bf16[4, 512, 8, 128]" = torch.ops.aten.permute.default(add_81, [0, 2, 1, 3]);  add_81 = None
        clone_23: "bf16[4, 512, 8, 128]" = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
        view_209: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_23, [4, 512, 1024]);  clone_23 = None
        view_210: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(view_209, [2048, 1024]);  view_209 = None
        permute_149: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_210, [1, 0])
        mm_83: "bf16[1024, 4096]" = torch.ops.aten.mm.default(permute_149, view_124);  permute_149 = None
        permute_69: "bf16[4096, 1024]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
        permute_151: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
        mm_84: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_210, permute_151);  view_210 = permute_151 = None
        view_211: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_84, [4, 512, 4096]);  mm_84 = None
        add_84: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(view_208, view_211);  view_208 = view_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_153: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(add_83, [0, 2, 1, 3]);  add_83 = None
        clone_24: "bf16[4, 512, 32, 128]" = torch.ops.aten.clone.default(permute_153, memory_format = torch.contiguous_format);  permute_153 = None
        view_212: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(clone_24, [4, 512, 4096]);  clone_24 = None
        view_213: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_212, [2048, 4096]);  view_212 = None
        permute_154: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_213, [1, 0])
        mm_85: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_154, view_124);  permute_154 = view_124 = None
        permute_67: "bf16[4096, 4096]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
        permute_156: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
        mm_86: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_213, permute_156);  view_213 = permute_156 = None
        view_214: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_86, [4, 512, 4096]);  mm_86 = None
        add_85: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_84, view_214);  add_84 = view_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_131: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_85, primals_58);  primals_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_123: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_44, torch.float32);  add_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_57: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_123, rsqrt_12)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_124: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_57, torch.bfloat16);  mul_57 = None
        mul_132: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_85, convert_element_type_124);  add_85 = convert_element_type_124 = None
        sum_17: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_132, [0, 1], True);  mul_132 = None
        view_215: "bf16[4096]" = torch.ops.aten.reshape.default(sum_17, [4096]);  sum_17 = None
        convert_element_type_244: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_131, torch.float32);  mul_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_133: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_244, convert_element_type_123)
        mul_134: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_244, rsqrt_12);  convert_element_type_244 = None
        sum_18: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_133, [2], True);  mul_133 = None
        pow_26: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_12, 3);  rsqrt_12 = None
        mul_135: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_18, -0.5);  sum_18 = None
        mul_136: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_135, pow_26);  mul_135 = pow_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_26: "f32[4, 512, 4096]" = torch.ops.aten.expand.default(mul_136, [4, 512, 4096]);  mul_136 = None
        div_14: "f32[4, 512, 4096]" = torch.ops.aten.div.Scalar(expand_26, 4096);  expand_26 = None
        pow_27: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_123, 1.0);  convert_element_type_123 = None
        mul_137: "f32[4, 512, 4096]" = torch.ops.aten.mul.Scalar(pow_27, 2.0);  pow_27 = None
        mul_138: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(div_14, mul_137);  div_14 = mul_137 = None
        add_86: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_134, mul_138);  mul_134 = mul_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_245: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_86, torch.bfloat16);  add_86 = None
        add_87: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_79, convert_element_type_245);  add_79 = convert_element_type_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_216: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(add_87, [2048, 4096])
        permute_158: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_216, [1, 0])
        mm_87: "bf16[4096, 14336]" = torch.ops.aten.mm.default(permute_158, view_122);  permute_158 = view_122 = None
        permute_66: "bf16[14336, 4096]" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
        permute_160: "bf16[4096, 14336]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        mm_88: "bf16[2048, 14336]" = torch.ops.aten.mm.default(view_216, permute_160);  view_216 = permute_160 = None
        view_217: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_88, [4, 512, 14336]);  mm_88 = None
        view_119: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_39, [4, 512, 14336]);  mm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_117: "f32[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(view_119, torch.float32);  view_119 = None
        neg_17: "f32[4, 512, 14336]" = torch.ops.aten.neg.default(convert_element_type_117)
        exp_5: "f32[4, 512, 14336]" = torch.ops.aten.exp.default(neg_17);  neg_17 = None
        add_43: "f32[4, 512, 14336]" = torch.ops.aten.add.Tensor(exp_5, 1);  exp_5 = None
        div_5: "f32[4, 512, 14336]" = torch.ops.aten.div.Tensor(convert_element_type_117, add_43)
        convert_element_type_118: "bf16[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_139: "bf16[4, 512, 14336]" = torch.ops.aten.mul.Tensor(view_217, convert_element_type_118);  convert_element_type_118 = None
        view_121: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_40, [4, 512, 14336]);  mm_40 = None
        mul_140: "bf16[4, 512, 14336]" = torch.ops.aten.mul.Tensor(view_217, view_121);  view_217 = view_121 = None
        view_218: "bf16[2048, 14336]" = torch.ops.aten.reshape.default(mul_139, [2048, 14336]);  mul_139 = None
        permute_162: "bf16[14336, 2048]" = torch.ops.aten.permute.default(view_218, [1, 0])
        mm_89: "bf16[14336, 4096]" = torch.ops.aten.mm.default(permute_162, view_118);  permute_162 = None
        permute_65: "bf16[4096, 14336]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
        permute_164: "bf16[14336, 4096]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        mm_90: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_218, permute_164);  view_218 = permute_164 = None
        view_219: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_90, [4, 512, 4096]);  mm_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_254: "f32[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(mul_140, torch.float32);  mul_140 = None
        reciprocal_2: "f32[4, 512, 14336]" = torch.ops.aten.reciprocal.default(add_43);  add_43 = None
        mul_141: "f32[4, 512, 14336]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
        mul_142: "f32[4, 512, 14336]" = torch.ops.aten.mul.Tensor(convert_element_type_254, mul_141);  convert_element_type_254 = None
        sub_7: "f32[4, 512, 14336]" = torch.ops.aten.sub.Tensor(1, mul_141);  mul_141 = None
        mul_143: "f32[4, 512, 14336]" = torch.ops.aten.mul.Tensor(convert_element_type_117, sub_7);  convert_element_type_117 = sub_7 = None
        add_89: "f32[4, 512, 14336]" = torch.ops.aten.add.Tensor(mul_143, 1);  mul_143 = None
        mul_144: "f32[4, 512, 14336]" = torch.ops.aten.mul.Tensor(mul_142, add_89);  mul_142 = add_89 = None
        convert_element_type_256: "bf16[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(mul_144, torch.bfloat16);  mul_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_220: "bf16[2048, 14336]" = torch.ops.aten.reshape.default(convert_element_type_256, [2048, 14336]);  convert_element_type_256 = None
        permute_166: "bf16[14336, 2048]" = torch.ops.aten.permute.default(view_220, [1, 0])
        mm_91: "bf16[14336, 4096]" = torch.ops.aten.mm.default(permute_166, view_118);  permute_166 = view_118 = None
        permute_64: "bf16[4096, 14336]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
        permute_168: "bf16[14336, 4096]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        mm_92: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_220, permute_168);  view_220 = permute_168 = None
        view_221: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_92, [4, 512, 4096]);  mm_92 = None
        add_90: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(view_219, view_221);  view_219 = view_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_145: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_90, primals_54);  primals_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_113: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_41, torch.float32);  add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_54: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_113, rsqrt_11)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_114: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_54, torch.bfloat16);  mul_54 = None
        mul_146: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_90, convert_element_type_114);  add_90 = convert_element_type_114 = None
        sum_19: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_146, [0, 1], True);  mul_146 = None
        view_222: "bf16[4096]" = torch.ops.aten.reshape.default(sum_19, [4096]);  sum_19 = None
        convert_element_type_261: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_145, torch.float32);  mul_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_147: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_261, convert_element_type_113)
        mul_148: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_261, rsqrt_11);  convert_element_type_261 = None
        sum_20: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_147, [2], True);  mul_147 = None
        pow_28: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_11, 3);  rsqrt_11 = None
        mul_149: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_20, -0.5);  sum_20 = None
        mul_150: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_149, pow_28);  mul_149 = pow_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_27: "f32[4, 512, 4096]" = torch.ops.aten.expand.default(mul_150, [4, 512, 4096]);  mul_150 = None
        div_15: "f32[4, 512, 4096]" = torch.ops.aten.div.Scalar(expand_27, 4096);  expand_27 = None
        pow_29: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_113, 1.0);  convert_element_type_113 = None
        mul_151: "f32[4, 512, 4096]" = torch.ops.aten.mul.Scalar(pow_29, 2.0);  pow_29 = None
        mul_152: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(div_15, mul_151);  div_15 = mul_151 = None
        add_91: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_148, mul_152);  mul_148 = mul_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_262: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_91, torch.bfloat16);  add_91 = None
        add_92: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_87, convert_element_type_262);  add_87 = convert_element_type_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_223: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(add_92, [2048, 4096])
        permute_170: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_223, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_62: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_45, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_115: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_62, [4, 512, -1]);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_116: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_115, [2048, 4096]);  view_115 = None
        mm_93: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_170, view_116);  permute_170 = view_116 = None
        permute_63: "bf16[4096, 4096]" = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
        permute_172: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        mm_94: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_223, permute_172);  view_223 = permute_172 = None
        view_224: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_94, [4, 512, 4096]);  mm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_225: "bf16[4, 512, 32, 128]" = torch.ops.aten.reshape.default(view_224, [4, 512, 32, 128]);  view_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_174: "bf16[4, 32, 512, 128]" = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_2 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_174, add_39, view_113, view_114, getitem_45, getitem_46, getitem_51, getitem_52, where, None, None, 512, 512, 0.0, False, scale = 0.08838834764831845);  permute_174 = add_39 = view_113 = view_114 = getitem_45 = getitem_46 = getitem_51 = getitem_52 = None
        getitem_78: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward_2[0]
        getitem_79: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward_2[1]
        getitem_80: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward_2[2];  _scaled_dot_product_cudnn_attention_backward_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_226: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.reshape.default(getitem_80, [4, 8, 4, 512, 128]);  getitem_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_21: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.sum.dim_IntList(view_226, [2], True);  view_226 = None
        squeeze_5: "bf16[4, 8, 512, 128]" = torch.ops.aten.squeeze.dim(sum_21, 2);  sum_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_227: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.reshape.default(getitem_79, [4, 8, 4, 512, 128]);  getitem_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_22: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.sum.dim_IntList(view_227, [2], True);  view_227 = None
        squeeze_6: "bf16[4, 8, 512, 128]" = torch.ops.aten.squeeze.dim(sum_22, 2);  sum_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_153: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(squeeze_6, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_45: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(mul_153, 3, 0, 64)
        slice_46: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(mul_153, 3, 64, 128);  mul_153 = None
        neg_32: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_45);  slice_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_8: "bf16[4, 8, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_22, neg_32, 3, 64, 9223372036854775807);  neg_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_9: "bf16[4, 8, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_22, slice_46, 3, 0, 64);  slice_46 = None
        add_93: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(slice_scatter_8, slice_scatter_9);  slice_scatter_8 = slice_scatter_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_154: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(squeeze_6, unsqueeze_14);  squeeze_6 = None
        add_94: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(add_93, mul_154);  add_93 = mul_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_155: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(getitem_78, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_47: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(mul_155, 3, 0, 64)
        slice_48: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(mul_155, 3, 64, 128);  mul_155 = None
        neg_33: "bf16[4, 32, 512, 64]" = torch.ops.aten.neg.default(slice_47);  slice_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_10: "bf16[4, 32, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_24, neg_33, 3, 64, 9223372036854775807);  neg_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_11: "bf16[4, 32, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_24, slice_48, 3, 0, 64);  slice_48 = None
        add_95: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(slice_scatter_10, slice_scatter_11);  slice_scatter_10 = slice_scatter_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_156: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(getitem_78, unsqueeze_14);  getitem_78 = None
        add_96: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(add_95, mul_156);  add_95 = mul_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_175: "bf16[4, 512, 8, 128]" = torch.ops.aten.permute.default(squeeze_5, [0, 2, 1, 3]);  squeeze_5 = None
        clone_25: "bf16[4, 512, 8, 128]" = torch.ops.aten.clone.default(permute_175, memory_format = torch.contiguous_format);  permute_175 = None
        view_228: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_25, [4, 512, 1024]);  clone_25 = None
        view_229: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(view_228, [2048, 1024]);  view_228 = None
        permute_176: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_229, [1, 0])
        mm_95: "bf16[1024, 4096]" = torch.ops.aten.mm.default(permute_176, view_104);  permute_176 = None
        permute_60: "bf16[4096, 1024]" = torch.ops.aten.permute.default(primals_52, [1, 0]);  primals_52 = None
        permute_178: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
        mm_96: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_229, permute_178);  view_229 = permute_178 = None
        view_230: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_96, [4, 512, 4096]);  mm_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_180: "bf16[4, 512, 8, 128]" = torch.ops.aten.permute.default(add_94, [0, 2, 1, 3]);  add_94 = None
        clone_26: "bf16[4, 512, 8, 128]" = torch.ops.aten.clone.default(permute_180, memory_format = torch.contiguous_format);  permute_180 = None
        view_231: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_26, [4, 512, 1024]);  clone_26 = None
        view_232: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(view_231, [2048, 1024]);  view_231 = None
        permute_181: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_232, [1, 0])
        mm_97: "bf16[1024, 4096]" = torch.ops.aten.mm.default(permute_181, view_104);  permute_181 = None
        permute_58: "bf16[4096, 1024]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
        permute_183: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
        mm_98: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_232, permute_183);  view_232 = permute_183 = None
        view_233: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_98, [4, 512, 4096]);  mm_98 = None
        add_97: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(view_230, view_233);  view_230 = view_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_185: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(add_96, [0, 2, 1, 3]);  add_96 = None
        clone_27: "bf16[4, 512, 32, 128]" = torch.ops.aten.clone.default(permute_185, memory_format = torch.contiguous_format);  permute_185 = None
        view_234: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(clone_27, [4, 512, 4096]);  clone_27 = None
        view_235: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_234, [2048, 4096]);  view_234 = None
        permute_186: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_235, [1, 0])
        mm_99: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_186, view_104);  permute_186 = view_104 = None
        permute_56: "bf16[4096, 4096]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
        permute_188: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
        mm_100: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_235, permute_188);  view_235 = permute_188 = None
        view_236: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_100, [4, 512, 4096]);  mm_100 = None
        add_98: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_97, view_236);  add_97 = view_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_157: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_98, primals_49);  primals_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_103: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_37, torch.float32);  add_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_48: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_103, rsqrt_10)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_104: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_48, torch.bfloat16);  mul_48 = None
        mul_158: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_98, convert_element_type_104);  add_98 = convert_element_type_104 = None
        sum_23: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_158, [0, 1], True);  mul_158 = None
        view_237: "bf16[4096]" = torch.ops.aten.reshape.default(sum_23, [4096]);  sum_23 = None
        convert_element_type_279: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_157, torch.float32);  mul_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_159: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_279, convert_element_type_103)
        mul_160: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_279, rsqrt_10);  convert_element_type_279 = None
        sum_24: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_159, [2], True);  mul_159 = None
        pow_30: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_10, 3);  rsqrt_10 = None
        mul_161: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_24, -0.5);  sum_24 = None
        mul_162: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_161, pow_30);  mul_161 = pow_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_28: "f32[4, 512, 4096]" = torch.ops.aten.expand.default(mul_162, [4, 512, 4096]);  mul_162 = None
        div_16: "f32[4, 512, 4096]" = torch.ops.aten.div.Scalar(expand_28, 4096);  expand_28 = None
        pow_31: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_103, 1.0);  convert_element_type_103 = None
        mul_163: "f32[4, 512, 4096]" = torch.ops.aten.mul.Scalar(pow_31, 2.0);  pow_31 = None
        mul_164: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(div_16, mul_163);  div_16 = mul_163 = None
        add_99: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_160, mul_164);  mul_160 = mul_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_280: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_99, torch.bfloat16);  add_99 = None
        add_100: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_92, convert_element_type_280);  add_92 = convert_element_type_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_238: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(add_100, [2048, 4096])
        permute_190: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_238, [1, 0])
        mm_101: "bf16[4096, 14336]" = torch.ops.aten.mm.default(permute_190, view_102);  permute_190 = view_102 = None
        permute_55: "bf16[14336, 4096]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
        permute_192: "bf16[4096, 14336]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        mm_102: "bf16[2048, 14336]" = torch.ops.aten.mm.default(view_238, permute_192);  view_238 = permute_192 = None
        view_239: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_102, [4, 512, 14336]);  mm_102 = None
        view_99: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_32, [4, 512, 14336]);  mm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_97: "f32[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(view_99, torch.float32);  view_99 = None
        neg_14: "f32[4, 512, 14336]" = torch.ops.aten.neg.default(convert_element_type_97)
        exp_4: "f32[4, 512, 14336]" = torch.ops.aten.exp.default(neg_14);  neg_14 = None
        add_36: "f32[4, 512, 14336]" = torch.ops.aten.add.Tensor(exp_4, 1);  exp_4 = None
        div_4: "f32[4, 512, 14336]" = torch.ops.aten.div.Tensor(convert_element_type_97, add_36)
        convert_element_type_98: "bf16[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_165: "bf16[4, 512, 14336]" = torch.ops.aten.mul.Tensor(view_239, convert_element_type_98);  convert_element_type_98 = None
        view_101: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_33, [4, 512, 14336]);  mm_33 = None
        mul_166: "bf16[4, 512, 14336]" = torch.ops.aten.mul.Tensor(view_239, view_101);  view_239 = view_101 = None
        view_240: "bf16[2048, 14336]" = torch.ops.aten.reshape.default(mul_165, [2048, 14336]);  mul_165 = None
        permute_194: "bf16[14336, 2048]" = torch.ops.aten.permute.default(view_240, [1, 0])
        mm_103: "bf16[14336, 4096]" = torch.ops.aten.mm.default(permute_194, view_98);  permute_194 = None
        permute_54: "bf16[4096, 14336]" = torch.ops.aten.permute.default(primals_47, [1, 0]);  primals_47 = None
        permute_196: "bf16[14336, 4096]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        mm_104: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_240, permute_196);  view_240 = permute_196 = None
        view_241: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_104, [4, 512, 4096]);  mm_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_289: "f32[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(mul_166, torch.float32);  mul_166 = None
        reciprocal_3: "f32[4, 512, 14336]" = torch.ops.aten.reciprocal.default(add_36);  add_36 = None
        mul_167: "f32[4, 512, 14336]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
        mul_168: "f32[4, 512, 14336]" = torch.ops.aten.mul.Tensor(convert_element_type_289, mul_167);  convert_element_type_289 = None
        sub_8: "f32[4, 512, 14336]" = torch.ops.aten.sub.Tensor(1, mul_167);  mul_167 = None
        mul_169: "f32[4, 512, 14336]" = torch.ops.aten.mul.Tensor(convert_element_type_97, sub_8);  convert_element_type_97 = sub_8 = None
        add_102: "f32[4, 512, 14336]" = torch.ops.aten.add.Tensor(mul_169, 1);  mul_169 = None
        mul_170: "f32[4, 512, 14336]" = torch.ops.aten.mul.Tensor(mul_168, add_102);  mul_168 = add_102 = None
        convert_element_type_291: "bf16[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(mul_170, torch.bfloat16);  mul_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_242: "bf16[2048, 14336]" = torch.ops.aten.reshape.default(convert_element_type_291, [2048, 14336]);  convert_element_type_291 = None
        permute_198: "bf16[14336, 2048]" = torch.ops.aten.permute.default(view_242, [1, 0])
        mm_105: "bf16[14336, 4096]" = torch.ops.aten.mm.default(permute_198, view_98);  permute_198 = view_98 = None
        permute_53: "bf16[4096, 14336]" = torch.ops.aten.permute.default(primals_46, [1, 0]);  primals_46 = None
        permute_200: "bf16[14336, 4096]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        mm_106: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_242, permute_200);  view_242 = permute_200 = None
        view_243: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_106, [4, 512, 4096]);  mm_106 = None
        add_103: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(view_241, view_243);  view_241 = view_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_171: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_103, primals_45);  primals_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_93: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_34, torch.float32);  add_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_45: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_93, rsqrt_9)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_94: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_45, torch.bfloat16);  mul_45 = None
        mul_172: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_103, convert_element_type_94);  add_103 = convert_element_type_94 = None
        sum_25: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_172, [0, 1], True);  mul_172 = None
        view_244: "bf16[4096]" = torch.ops.aten.reshape.default(sum_25, [4096]);  sum_25 = None
        convert_element_type_296: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_171, torch.float32);  mul_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_173: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_296, convert_element_type_93)
        mul_174: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_296, rsqrt_9);  convert_element_type_296 = None
        sum_26: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_173, [2], True);  mul_173 = None
        pow_32: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_9, 3);  rsqrt_9 = None
        mul_175: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_26, -0.5);  sum_26 = None
        mul_176: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_175, pow_32);  mul_175 = pow_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_29: "f32[4, 512, 4096]" = torch.ops.aten.expand.default(mul_176, [4, 512, 4096]);  mul_176 = None
        div_17: "f32[4, 512, 4096]" = torch.ops.aten.div.Scalar(expand_29, 4096);  expand_29 = None
        pow_33: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_93, 1.0);  convert_element_type_93 = None
        mul_177: "f32[4, 512, 4096]" = torch.ops.aten.mul.Scalar(pow_33, 2.0);  pow_33 = None
        mul_178: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(div_17, mul_177);  div_17 = mul_177 = None
        add_104: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_174, mul_178);  mul_174 = mul_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_297: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_104, torch.bfloat16);  add_104 = None
        add_105: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_100, convert_element_type_297);  add_100 = convert_element_type_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_245: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(add_105, [2048, 4096])
        permute_202: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_245, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_51: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_36, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_95: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_51, [4, 512, -1]);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_96: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_95, [2048, 4096]);  view_95 = None
        mm_107: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_202, view_96);  permute_202 = view_96 = None
        permute_52: "bf16[4096, 4096]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
        permute_204: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        mm_108: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_245, permute_204);  view_245 = permute_204 = None
        view_246: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_108, [4, 512, 4096]);  mm_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_247: "bf16[4, 512, 32, 128]" = torch.ops.aten.reshape.default(view_246, [4, 512, 32, 128]);  view_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_206: "bf16[4, 32, 512, 128]" = torch.ops.aten.permute.default(view_247, [0, 2, 1, 3]);  view_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_3 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_206, add_32, view_93, view_94, getitem_36, getitem_37, getitem_42, getitem_43, where, None, None, 512, 512, 0.0, False, scale = 0.08838834764831845);  permute_206 = add_32 = view_93 = view_94 = getitem_36 = getitem_37 = getitem_42 = getitem_43 = None
        getitem_81: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward_3[0]
        getitem_82: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward_3[1]
        getitem_83: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward_3[2];  _scaled_dot_product_cudnn_attention_backward_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_248: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.reshape.default(getitem_83, [4, 8, 4, 512, 128]);  getitem_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_27: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.sum.dim_IntList(view_248, [2], True);  view_248 = None
        squeeze_7: "bf16[4, 8, 512, 128]" = torch.ops.aten.squeeze.dim(sum_27, 2);  sum_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_249: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.reshape.default(getitem_82, [4, 8, 4, 512, 128]);  getitem_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_28: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.sum.dim_IntList(view_249, [2], True);  view_249 = None
        squeeze_8: "bf16[4, 8, 512, 128]" = torch.ops.aten.squeeze.dim(sum_28, 2);  sum_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_179: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(squeeze_8, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_49: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(mul_179, 3, 0, 64)
        slice_50: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(mul_179, 3, 64, 128);  mul_179 = None
        neg_35: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_49);  slice_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_12: "bf16[4, 8, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_22, neg_35, 3, 64, 9223372036854775807);  neg_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_13: "bf16[4, 8, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_22, slice_50, 3, 0, 64);  slice_50 = None
        add_106: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(slice_scatter_12, slice_scatter_13);  slice_scatter_12 = slice_scatter_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_180: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(squeeze_8, unsqueeze_14);  squeeze_8 = None
        add_107: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(add_106, mul_180);  add_106 = mul_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_181: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(getitem_81, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_51: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(mul_181, 3, 0, 64)
        slice_52: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(mul_181, 3, 64, 128);  mul_181 = None
        neg_36: "bf16[4, 32, 512, 64]" = torch.ops.aten.neg.default(slice_51);  slice_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_14: "bf16[4, 32, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_24, neg_36, 3, 64, 9223372036854775807);  neg_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_15: "bf16[4, 32, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_24, slice_52, 3, 0, 64);  slice_52 = None
        add_108: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(slice_scatter_14, slice_scatter_15);  slice_scatter_14 = slice_scatter_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_182: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(getitem_81, unsqueeze_14);  getitem_81 = None
        add_109: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(add_108, mul_182);  add_108 = mul_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_207: "bf16[4, 512, 8, 128]" = torch.ops.aten.permute.default(squeeze_7, [0, 2, 1, 3]);  squeeze_7 = None
        clone_28: "bf16[4, 512, 8, 128]" = torch.ops.aten.clone.default(permute_207, memory_format = torch.contiguous_format);  permute_207 = None
        view_250: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_28, [4, 512, 1024]);  clone_28 = None
        view_251: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(view_250, [2048, 1024]);  view_250 = None
        permute_208: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_251, [1, 0])
        mm_109: "bf16[1024, 4096]" = torch.ops.aten.mm.default(permute_208, view_84);  permute_208 = None
        permute_49: "bf16[4096, 1024]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
        permute_210: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
        mm_110: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_251, permute_210);  view_251 = permute_210 = None
        view_252: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_110, [4, 512, 4096]);  mm_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_212: "bf16[4, 512, 8, 128]" = torch.ops.aten.permute.default(add_107, [0, 2, 1, 3]);  add_107 = None
        clone_29: "bf16[4, 512, 8, 128]" = torch.ops.aten.clone.default(permute_212, memory_format = torch.contiguous_format);  permute_212 = None
        view_253: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_29, [4, 512, 1024]);  clone_29 = None
        view_254: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(view_253, [2048, 1024]);  view_253 = None
        permute_213: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_254, [1, 0])
        mm_111: "bf16[1024, 4096]" = torch.ops.aten.mm.default(permute_213, view_84);  permute_213 = None
        permute_47: "bf16[4096, 1024]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
        permute_215: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
        mm_112: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_254, permute_215);  view_254 = permute_215 = None
        view_255: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_112, [4, 512, 4096]);  mm_112 = None
        add_110: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(view_252, view_255);  view_252 = view_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_217: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(add_109, [0, 2, 1, 3]);  add_109 = None
        clone_30: "bf16[4, 512, 32, 128]" = torch.ops.aten.clone.default(permute_217, memory_format = torch.contiguous_format);  permute_217 = None
        view_256: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(clone_30, [4, 512, 4096]);  clone_30 = None
        view_257: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_256, [2048, 4096]);  view_256 = None
        permute_218: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_257, [1, 0])
        mm_113: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_218, view_84);  permute_218 = view_84 = None
        permute_45: "bf16[4096, 4096]" = torch.ops.aten.permute.default(primals_41, [1, 0]);  primals_41 = None
        permute_220: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
        mm_114: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_257, permute_220);  view_257 = permute_220 = None
        view_258: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_114, [4, 512, 4096]);  mm_114 = None
        add_111: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_110, view_258);  add_110 = view_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_183: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_111, primals_40);  primals_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_83: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_30, torch.float32);  add_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_39: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_83, rsqrt_8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_84: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_39, torch.bfloat16);  mul_39 = None
        mul_184: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_111, convert_element_type_84);  add_111 = convert_element_type_84 = None
        sum_29: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_184, [0, 1], True);  mul_184 = None
        view_259: "bf16[4096]" = torch.ops.aten.reshape.default(sum_29, [4096]);  sum_29 = None
        convert_element_type_314: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_183, torch.float32);  mul_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_185: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_314, convert_element_type_83)
        mul_186: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_314, rsqrt_8);  convert_element_type_314 = None
        sum_30: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_185, [2], True);  mul_185 = None
        pow_34: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_8, 3);  rsqrt_8 = None
        mul_187: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_30, -0.5);  sum_30 = None
        mul_188: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_187, pow_34);  mul_187 = pow_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_30: "f32[4, 512, 4096]" = torch.ops.aten.expand.default(mul_188, [4, 512, 4096]);  mul_188 = None
        div_18: "f32[4, 512, 4096]" = torch.ops.aten.div.Scalar(expand_30, 4096);  expand_30 = None
        pow_35: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_83, 1.0);  convert_element_type_83 = None
        mul_189: "f32[4, 512, 4096]" = torch.ops.aten.mul.Scalar(pow_35, 2.0);  pow_35 = None
        mul_190: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(div_18, mul_189);  div_18 = mul_189 = None
        add_112: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_186, mul_190);  mul_186 = mul_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_315: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_112, torch.bfloat16);  add_112 = None
        add_113: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_105, convert_element_type_315);  add_105 = convert_element_type_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_260: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(add_113, [2048, 4096])
        permute_222: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_260, [1, 0])
        mm_115: "bf16[4096, 14336]" = torch.ops.aten.mm.default(permute_222, view_82);  permute_222 = view_82 = None
        permute_44: "bf16[14336, 4096]" = torch.ops.aten.permute.default(primals_39, [1, 0]);  primals_39 = None
        permute_224: "bf16[4096, 14336]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        mm_116: "bf16[2048, 14336]" = torch.ops.aten.mm.default(view_260, permute_224);  view_260 = permute_224 = None
        view_261: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_116, [4, 512, 14336]);  mm_116 = None
        view_79: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_25, [4, 512, 14336]);  mm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_77: "f32[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(view_79, torch.float32);  view_79 = None
        neg_11: "f32[4, 512, 14336]" = torch.ops.aten.neg.default(convert_element_type_77)
        exp_3: "f32[4, 512, 14336]" = torch.ops.aten.exp.default(neg_11);  neg_11 = None
        add_29: "f32[4, 512, 14336]" = torch.ops.aten.add.Tensor(exp_3, 1);  exp_3 = None
        div_3: "f32[4, 512, 14336]" = torch.ops.aten.div.Tensor(convert_element_type_77, add_29)
        convert_element_type_78: "bf16[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_191: "bf16[4, 512, 14336]" = torch.ops.aten.mul.Tensor(view_261, convert_element_type_78);  convert_element_type_78 = None
        view_81: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_26, [4, 512, 14336]);  mm_26 = None
        mul_192: "bf16[4, 512, 14336]" = torch.ops.aten.mul.Tensor(view_261, view_81);  view_261 = view_81 = None
        view_262: "bf16[2048, 14336]" = torch.ops.aten.reshape.default(mul_191, [2048, 14336]);  mul_191 = None
        permute_226: "bf16[14336, 2048]" = torch.ops.aten.permute.default(view_262, [1, 0])
        mm_117: "bf16[14336, 4096]" = torch.ops.aten.mm.default(permute_226, view_78);  permute_226 = None
        permute_43: "bf16[4096, 14336]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
        permute_228: "bf16[14336, 4096]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        mm_118: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_262, permute_228);  view_262 = permute_228 = None
        view_263: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_118, [4, 512, 4096]);  mm_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_324: "f32[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(mul_192, torch.float32);  mul_192 = None
        reciprocal_4: "f32[4, 512, 14336]" = torch.ops.aten.reciprocal.default(add_29);  add_29 = None
        mul_193: "f32[4, 512, 14336]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
        mul_194: "f32[4, 512, 14336]" = torch.ops.aten.mul.Tensor(convert_element_type_324, mul_193);  convert_element_type_324 = None
        sub_9: "f32[4, 512, 14336]" = torch.ops.aten.sub.Tensor(1, mul_193);  mul_193 = None
        mul_195: "f32[4, 512, 14336]" = torch.ops.aten.mul.Tensor(convert_element_type_77, sub_9);  convert_element_type_77 = sub_9 = None
        add_115: "f32[4, 512, 14336]" = torch.ops.aten.add.Tensor(mul_195, 1);  mul_195 = None
        mul_196: "f32[4, 512, 14336]" = torch.ops.aten.mul.Tensor(mul_194, add_115);  mul_194 = add_115 = None
        convert_element_type_326: "bf16[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(mul_196, torch.bfloat16);  mul_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_264: "bf16[2048, 14336]" = torch.ops.aten.reshape.default(convert_element_type_326, [2048, 14336]);  convert_element_type_326 = None
        permute_230: "bf16[14336, 2048]" = torch.ops.aten.permute.default(view_264, [1, 0])
        mm_119: "bf16[14336, 4096]" = torch.ops.aten.mm.default(permute_230, view_78);  permute_230 = view_78 = None
        permute_42: "bf16[4096, 14336]" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
        permute_232: "bf16[14336, 4096]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        mm_120: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_264, permute_232);  view_264 = permute_232 = None
        view_265: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_120, [4, 512, 4096]);  mm_120 = None
        add_116: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(view_263, view_265);  view_263 = view_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_197: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_116, primals_36);  primals_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_73: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_27, torch.float32);  add_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_36: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_73, rsqrt_7)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_74: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_36, torch.bfloat16);  mul_36 = None
        mul_198: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_116, convert_element_type_74);  add_116 = convert_element_type_74 = None
        sum_31: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_198, [0, 1], True);  mul_198 = None
        view_266: "bf16[4096]" = torch.ops.aten.reshape.default(sum_31, [4096]);  sum_31 = None
        convert_element_type_331: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_197, torch.float32);  mul_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_199: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_331, convert_element_type_73)
        mul_200: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_331, rsqrt_7);  convert_element_type_331 = None
        sum_32: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_199, [2], True);  mul_199 = None
        pow_36: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_7, 3);  rsqrt_7 = None
        mul_201: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_32, -0.5);  sum_32 = None
        mul_202: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_201, pow_36);  mul_201 = pow_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_31: "f32[4, 512, 4096]" = torch.ops.aten.expand.default(mul_202, [4, 512, 4096]);  mul_202 = None
        div_19: "f32[4, 512, 4096]" = torch.ops.aten.div.Scalar(expand_31, 4096);  expand_31 = None
        pow_37: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_73, 1.0);  convert_element_type_73 = None
        mul_203: "f32[4, 512, 4096]" = torch.ops.aten.mul.Scalar(pow_37, 2.0);  pow_37 = None
        mul_204: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(div_19, mul_203);  div_19 = mul_203 = None
        add_117: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_200, mul_204);  mul_200 = mul_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_332: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_117, torch.bfloat16);  add_117 = None
        add_118: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_113, convert_element_type_332);  add_113 = convert_element_type_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_267: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(add_118, [2048, 4096])
        permute_234: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_267, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_40: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_75: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_40, [4, 512, -1]);  permute_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_76: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_75, [2048, 4096]);  view_75 = None
        mm_121: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_234, view_76);  permute_234 = view_76 = None
        permute_41: "bf16[4096, 4096]" = torch.ops.aten.permute.default(primals_35, [1, 0]);  primals_35 = None
        permute_236: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        mm_122: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_267, permute_236);  view_267 = permute_236 = None
        view_268: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_122, [4, 512, 4096]);  mm_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_269: "bf16[4, 512, 32, 128]" = torch.ops.aten.reshape.default(view_268, [4, 512, 32, 128]);  view_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_238: "bf16[4, 32, 512, 128]" = torch.ops.aten.permute.default(view_269, [0, 2, 1, 3]);  view_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_4 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_238, add_25, view_73, view_74, getitem_27, getitem_28, getitem_33, getitem_34, where, None, None, 512, 512, 0.0, False, scale = 0.08838834764831845);  permute_238 = add_25 = view_73 = view_74 = getitem_27 = getitem_28 = getitem_33 = getitem_34 = None
        getitem_84: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward_4[0]
        getitem_85: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward_4[1]
        getitem_86: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward_4[2];  _scaled_dot_product_cudnn_attention_backward_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_270: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.reshape.default(getitem_86, [4, 8, 4, 512, 128]);  getitem_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_33: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.sum.dim_IntList(view_270, [2], True);  view_270 = None
        squeeze_9: "bf16[4, 8, 512, 128]" = torch.ops.aten.squeeze.dim(sum_33, 2);  sum_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_271: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.reshape.default(getitem_85, [4, 8, 4, 512, 128]);  getitem_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_34: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.sum.dim_IntList(view_271, [2], True);  view_271 = None
        squeeze_10: "bf16[4, 8, 512, 128]" = torch.ops.aten.squeeze.dim(sum_34, 2);  sum_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_205: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(squeeze_10, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_53: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(mul_205, 3, 0, 64)
        slice_54: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(mul_205, 3, 64, 128);  mul_205 = None
        neg_38: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_53);  slice_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_16: "bf16[4, 8, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_22, neg_38, 3, 64, 9223372036854775807);  neg_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_17: "bf16[4, 8, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_22, slice_54, 3, 0, 64);  slice_54 = None
        add_119: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(slice_scatter_16, slice_scatter_17);  slice_scatter_16 = slice_scatter_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_206: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(squeeze_10, unsqueeze_14);  squeeze_10 = None
        add_120: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(add_119, mul_206);  add_119 = mul_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_207: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(getitem_84, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_55: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(mul_207, 3, 0, 64)
        slice_56: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(mul_207, 3, 64, 128);  mul_207 = None
        neg_39: "bf16[4, 32, 512, 64]" = torch.ops.aten.neg.default(slice_55);  slice_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_18: "bf16[4, 32, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_24, neg_39, 3, 64, 9223372036854775807);  neg_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_19: "bf16[4, 32, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_24, slice_56, 3, 0, 64);  slice_56 = None
        add_121: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(slice_scatter_18, slice_scatter_19);  slice_scatter_18 = slice_scatter_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_208: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(getitem_84, unsqueeze_14);  getitem_84 = None
        add_122: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(add_121, mul_208);  add_121 = mul_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_239: "bf16[4, 512, 8, 128]" = torch.ops.aten.permute.default(squeeze_9, [0, 2, 1, 3]);  squeeze_9 = None
        clone_31: "bf16[4, 512, 8, 128]" = torch.ops.aten.clone.default(permute_239, memory_format = torch.contiguous_format);  permute_239 = None
        view_272: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_31, [4, 512, 1024]);  clone_31 = None
        view_273: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(view_272, [2048, 1024]);  view_272 = None
        permute_240: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_273, [1, 0])
        mm_123: "bf16[1024, 4096]" = torch.ops.aten.mm.default(permute_240, view_64);  permute_240 = None
        permute_38: "bf16[4096, 1024]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
        permute_242: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
        mm_124: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_273, permute_242);  view_273 = permute_242 = None
        view_274: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_124, [4, 512, 4096]);  mm_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_244: "bf16[4, 512, 8, 128]" = torch.ops.aten.permute.default(add_120, [0, 2, 1, 3]);  add_120 = None
        clone_32: "bf16[4, 512, 8, 128]" = torch.ops.aten.clone.default(permute_244, memory_format = torch.contiguous_format);  permute_244 = None
        view_275: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_32, [4, 512, 1024]);  clone_32 = None
        view_276: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(view_275, [2048, 1024]);  view_275 = None
        permute_245: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_276, [1, 0])
        mm_125: "bf16[1024, 4096]" = torch.ops.aten.mm.default(permute_245, view_64);  permute_245 = None
        permute_36: "bf16[4096, 1024]" = torch.ops.aten.permute.default(primals_33, [1, 0]);  primals_33 = None
        permute_247: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
        mm_126: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_276, permute_247);  view_276 = permute_247 = None
        view_277: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_126, [4, 512, 4096]);  mm_126 = None
        add_123: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(view_274, view_277);  view_274 = view_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_249: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(add_122, [0, 2, 1, 3]);  add_122 = None
        clone_33: "bf16[4, 512, 32, 128]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
        view_278: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(clone_33, [4, 512, 4096]);  clone_33 = None
        view_279: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_278, [2048, 4096]);  view_278 = None
        permute_250: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_279, [1, 0])
        mm_127: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_250, view_64);  permute_250 = view_64 = None
        permute_34: "bf16[4096, 4096]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
        permute_252: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        mm_128: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_279, permute_252);  view_279 = permute_252 = None
        view_280: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_128, [4, 512, 4096]);  mm_128 = None
        add_124: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_123, view_280);  add_123 = view_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_209: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_124, primals_31);  primals_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_63: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_23, torch.float32);  add_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_30: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_63, rsqrt_6)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_64: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_30, torch.bfloat16);  mul_30 = None
        mul_210: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_124, convert_element_type_64);  add_124 = convert_element_type_64 = None
        sum_35: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_210, [0, 1], True);  mul_210 = None
        view_281: "bf16[4096]" = torch.ops.aten.reshape.default(sum_35, [4096]);  sum_35 = None
        convert_element_type_349: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_209, torch.float32);  mul_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_211: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_349, convert_element_type_63)
        mul_212: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_349, rsqrt_6);  convert_element_type_349 = None
        sum_36: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_211, [2], True);  mul_211 = None
        pow_38: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_6, 3);  rsqrt_6 = None
        mul_213: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_36, -0.5);  sum_36 = None
        mul_214: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_213, pow_38);  mul_213 = pow_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_32: "f32[4, 512, 4096]" = torch.ops.aten.expand.default(mul_214, [4, 512, 4096]);  mul_214 = None
        div_20: "f32[4, 512, 4096]" = torch.ops.aten.div.Scalar(expand_32, 4096);  expand_32 = None
        pow_39: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_63, 1.0);  convert_element_type_63 = None
        mul_215: "f32[4, 512, 4096]" = torch.ops.aten.mul.Scalar(pow_39, 2.0);  pow_39 = None
        mul_216: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(div_20, mul_215);  div_20 = mul_215 = None
        add_125: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_212, mul_216);  mul_212 = mul_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_350: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_125, torch.bfloat16);  add_125 = None
        add_126: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_118, convert_element_type_350);  add_118 = convert_element_type_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_282: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(add_126, [2048, 4096])
        permute_254: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_282, [1, 0])
        mm_129: "bf16[4096, 14336]" = torch.ops.aten.mm.default(permute_254, view_62);  permute_254 = view_62 = None
        permute_33: "bf16[14336, 4096]" = torch.ops.aten.permute.default(primals_30, [1, 0]);  primals_30 = None
        permute_256: "bf16[4096, 14336]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        mm_130: "bf16[2048, 14336]" = torch.ops.aten.mm.default(view_282, permute_256);  view_282 = permute_256 = None
        view_283: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_130, [4, 512, 14336]);  mm_130 = None
        view_59: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_18, [4, 512, 14336]);  mm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_57: "f32[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(view_59, torch.float32);  view_59 = None
        neg_8: "f32[4, 512, 14336]" = torch.ops.aten.neg.default(convert_element_type_57)
        exp_2: "f32[4, 512, 14336]" = torch.ops.aten.exp.default(neg_8);  neg_8 = None
        add_22: "f32[4, 512, 14336]" = torch.ops.aten.add.Tensor(exp_2, 1);  exp_2 = None
        div_2: "f32[4, 512, 14336]" = torch.ops.aten.div.Tensor(convert_element_type_57, add_22)
        convert_element_type_58: "bf16[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_217: "bf16[4, 512, 14336]" = torch.ops.aten.mul.Tensor(view_283, convert_element_type_58);  convert_element_type_58 = None
        view_61: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_19, [4, 512, 14336]);  mm_19 = None
        mul_218: "bf16[4, 512, 14336]" = torch.ops.aten.mul.Tensor(view_283, view_61);  view_283 = view_61 = None
        view_284: "bf16[2048, 14336]" = torch.ops.aten.reshape.default(mul_217, [2048, 14336]);  mul_217 = None
        permute_258: "bf16[14336, 2048]" = torch.ops.aten.permute.default(view_284, [1, 0])
        mm_131: "bf16[14336, 4096]" = torch.ops.aten.mm.default(permute_258, view_58);  permute_258 = None
        permute_32: "bf16[4096, 14336]" = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
        permute_260: "bf16[14336, 4096]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        mm_132: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_284, permute_260);  view_284 = permute_260 = None
        view_285: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_132, [4, 512, 4096]);  mm_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_359: "f32[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(mul_218, torch.float32);  mul_218 = None
        reciprocal_5: "f32[4, 512, 14336]" = torch.ops.aten.reciprocal.default(add_22);  add_22 = None
        mul_219: "f32[4, 512, 14336]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
        mul_220: "f32[4, 512, 14336]" = torch.ops.aten.mul.Tensor(convert_element_type_359, mul_219);  convert_element_type_359 = None
        sub_10: "f32[4, 512, 14336]" = torch.ops.aten.sub.Tensor(1, mul_219);  mul_219 = None
        mul_221: "f32[4, 512, 14336]" = torch.ops.aten.mul.Tensor(convert_element_type_57, sub_10);  convert_element_type_57 = sub_10 = None
        add_128: "f32[4, 512, 14336]" = torch.ops.aten.add.Tensor(mul_221, 1);  mul_221 = None
        mul_222: "f32[4, 512, 14336]" = torch.ops.aten.mul.Tensor(mul_220, add_128);  mul_220 = add_128 = None
        convert_element_type_361: "bf16[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(mul_222, torch.bfloat16);  mul_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_286: "bf16[2048, 14336]" = torch.ops.aten.reshape.default(convert_element_type_361, [2048, 14336]);  convert_element_type_361 = None
        permute_262: "bf16[14336, 2048]" = torch.ops.aten.permute.default(view_286, [1, 0])
        mm_133: "bf16[14336, 4096]" = torch.ops.aten.mm.default(permute_262, view_58);  permute_262 = view_58 = None
        permute_31: "bf16[4096, 14336]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
        permute_264: "bf16[14336, 4096]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        mm_134: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_286, permute_264);  view_286 = permute_264 = None
        view_287: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_134, [4, 512, 4096]);  mm_134 = None
        add_129: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(view_285, view_287);  view_285 = view_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_223: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_129, primals_27);  primals_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_53: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_20, torch.float32);  add_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_27: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_53, rsqrt_5)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_54: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_27, torch.bfloat16);  mul_27 = None
        mul_224: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_129, convert_element_type_54);  add_129 = convert_element_type_54 = None
        sum_37: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_224, [0, 1], True);  mul_224 = None
        view_288: "bf16[4096]" = torch.ops.aten.reshape.default(sum_37, [4096]);  sum_37 = None
        convert_element_type_366: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_223, torch.float32);  mul_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_225: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_366, convert_element_type_53)
        mul_226: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_366, rsqrt_5);  convert_element_type_366 = None
        sum_38: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_225, [2], True);  mul_225 = None
        pow_40: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_5, 3);  rsqrt_5 = None
        mul_227: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_38, -0.5);  sum_38 = None
        mul_228: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_227, pow_40);  mul_227 = pow_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_33: "f32[4, 512, 4096]" = torch.ops.aten.expand.default(mul_228, [4, 512, 4096]);  mul_228 = None
        div_21: "f32[4, 512, 4096]" = torch.ops.aten.div.Scalar(expand_33, 4096);  expand_33 = None
        pow_41: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_53, 1.0);  convert_element_type_53 = None
        mul_229: "f32[4, 512, 4096]" = torch.ops.aten.mul.Scalar(pow_41, 2.0);  pow_41 = None
        mul_230: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(div_21, mul_229);  div_21 = mul_229 = None
        add_130: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_226, mul_230);  mul_226 = mul_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_367: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_130, torch.bfloat16);  add_130 = None
        add_131: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_126, convert_element_type_367);  add_126 = convert_element_type_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_289: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(add_131, [2048, 4096])
        permute_266: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_289, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_29: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_18, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_55: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_29, [4, 512, -1]);  permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_56: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_55, [2048, 4096]);  view_55 = None
        mm_135: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_266, view_56);  permute_266 = view_56 = None
        permute_30: "bf16[4096, 4096]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
        permute_268: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        mm_136: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_289, permute_268);  view_289 = permute_268 = None
        view_290: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_136, [4, 512, 4096]);  mm_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_291: "bf16[4, 512, 32, 128]" = torch.ops.aten.reshape.default(view_290, [4, 512, 32, 128]);  view_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_270: "bf16[4, 32, 512, 128]" = torch.ops.aten.permute.default(view_291, [0, 2, 1, 3]);  view_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_5 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_270, add_18, view_53, view_54, getitem_18, getitem_19, getitem_24, getitem_25, where, None, None, 512, 512, 0.0, False, scale = 0.08838834764831845);  permute_270 = add_18 = view_53 = view_54 = getitem_18 = getitem_19 = getitem_24 = getitem_25 = None
        getitem_87: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward_5[0]
        getitem_88: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward_5[1]
        getitem_89: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward_5[2];  _scaled_dot_product_cudnn_attention_backward_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_292: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.reshape.default(getitem_89, [4, 8, 4, 512, 128]);  getitem_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_39: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.sum.dim_IntList(view_292, [2], True);  view_292 = None
        squeeze_11: "bf16[4, 8, 512, 128]" = torch.ops.aten.squeeze.dim(sum_39, 2);  sum_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_293: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.reshape.default(getitem_88, [4, 8, 4, 512, 128]);  getitem_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_40: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.sum.dim_IntList(view_293, [2], True);  view_293 = None
        squeeze_12: "bf16[4, 8, 512, 128]" = torch.ops.aten.squeeze.dim(sum_40, 2);  sum_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_231: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(squeeze_12, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_57: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(mul_231, 3, 0, 64)
        slice_58: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(mul_231, 3, 64, 128);  mul_231 = None
        neg_41: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_57);  slice_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_20: "bf16[4, 8, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_22, neg_41, 3, 64, 9223372036854775807);  neg_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_21: "bf16[4, 8, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_22, slice_58, 3, 0, 64);  slice_58 = None
        add_132: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(slice_scatter_20, slice_scatter_21);  slice_scatter_20 = slice_scatter_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_232: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(squeeze_12, unsqueeze_14);  squeeze_12 = None
        add_133: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(add_132, mul_232);  add_132 = mul_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_233: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(getitem_87, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_59: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(mul_233, 3, 0, 64)
        slice_60: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(mul_233, 3, 64, 128);  mul_233 = None
        neg_42: "bf16[4, 32, 512, 64]" = torch.ops.aten.neg.default(slice_59);  slice_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_22: "bf16[4, 32, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_24, neg_42, 3, 64, 9223372036854775807);  neg_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_23: "bf16[4, 32, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_24, slice_60, 3, 0, 64);  slice_60 = None
        add_134: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(slice_scatter_22, slice_scatter_23);  slice_scatter_22 = slice_scatter_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_234: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(getitem_87, unsqueeze_14);  getitem_87 = None
        add_135: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(add_134, mul_234);  add_134 = mul_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_271: "bf16[4, 512, 8, 128]" = torch.ops.aten.permute.default(squeeze_11, [0, 2, 1, 3]);  squeeze_11 = None
        clone_34: "bf16[4, 512, 8, 128]" = torch.ops.aten.clone.default(permute_271, memory_format = torch.contiguous_format);  permute_271 = None
        view_294: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_34, [4, 512, 1024]);  clone_34 = None
        view_295: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(view_294, [2048, 1024]);  view_294 = None
        permute_272: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_295, [1, 0])
        mm_137: "bf16[1024, 4096]" = torch.ops.aten.mm.default(permute_272, view_44);  permute_272 = None
        permute_27: "bf16[4096, 1024]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
        permute_274: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
        mm_138: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_295, permute_274);  view_295 = permute_274 = None
        view_296: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_138, [4, 512, 4096]);  mm_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_276: "bf16[4, 512, 8, 128]" = torch.ops.aten.permute.default(add_133, [0, 2, 1, 3]);  add_133 = None
        clone_35: "bf16[4, 512, 8, 128]" = torch.ops.aten.clone.default(permute_276, memory_format = torch.contiguous_format);  permute_276 = None
        view_297: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_35, [4, 512, 1024]);  clone_35 = None
        view_298: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(view_297, [2048, 1024]);  view_297 = None
        permute_277: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_298, [1, 0])
        mm_139: "bf16[1024, 4096]" = torch.ops.aten.mm.default(permute_277, view_44);  permute_277 = None
        permute_25: "bf16[4096, 1024]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
        permute_279: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
        mm_140: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_298, permute_279);  view_298 = permute_279 = None
        view_299: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_140, [4, 512, 4096]);  mm_140 = None
        add_136: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(view_296, view_299);  view_296 = view_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_281: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(add_135, [0, 2, 1, 3]);  add_135 = None
        clone_36: "bf16[4, 512, 32, 128]" = torch.ops.aten.clone.default(permute_281, memory_format = torch.contiguous_format);  permute_281 = None
        view_300: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(clone_36, [4, 512, 4096]);  clone_36 = None
        view_301: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_300, [2048, 4096]);  view_300 = None
        permute_282: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_301, [1, 0])
        mm_141: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_282, view_44);  permute_282 = view_44 = None
        permute_23: "bf16[4096, 4096]" = torch.ops.aten.permute.default(primals_23, [1, 0]);  primals_23 = None
        permute_284: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        mm_142: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_301, permute_284);  view_301 = permute_284 = None
        view_302: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_142, [4, 512, 4096]);  mm_142 = None
        add_137: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_136, view_302);  add_136 = view_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_235: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_137, primals_22);  primals_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_43: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_16, torch.float32);  add_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_21: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_43, rsqrt_4)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_44: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_21, torch.bfloat16);  mul_21 = None
        mul_236: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_137, convert_element_type_44);  add_137 = convert_element_type_44 = None
        sum_41: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_236, [0, 1], True);  mul_236 = None
        view_303: "bf16[4096]" = torch.ops.aten.reshape.default(sum_41, [4096]);  sum_41 = None
        convert_element_type_384: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_235, torch.float32);  mul_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_237: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_384, convert_element_type_43)
        mul_238: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_384, rsqrt_4);  convert_element_type_384 = None
        sum_42: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_237, [2], True);  mul_237 = None
        pow_42: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_4, 3);  rsqrt_4 = None
        mul_239: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_42, -0.5);  sum_42 = None
        mul_240: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_239, pow_42);  mul_239 = pow_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_34: "f32[4, 512, 4096]" = torch.ops.aten.expand.default(mul_240, [4, 512, 4096]);  mul_240 = None
        div_22: "f32[4, 512, 4096]" = torch.ops.aten.div.Scalar(expand_34, 4096);  expand_34 = None
        pow_43: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_43, 1.0);  convert_element_type_43 = None
        mul_241: "f32[4, 512, 4096]" = torch.ops.aten.mul.Scalar(pow_43, 2.0);  pow_43 = None
        mul_242: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(div_22, mul_241);  div_22 = mul_241 = None
        add_138: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_238, mul_242);  mul_238 = mul_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_385: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_138, torch.bfloat16);  add_138 = None
        add_139: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_131, convert_element_type_385);  add_131 = convert_element_type_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_304: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(add_139, [2048, 4096])
        permute_286: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_304, [1, 0])
        mm_143: "bf16[4096, 14336]" = torch.ops.aten.mm.default(permute_286, view_42);  permute_286 = view_42 = None
        permute_22: "bf16[14336, 4096]" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
        permute_288: "bf16[4096, 14336]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        mm_144: "bf16[2048, 14336]" = torch.ops.aten.mm.default(view_304, permute_288);  view_304 = permute_288 = None
        view_305: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_144, [4, 512, 14336]);  mm_144 = None
        view_39: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_11, [4, 512, 14336]);  mm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_37: "f32[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(view_39, torch.float32);  view_39 = None
        neg_5: "f32[4, 512, 14336]" = torch.ops.aten.neg.default(convert_element_type_37)
        exp_1: "f32[4, 512, 14336]" = torch.ops.aten.exp.default(neg_5);  neg_5 = None
        add_15: "f32[4, 512, 14336]" = torch.ops.aten.add.Tensor(exp_1, 1);  exp_1 = None
        div_1: "f32[4, 512, 14336]" = torch.ops.aten.div.Tensor(convert_element_type_37, add_15)
        convert_element_type_38: "bf16[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_243: "bf16[4, 512, 14336]" = torch.ops.aten.mul.Tensor(view_305, convert_element_type_38);  convert_element_type_38 = None
        view_41: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_12, [4, 512, 14336]);  mm_12 = None
        mul_244: "bf16[4, 512, 14336]" = torch.ops.aten.mul.Tensor(view_305, view_41);  view_305 = view_41 = None
        view_306: "bf16[2048, 14336]" = torch.ops.aten.reshape.default(mul_243, [2048, 14336]);  mul_243 = None
        permute_290: "bf16[14336, 2048]" = torch.ops.aten.permute.default(view_306, [1, 0])
        mm_145: "bf16[14336, 4096]" = torch.ops.aten.mm.default(permute_290, view_38);  permute_290 = None
        permute_21: "bf16[4096, 14336]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
        permute_292: "bf16[14336, 4096]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        mm_146: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_306, permute_292);  view_306 = permute_292 = None
        view_307: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_146, [4, 512, 4096]);  mm_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_394: "f32[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(mul_244, torch.float32);  mul_244 = None
        reciprocal_6: "f32[4, 512, 14336]" = torch.ops.aten.reciprocal.default(add_15);  add_15 = None
        mul_245: "f32[4, 512, 14336]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
        mul_246: "f32[4, 512, 14336]" = torch.ops.aten.mul.Tensor(convert_element_type_394, mul_245);  convert_element_type_394 = None
        sub_11: "f32[4, 512, 14336]" = torch.ops.aten.sub.Tensor(1, mul_245);  mul_245 = None
        mul_247: "f32[4, 512, 14336]" = torch.ops.aten.mul.Tensor(convert_element_type_37, sub_11);  convert_element_type_37 = sub_11 = None
        add_141: "f32[4, 512, 14336]" = torch.ops.aten.add.Tensor(mul_247, 1);  mul_247 = None
        mul_248: "f32[4, 512, 14336]" = torch.ops.aten.mul.Tensor(mul_246, add_141);  mul_246 = add_141 = None
        convert_element_type_396: "bf16[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(mul_248, torch.bfloat16);  mul_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_308: "bf16[2048, 14336]" = torch.ops.aten.reshape.default(convert_element_type_396, [2048, 14336]);  convert_element_type_396 = None
        permute_294: "bf16[14336, 2048]" = torch.ops.aten.permute.default(view_308, [1, 0])
        mm_147: "bf16[14336, 4096]" = torch.ops.aten.mm.default(permute_294, view_38);  permute_294 = view_38 = None
        permute_20: "bf16[4096, 14336]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
        permute_296: "bf16[14336, 4096]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        mm_148: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_308, permute_296);  view_308 = permute_296 = None
        view_309: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_148, [4, 512, 4096]);  mm_148 = None
        add_142: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(view_307, view_309);  view_307 = view_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_249: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_142, primals_18);  primals_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_33: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_13, torch.float32);  add_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_18: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_33, rsqrt_3)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_34: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_18, torch.bfloat16);  mul_18 = None
        mul_250: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_142, convert_element_type_34);  add_142 = convert_element_type_34 = None
        sum_43: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_250, [0, 1], True);  mul_250 = None
        view_310: "bf16[4096]" = torch.ops.aten.reshape.default(sum_43, [4096]);  sum_43 = None
        convert_element_type_401: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_249, torch.float32);  mul_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_251: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_401, convert_element_type_33)
        mul_252: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_401, rsqrt_3);  convert_element_type_401 = None
        sum_44: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [2], True);  mul_251 = None
        pow_44: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_3, 3);  rsqrt_3 = None
        mul_253: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_44, -0.5);  sum_44 = None
        mul_254: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_253, pow_44);  mul_253 = pow_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_35: "f32[4, 512, 4096]" = torch.ops.aten.expand.default(mul_254, [4, 512, 4096]);  mul_254 = None
        div_23: "f32[4, 512, 4096]" = torch.ops.aten.div.Scalar(expand_35, 4096);  expand_35 = None
        pow_45: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_33, 1.0);  convert_element_type_33 = None
        mul_255: "f32[4, 512, 4096]" = torch.ops.aten.mul.Scalar(pow_45, 2.0);  pow_45 = None
        mul_256: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(div_23, mul_255);  div_23 = mul_255 = None
        add_143: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_252, mul_256);  mul_252 = mul_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_402: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_143, torch.bfloat16);  add_143 = None
        add_144: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_139, convert_element_type_402);  add_139 = convert_element_type_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_311: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(add_144, [2048, 4096])
        permute_298: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_311, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_18: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem_9, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_35: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_18, [4, 512, -1]);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_36: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_35, [2048, 4096]);  view_35 = None
        mm_149: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_298, view_36);  permute_298 = view_36 = None
        permute_19: "bf16[4096, 4096]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
        permute_300: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        mm_150: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_311, permute_300);  view_311 = permute_300 = None
        view_312: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_150, [4, 512, 4096]);  mm_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_313: "bf16[4, 512, 32, 128]" = torch.ops.aten.reshape.default(view_312, [4, 512, 32, 128]);  view_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_302: "bf16[4, 32, 512, 128]" = torch.ops.aten.permute.default(view_313, [0, 2, 1, 3]);  view_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_6 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_302, add_11, view_33, view_34, getitem_9, getitem_10, getitem_15, getitem_16, where, None, None, 512, 512, 0.0, False, scale = 0.08838834764831845);  permute_302 = add_11 = view_33 = view_34 = getitem_9 = getitem_10 = getitem_15 = getitem_16 = None
        getitem_90: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward_6[0]
        getitem_91: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward_6[1]
        getitem_92: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward_6[2];  _scaled_dot_product_cudnn_attention_backward_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_314: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.reshape.default(getitem_92, [4, 8, 4, 512, 128]);  getitem_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_45: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.sum.dim_IntList(view_314, [2], True);  view_314 = None
        squeeze_13: "bf16[4, 8, 512, 128]" = torch.ops.aten.squeeze.dim(sum_45, 2);  sum_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_315: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.reshape.default(getitem_91, [4, 8, 4, 512, 128]);  getitem_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_46: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.sum.dim_IntList(view_315, [2], True);  view_315 = None
        squeeze_14: "bf16[4, 8, 512, 128]" = torch.ops.aten.squeeze.dim(sum_46, 2);  sum_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_257: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(squeeze_14, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_61: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(mul_257, 3, 0, 64)
        slice_62: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(mul_257, 3, 64, 128);  mul_257 = None
        neg_44: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_61);  slice_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_24: "bf16[4, 8, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_22, neg_44, 3, 64, 9223372036854775807);  neg_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_25: "bf16[4, 8, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_22, slice_62, 3, 0, 64);  slice_62 = None
        add_145: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(slice_scatter_24, slice_scatter_25);  slice_scatter_24 = slice_scatter_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_258: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(squeeze_14, unsqueeze_14);  squeeze_14 = None
        add_146: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(add_145, mul_258);  add_145 = mul_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_259: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(getitem_90, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_63: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(mul_259, 3, 0, 64)
        slice_64: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(mul_259, 3, 64, 128);  mul_259 = None
        neg_45: "bf16[4, 32, 512, 64]" = torch.ops.aten.neg.default(slice_63);  slice_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_26: "bf16[4, 32, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_24, neg_45, 3, 64, 9223372036854775807);  neg_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_27: "bf16[4, 32, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_24, slice_64, 3, 0, 64);  slice_64 = None
        add_147: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(slice_scatter_26, slice_scatter_27);  slice_scatter_26 = slice_scatter_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_260: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(getitem_90, unsqueeze_14);  getitem_90 = None
        add_148: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(add_147, mul_260);  add_147 = mul_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_303: "bf16[4, 512, 8, 128]" = torch.ops.aten.permute.default(squeeze_13, [0, 2, 1, 3]);  squeeze_13 = None
        clone_37: "bf16[4, 512, 8, 128]" = torch.ops.aten.clone.default(permute_303, memory_format = torch.contiguous_format);  permute_303 = None
        view_316: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_37, [4, 512, 1024]);  clone_37 = None
        view_317: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(view_316, [2048, 1024]);  view_316 = None
        permute_304: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_317, [1, 0])
        mm_151: "bf16[1024, 4096]" = torch.ops.aten.mm.default(permute_304, view_24);  permute_304 = None
        permute_16: "bf16[4096, 1024]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        permute_306: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
        mm_152: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_317, permute_306);  view_317 = permute_306 = None
        view_318: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_152, [4, 512, 4096]);  mm_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_308: "bf16[4, 512, 8, 128]" = torch.ops.aten.permute.default(add_146, [0, 2, 1, 3]);  add_146 = None
        clone_38: "bf16[4, 512, 8, 128]" = torch.ops.aten.clone.default(permute_308, memory_format = torch.contiguous_format);  permute_308 = None
        view_319: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_38, [4, 512, 1024]);  clone_38 = None
        view_320: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(view_319, [2048, 1024]);  view_319 = None
        permute_309: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_320, [1, 0])
        mm_153: "bf16[1024, 4096]" = torch.ops.aten.mm.default(permute_309, view_24);  permute_309 = None
        permute_14: "bf16[4096, 1024]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
        permute_311: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
        mm_154: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_320, permute_311);  view_320 = permute_311 = None
        view_321: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_154, [4, 512, 4096]);  mm_154 = None
        add_149: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(view_318, view_321);  view_318 = view_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_313: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(add_148, [0, 2, 1, 3]);  add_148 = None
        clone_39: "bf16[4, 512, 32, 128]" = torch.ops.aten.clone.default(permute_313, memory_format = torch.contiguous_format);  permute_313 = None
        view_322: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(clone_39, [4, 512, 4096]);  clone_39 = None
        view_323: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_322, [2048, 4096]);  view_322 = None
        permute_314: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_323, [1, 0])
        mm_155: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_314, view_24);  permute_314 = view_24 = None
        permute_12: "bf16[4096, 4096]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
        permute_316: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        mm_156: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_323, permute_316);  view_323 = permute_316 = None
        view_324: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_156, [4, 512, 4096]);  mm_156 = None
        add_150: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_149, view_324);  add_149 = view_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_261: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_150, primals_13);  primals_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_23: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_9, torch.float32);  add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_12: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_23, rsqrt_2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_24: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_12, torch.bfloat16);  mul_12 = None
        mul_262: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_150, convert_element_type_24);  add_150 = convert_element_type_24 = None
        sum_47: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_262, [0, 1], True);  mul_262 = None
        view_325: "bf16[4096]" = torch.ops.aten.reshape.default(sum_47, [4096]);  sum_47 = None
        convert_element_type_419: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_261, torch.float32);  mul_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_263: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_419, convert_element_type_23)
        mul_264: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_419, rsqrt_2);  convert_element_type_419 = None
        sum_48: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_263, [2], True);  mul_263 = None
        pow_46: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_2, 3);  rsqrt_2 = None
        mul_265: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_48, -0.5);  sum_48 = None
        mul_266: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_265, pow_46);  mul_265 = pow_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_36: "f32[4, 512, 4096]" = torch.ops.aten.expand.default(mul_266, [4, 512, 4096]);  mul_266 = None
        div_24: "f32[4, 512, 4096]" = torch.ops.aten.div.Scalar(expand_36, 4096);  expand_36 = None
        pow_47: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_23, 1.0);  convert_element_type_23 = None
        mul_267: "f32[4, 512, 4096]" = torch.ops.aten.mul.Scalar(pow_47, 2.0);  pow_47 = None
        mul_268: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(div_24, mul_267);  div_24 = mul_267 = None
        add_151: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_264, mul_268);  mul_264 = mul_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_420: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_151, torch.bfloat16);  add_151 = None
        add_152: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_144, convert_element_type_420);  add_144 = convert_element_type_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_326: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(add_152, [2048, 4096])
        permute_318: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_326, [1, 0])
        mm_157: "bf16[4096, 14336]" = torch.ops.aten.mm.default(permute_318, view_22);  permute_318 = view_22 = None
        permute_11: "bf16[14336, 4096]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        permute_320: "bf16[4096, 14336]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        mm_158: "bf16[2048, 14336]" = torch.ops.aten.mm.default(view_326, permute_320);  view_326 = permute_320 = None
        view_327: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_158, [4, 512, 14336]);  mm_158 = None
        view_19: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_4, [4, 512, 14336]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_17: "f32[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        neg_2: "f32[4, 512, 14336]" = torch.ops.aten.neg.default(convert_element_type_17)
        exp: "f32[4, 512, 14336]" = torch.ops.aten.exp.default(neg_2);  neg_2 = None
        add_8: "f32[4, 512, 14336]" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[4, 512, 14336]" = torch.ops.aten.div.Tensor(convert_element_type_17, add_8)
        convert_element_type_18: "bf16[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_269: "bf16[4, 512, 14336]" = torch.ops.aten.mul.Tensor(view_327, convert_element_type_18);  convert_element_type_18 = None
        view_21: "bf16[4, 512, 14336]" = torch.ops.aten.reshape.default(mm_5, [4, 512, 14336]);  mm_5 = None
        mul_270: "bf16[4, 512, 14336]" = torch.ops.aten.mul.Tensor(view_327, view_21);  view_327 = view_21 = None
        view_328: "bf16[2048, 14336]" = torch.ops.aten.reshape.default(mul_269, [2048, 14336]);  mul_269 = None
        permute_322: "bf16[14336, 2048]" = torch.ops.aten.permute.default(view_328, [1, 0])
        mm_159: "bf16[14336, 4096]" = torch.ops.aten.mm.default(permute_322, view_18);  permute_322 = None
        permute_10: "bf16[4096, 14336]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_324: "bf16[14336, 4096]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm_160: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_328, permute_324);  view_328 = permute_324 = None
        view_329: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_160, [4, 512, 4096]);  mm_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_429: "f32[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(mul_270, torch.float32);  mul_270 = None
        reciprocal_7: "f32[4, 512, 14336]" = torch.ops.aten.reciprocal.default(add_8);  add_8 = None
        mul_271: "f32[4, 512, 14336]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
        mul_272: "f32[4, 512, 14336]" = torch.ops.aten.mul.Tensor(convert_element_type_429, mul_271);  convert_element_type_429 = None
        sub_12: "f32[4, 512, 14336]" = torch.ops.aten.sub.Tensor(1, mul_271);  mul_271 = None
        mul_273: "f32[4, 512, 14336]" = torch.ops.aten.mul.Tensor(convert_element_type_17, sub_12);  convert_element_type_17 = sub_12 = None
        add_154: "f32[4, 512, 14336]" = torch.ops.aten.add.Tensor(mul_273, 1);  mul_273 = None
        mul_274: "f32[4, 512, 14336]" = torch.ops.aten.mul.Tensor(mul_272, add_154);  mul_272 = add_154 = None
        convert_element_type_431: "bf16[4, 512, 14336]" = torch.ops.prims.convert_element_type.default(mul_274, torch.bfloat16);  mul_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_330: "bf16[2048, 14336]" = torch.ops.aten.reshape.default(convert_element_type_431, [2048, 14336]);  convert_element_type_431 = None
        permute_326: "bf16[14336, 2048]" = torch.ops.aten.permute.default(view_330, [1, 0])
        mm_161: "bf16[14336, 4096]" = torch.ops.aten.mm.default(permute_326, view_18);  permute_326 = view_18 = None
        permute_9: "bf16[4096, 14336]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        permute_328: "bf16[14336, 4096]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_162: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_330, permute_328);  view_330 = permute_328 = None
        view_331: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_162, [4, 512, 4096]);  mm_162 = None
        add_155: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(view_329, view_331);  view_329 = view_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_275: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_155, primals_9);  primals_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_17: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_3, [4, 512, 4096]);  mm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:233 in forward, code: hidden_states = residual + hidden_states
        add_6: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(embedding, view_17);  view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_13: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_6, torch.float32);  add_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_9: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_13, rsqrt_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_14: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
        mul_276: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_155, convert_element_type_14);  add_155 = convert_element_type_14 = None
        sum_49: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_276, [0, 1], True);  mul_276 = None
        view_332: "bf16[4096]" = torch.ops.aten.reshape.default(sum_49, [4096]);  sum_49 = None
        convert_element_type_436: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_275, torch.float32);  mul_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_277: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_436, convert_element_type_13)
        mul_278: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_436, rsqrt_1);  convert_element_type_436 = None
        sum_50: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_277, [2], True);  mul_277 = None
        pow_48: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_1, 3);  rsqrt_1 = None
        mul_279: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_50, -0.5);  sum_50 = None
        mul_280: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_279, pow_48);  mul_279 = pow_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_37: "f32[4, 512, 4096]" = torch.ops.aten.expand.default(mul_280, [4, 512, 4096]);  mul_280 = None
        div_25: "f32[4, 512, 4096]" = torch.ops.aten.div.Scalar(expand_37, 4096);  expand_37 = None
        pow_49: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_13, 1.0);  convert_element_type_13 = None
        mul_281: "f32[4, 512, 4096]" = torch.ops.aten.mul.Scalar(pow_49, 2.0);  pow_49 = None
        mul_282: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(div_25, mul_281);  div_25 = mul_281 = None
        add_156: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_278, mul_282);  mul_278 = mul_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_437: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_156, torch.bfloat16);  add_156 = None
        add_157: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_152, convert_element_type_437);  add_152 = convert_element_type_437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_333: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(add_157, [2048, 4096])
        permute_330: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_333, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_15: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_7, [4, 512, -1]);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:177 in forward, code: attn_output = self.o_proj(attn_output)
        view_16: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_15, [2048, 4096]);  view_15 = None
        mm_163: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_330, view_16);  permute_330 = view_16 = None
        permute_8: "bf16[4096, 4096]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_332: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        mm_164: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_333, permute_332);  view_333 = permute_332 = None
        view_334: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_164, [4, 512, 4096]);  mm_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:176 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_335: "bf16[4, 512, 32, 128]" = torch.ops.aten.reshape.default(view_334, [4, 512, 32, 128]);  view_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_334: "bf16[4, 32, 512, 128]" = torch.ops.aten.permute.default(view_335, [0, 2, 1, 3]);  view_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_7 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_334, add_4, view_13, view_14, getitem, getitem_1, getitem_6, getitem_7, where, None, None, 512, 512, 0.0, False, scale = 0.08838834764831845);  permute_334 = add_4 = view_13 = view_14 = getitem = getitem_1 = getitem_6 = getitem_7 = where = None
        getitem_93: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward_7[0]
        getitem_94: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward_7[1]
        getitem_95: "bf16[4, 32, 512, 128]" = _scaled_dot_product_cudnn_attention_backward_7[2];  _scaled_dot_product_cudnn_attention_backward_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_336: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.reshape.default(getitem_95, [4, 8, 4, 512, 128]);  getitem_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_51: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.sum.dim_IntList(view_336, [2], True);  view_336 = None
        squeeze_15: "bf16[4, 8, 512, 128]" = torch.ops.aten.squeeze.dim(sum_51, 2);  sum_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        view_337: "bf16[4, 8, 4, 512, 128]" = torch.ops.aten.reshape.default(getitem_94, [4, 8, 4, 512, 128]);  getitem_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        sum_52: "bf16[4, 8, 1, 512, 128]" = torch.ops.aten.sum.dim_IntList(view_337, [2], True);  view_337 = None
        squeeze_16: "bf16[4, 8, 512, 128]" = torch.ops.aten.squeeze.dim(sum_52, 2);  sum_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_283: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(squeeze_16, unsqueeze_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_65: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(mul_283, 3, 0, 64)
        slice_66: "bf16[4, 8, 512, 64]" = torch.ops.aten.slice.Tensor(mul_283, 3, 64, 128);  mul_283 = None
        neg_47: "bf16[4, 8, 512, 64]" = torch.ops.aten.neg.default(slice_65);  slice_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_28: "bf16[4, 8, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_22, neg_47, 3, 64, 9223372036854775807);  neg_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_29: "bf16[4, 8, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_22, slice_66, 3, 0, 64);  full_default_22 = slice_66 = None
        add_158: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(slice_scatter_28, slice_scatter_29);  slice_scatter_28 = slice_scatter_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:80 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_284: "bf16[4, 8, 512, 128]" = torch.ops.aten.mul.Tensor(squeeze_16, unsqueeze_14);  squeeze_16 = None
        add_159: "bf16[4, 8, 512, 128]" = torch.ops.aten.add.Tensor(add_158, mul_284);  add_158 = mul_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_285: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(getitem_93, unsqueeze_15);  unsqueeze_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:55 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_67: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(mul_285, 3, 0, 64)
        slice_68: "bf16[4, 32, 512, 64]" = torch.ops.aten.slice.Tensor(mul_285, 3, 64, 128);  mul_285 = None
        neg_48: "bf16[4, 32, 512, 64]" = torch.ops.aten.neg.default(slice_67);  slice_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:54 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_scatter_30: "bf16[4, 32, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_24, neg_48, 3, 64, 9223372036854775807);  neg_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:53 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_scatter_31: "bf16[4, 32, 512, 128]" = torch.ops.aten.slice_scatter.default(full_default_24, slice_68, 3, 0, 64);  full_default_24 = slice_68 = None
        add_160: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(slice_scatter_30, slice_scatter_31);  slice_scatter_30 = slice_scatter_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:79 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_286: "bf16[4, 32, 512, 128]" = torch.ops.aten.mul.Tensor(getitem_93, unsqueeze_14);  getitem_93 = unsqueeze_14 = None
        add_161: "bf16[4, 32, 512, 128]" = torch.ops.aten.add.Tensor(add_160, mul_286);  add_160 = mul_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_335: "bf16[4, 512, 8, 128]" = torch.ops.aten.permute.default(squeeze_15, [0, 2, 1, 3]);  squeeze_15 = None
        clone_40: "bf16[4, 512, 8, 128]" = torch.ops.aten.clone.default(permute_335, memory_format = torch.contiguous_format);  permute_335 = None
        view_338: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_40, [4, 512, 1024]);  clone_40 = None
        view_339: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(view_338, [2048, 1024]);  view_338 = None
        permute_336: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_339, [1, 0])
        mm_165: "bf16[1024, 4096]" = torch.ops.aten.mm.default(permute_336, view_4);  permute_336 = None
        permute_5: "bf16[4096, 1024]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        permute_338: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        mm_166: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_339, permute_338);  view_339 = permute_338 = None
        view_340: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_166, [4, 512, 4096]);  mm_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_340: "bf16[4, 512, 8, 128]" = torch.ops.aten.permute.default(add_159, [0, 2, 1, 3]);  add_159 = None
        clone_41: "bf16[4, 512, 8, 128]" = torch.ops.aten.clone.default(permute_340, memory_format = torch.contiguous_format);  permute_340 = None
        view_341: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_41, [4, 512, 1024]);  clone_41 = None
        view_342: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(view_341, [2048, 1024]);  view_341 = None
        permute_341: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_342, [1, 0])
        mm_167: "bf16[1024, 4096]" = torch.ops.aten.mm.default(permute_341, view_4);  permute_341 = None
        permute_3: "bf16[4096, 1024]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_343: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        mm_168: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_342, permute_343);  view_342 = permute_343 = None
        view_343: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_168, [4, 512, 4096]);  mm_168 = None
        add_162: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(view_340, view_343);  view_340 = view_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_345: "bf16[4, 512, 32, 128]" = torch.ops.aten.permute.default(add_161, [0, 2, 1, 3]);  add_161 = None
        clone_42: "bf16[4, 512, 32, 128]" = torch.ops.aten.clone.default(permute_345, memory_format = torch.contiguous_format);  permute_345 = None
        view_344: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(clone_42, [4, 512, 4096]);  clone_42 = None
        view_345: "bf16[2048, 4096]" = torch.ops.aten.reshape.default(view_344, [2048, 4096]);  view_344 = None
        permute_346: "bf16[4096, 2048]" = torch.ops.aten.permute.default(view_345, [1, 0])
        mm_169: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_346, view_4);  permute_346 = view_4 = None
        permute_1: "bf16[4096, 4096]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        permute_348: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        mm_170: "bf16[2048, 4096]" = torch.ops.aten.mm.default(view_345, permute_348);  view_345 = permute_348 = None
        view_346: "bf16[4, 512, 4096]" = torch.ops.aten.reshape.default(mm_170, [4, 512, 4096]);  mm_170 = None
        add_163: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_162, view_346);  add_162 = view_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_287: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_163, primals_4);  primals_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_3: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(embedding, torch.float32);  embedding = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_3: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_3, rsqrt)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_4: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        mul_288: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_163, convert_element_type_4);  add_163 = convert_element_type_4 = None
        sum_53: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_288, [0, 1], True);  mul_288 = None
        view_347: "bf16[4096]" = torch.ops.aten.reshape.default(sum_53, [4096]);  sum_53 = None
        convert_element_type_454: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_287, torch.float32);  mul_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_289: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_454, convert_element_type_3)
        mul_290: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_454, rsqrt);  convert_element_type_454 = None
        sum_54: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_289, [2], True);  mul_289 = None
        pow_50: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_291: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_54, -0.5);  sum_54 = None
        mul_292: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_291, pow_50);  mul_291 = pow_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_38: "f32[4, 512, 4096]" = torch.ops.aten.expand.default(mul_292, [4, 512, 4096]);  mul_292 = None
        div_26: "f32[4, 512, 4096]" = torch.ops.aten.div.Scalar(expand_38, 4096);  expand_38 = None
        pow_51: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_3, 1.0);  convert_element_type_3 = None
        mul_293: "f32[4, 512, 4096]" = torch.ops.aten.mul.Scalar(pow_51, 2.0);  pow_51 = None
        mul_294: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(div_26, mul_293);  div_26 = mul_293 = None
        add_164: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_290, mul_294);  mul_290 = mul_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_455: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_164, torch.bfloat16);  add_164 = None
        add_165: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_157, convert_element_type_455);  add_157 = convert_element_type_455 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:362 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        convert_element_type_456: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_165, torch.float32);  add_165 = None
        eq_1: "b8[4, 512]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_48: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
        where_12: "f32[4, 512, 4096]" = torch.ops.aten.where.self(unsqueeze_48, full_default_18, convert_element_type_456);  unsqueeze_48 = full_default_18 = convert_element_type_456 = None
        full_default_55: "f32[32768, 4096]" = torch.ops.aten.full.default([32768, 4096], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[32768, 4096]" = torch.ops.aten.index_put.default(full_default_55, [primals_1], where_12, True);  full_default_55 = primals_1 = where_12 = None
        convert_element_type_457: "bf16[32768, 4096]" = torch.ops.prims.convert_element_type.default(index_put, torch.bfloat16);  index_put = None
        return (None, convert_element_type_457, None, view_347, mm_169, mm_167, mm_165, mm_163, view_332, mm_161, mm_159, mm_157, view_325, mm_155, mm_153, mm_151, mm_149, view_310, mm_147, mm_145, mm_143, view_303, mm_141, mm_139, mm_137, mm_135, view_288, mm_133, mm_131, mm_129, view_281, mm_127, mm_125, mm_123, mm_121, view_266, mm_119, mm_117, mm_115, view_259, mm_113, mm_111, mm_109, mm_107, view_244, mm_105, mm_103, mm_101, view_237, mm_99, mm_97, mm_95, mm_93, view_222, mm_91, mm_89, mm_87, view_215, mm_85, mm_83, mm_81, mm_79, view_200, mm_77, mm_75, mm_73, view_193, mm_71, mm_69, mm_67, mm_65, view_178, mm_63, mm_61, mm_59, view_171, mm_57, None)
