import torch
import torch._inductor.inductor_prims  # registers inductor prims ops
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[32, 512]", primals_2: "i64[1, 512]", primals_3: "i64[1, 512]", primals_4: "f32[32000, 768]", primals_5: "f32[4, 768]", primals_6: "f32[512, 768]", primals_7: "f32[768]", primals_8: "f32[768]", primals_9: "f32[768, 768]", primals_10: "f32[768]", primals_11: "f32[768]", primals_12: "f32[768]", primals_13: "f32[3072, 768]", primals_14: "f32[3072]", primals_15: "f32[768, 3072]", primals_16: "f32[768]", primals_17: "f32[768]", primals_18: "f32[768]", primals_19: "f32[768]", primals_20: "f32[768]", primals_21: "f32[3072, 768]", primals_22: "f32[3072]", primals_23: "f32[768, 3072]", primals_24: "f32[768]", primals_25: "f32[768]", primals_26: "f32[768]", primals_27: "f32[768]", primals_28: "f32[768]", primals_29: "f32[3072, 768]", primals_30: "f32[3072]", primals_31: "f32[768, 3072]", primals_32: "f32[768]", primals_33: "f32[768]", primals_34: "f32[768]", primals_35: "f32[768]", primals_36: "f32[768]", primals_37: "f32[3072, 768]", primals_38: "f32[3072]", primals_39: "f32[768, 3072]", primals_40: "f32[768]", primals_41: "f32[768]", primals_42: "f32[768]", primals_43: "f32[768]", primals_44: "f32[768]", primals_45: "f32[3072, 768]", primals_46: "f32[3072]", primals_47: "f32[768, 3072]", primals_48: "f32[768]", primals_49: "f32[768]", primals_50: "f32[768]", primals_51: "f32[768]", primals_52: "f32[768]", primals_53: "f32[3072, 768]", primals_54: "f32[3072]", primals_55: "f32[768, 3072]", primals_56: "f32[768]", primals_57: "f32[768]", primals_58: "f32[768]", primals_59: "f32[768]", primals_60: "f32[768]", primals_61: "f32[3072, 768]", primals_62: "f32[3072]", primals_63: "f32[768, 3072]", primals_64: "f32[768]", primals_65: "f32[768]", primals_66: "f32[768]", primals_67: "f32[768]", primals_68: "f32[768]", primals_69: "f32[3072, 768]", primals_70: "f32[3072]", primals_71: "f32[768, 3072]", primals_72: "f32[768]", primals_73: "f32[768]", primals_74: "f32[768]", primals_75: "f32[768]", primals_76: "f32[768]", primals_77: "f32[3072, 768]", primals_78: "f32[3072]", primals_79: "f32[768, 3072]", primals_80: "f32[768]", primals_81: "f32[768]", primals_82: "f32[768]", primals_83: "f32[768]", primals_84: "f32[768]", primals_85: "f32[3072, 768]", primals_86: "f32[3072]", primals_87: "f32[768, 3072]", primals_88: "f32[768]", primals_89: "f32[768]", primals_90: "f32[768]", primals_91: "f32[768]", primals_92: "f32[768]", primals_93: "f32[3072, 768]", primals_94: "f32[3072]", primals_95: "f32[768, 3072]", primals_96: "f32[768]", primals_97: "f32[768]", primals_98: "f32[768]", primals_99: "f32[768]", primals_100: "f32[768]", primals_101: "f32[3072, 768]", primals_102: "f32[3072]", primals_103: "f32[768, 3072]", primals_104: "f32[768]", primals_105: "f32[768]", primals_106: "f32[768]", primals_107: "f32[768, 768]", primals_108: "f32[768]", primals_109: "f32[768, 768]", primals_110: "f32[768]", primals_111: "f32[768]", primals_112: "f32[768]", primals_113: "f32[32000]", primals_114: "i64[32, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:479 in forward, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        expand: "i64[32, 512]" = torch.ops.aten.expand.default(primals_2, [32, 512])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:129 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(primals_4, primals_1, 3)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:130 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_1: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(primals_5, expand);  primals_5 = expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:132 in forward, code: embeddings = inputs_embeds + token_type_embeddings
        add: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:134 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_2: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_6, primals_3);  primals_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:135 in forward, code: embeddings += position_embeddings
        add_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:136 in forward, code: embeddings = self.LayerNorm(embeddings)
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean[0]
        getitem_1: "f32[32, 512, 1]" = var_mean[1];  var_mean = None
        add_2: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, primals_7)
        add_3: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_1, primals_8);  mul_1 = primals_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:137 in forward, code: embeddings = self.projection(embeddings)
        view: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_3, [16384, 768]);  add_3 = None
        permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_9, [1, 0])
        addmm: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_10, view, permute);  primals_10 = permute = None
        view_1: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm, [32, 512, 768]);  addmm = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[13]" = torch.ops.prims.inductor_seeds.default(13, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:138 in forward, code: embeddings = self.dropout(embeddings)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_12: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_12, 1e-30);  inductor_random_default_12 = None
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt, view_1);  view_1 = None
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, 1.0);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:176 in forward, code: outputs = self.fourier_transform(hidden_states).real
        convert_element_type: "c64[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_3, torch.complex64)
        _fft_c2c: "c64[32, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type, [1, 2], 0, True);  convert_element_type = None
        view_as_real: "f32[32, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c);  _fft_c2c = None
        select: "f32[32, 512, 768]" = torch.ops.aten.select.int(view_as_real, 3, 0);  view_as_real = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        add_4: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_3, select);  mul_3 = select = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
        getitem_2: "f32[32, 512, 1]" = var_mean_1[0]
        getitem_3: "f32[32, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        add_5: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_1: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_4, getitem_3);  add_4 = getitem_3 = None
        mul_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_5: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_4, primals_11)
        add_6: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_5, primals_12);  mul_5 = primals_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_2: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_6, [16384, 768])
        permute_1: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_13, [1, 0])
        addmm_1: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_14, view_2, permute_1);  primals_14 = permute_1 = None
        view_3: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_1, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_6: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_3, 0.5)
        pow_1: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_3, 3.0)
        mul_7: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_7: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_3, mul_7);  view_3 = mul_7 = None
        mul_8: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_7, 0.7978845608028654);  add_7 = None
        tanh: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_8);  mul_8 = None
        add_8: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_9: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_6, add_8);  mul_6 = add_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        view_4: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_9, [16384, 3072]);  mul_9 = None
        permute_2: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_15, [1, 0])
        addmm_2: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_16, view_4, permute_2);  primals_16 = permute_2 = None
        view_5: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_2, [32, 512, 768]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:229 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_11: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_1: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_11, 1e-30);  inductor_random_default_11 = None
        mul_10: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_1, view_5);  view_5 = None
        mul_11: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_10, 1.0);  mul_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_9: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_11, add_6);  mul_11 = add_6 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
        getitem_4: "f32[32, 512, 1]" = var_mean_2[0]
        getitem_5: "f32[32, 512, 1]" = var_mean_2[1];  var_mean_2 = None
        add_10: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_2: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_9, getitem_5);  add_9 = getitem_5 = None
        mul_12: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        mul_13: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_12, primals_17)
        add_11: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_13, primals_18);  mul_13 = primals_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:176 in forward, code: outputs = self.fourier_transform(hidden_states).real
        convert_element_type_1: "c64[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_11, torch.complex64)
        _fft_c2c_1: "c64[32, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type_1, [1, 2], 0, True);  convert_element_type_1 = None
        view_as_real_1: "f32[32, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_1);  _fft_c2c_1 = None
        select_1: "f32[32, 512, 768]" = torch.ops.aten.select.int(view_as_real_1, 3, 0);  view_as_real_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        add_12: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_11, select_1);  add_11 = select_1 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_12, [2], correction = 0, keepdim = True)
        getitem_6: "f32[32, 512, 1]" = var_mean_3[0]
        getitem_7: "f32[32, 512, 1]" = var_mean_3[1];  var_mean_3 = None
        add_13: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_3: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_12, getitem_7);  add_12 = getitem_7 = None
        mul_14: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        mul_15: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_14, primals_19)
        add_14: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_15, primals_20);  mul_15 = primals_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_6: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_14, [16384, 768])
        permute_3: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_21, [1, 0])
        addmm_3: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_22, view_6, permute_3);  primals_22 = permute_3 = None
        view_7: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_3, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_16: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_7, 0.5)
        pow_2: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_7, 3.0)
        mul_17: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_15: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_7, mul_17);  view_7 = mul_17 = None
        mul_18: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_15, 0.7978845608028654);  add_15 = None
        tanh_1: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_18);  mul_18 = None
        add_16: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_19: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_16, add_16);  mul_16 = add_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        view_8: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_19, [16384, 3072]);  mul_19 = None
        permute_4: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_23, [1, 0])
        addmm_4: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_24, view_8, permute_4);  primals_24 = permute_4 = None
        view_9: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_4, [32, 512, 768]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:229 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_10: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        gt_2: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_10, 1e-30);  inductor_random_default_10 = None
        mul_20: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_2, view_9);  view_9 = None
        mul_21: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_20, 1.0);  mul_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_17: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_21, add_14);  mul_21 = add_14 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_8: "f32[32, 512, 1]" = var_mean_4[0]
        getitem_9: "f32[32, 512, 1]" = var_mean_4[1];  var_mean_4 = None
        add_18: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_4: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_17, getitem_9);  add_17 = getitem_9 = None
        mul_22: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        mul_23: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_22, primals_25)
        add_19: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_23, primals_26);  mul_23 = primals_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:176 in forward, code: outputs = self.fourier_transform(hidden_states).real
        convert_element_type_2: "c64[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_19, torch.complex64)
        _fft_c2c_2: "c64[32, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type_2, [1, 2], 0, True);  convert_element_type_2 = None
        view_as_real_2: "f32[32, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_2);  _fft_c2c_2 = None
        select_2: "f32[32, 512, 768]" = torch.ops.aten.select.int(view_as_real_2, 3, 0);  view_as_real_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        add_20: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_19, select_2);  add_19 = select_2 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_20, [2], correction = 0, keepdim = True)
        getitem_10: "f32[32, 512, 1]" = var_mean_5[0]
        getitem_11: "f32[32, 512, 1]" = var_mean_5[1];  var_mean_5 = None
        add_21: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_5: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_20, getitem_11);  add_20 = getitem_11 = None
        mul_24: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        mul_25: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_24, primals_27)
        add_22: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_25, primals_28);  mul_25 = primals_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_10: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_22, [16384, 768])
        permute_5: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_29, [1, 0])
        addmm_5: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_30, view_10, permute_5);  primals_30 = permute_5 = None
        view_11: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_5, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_26: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_11, 0.5)
        pow_3: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_11, 3.0)
        mul_27: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_23: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_11, mul_27);  view_11 = mul_27 = None
        mul_28: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_23, 0.7978845608028654);  add_23 = None
        tanh_2: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_28);  mul_28 = None
        add_24: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_29: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_26, add_24);  mul_26 = add_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        view_12: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_29, [16384, 3072]);  mul_29 = None
        permute_6: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_31, [1, 0])
        addmm_6: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_32, view_12, permute_6);  primals_32 = permute_6 = None
        view_13: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_6, [32, 512, 768]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:229 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_3: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_9: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        gt_3: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_9, 1e-30);  inductor_random_default_9 = None
        mul_30: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_3, view_13);  view_13 = None
        mul_31: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_30, 1.0);  mul_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_25: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_31, add_22);  mul_31 = add_22 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_12: "f32[32, 512, 1]" = var_mean_6[0]
        getitem_13: "f32[32, 512, 1]" = var_mean_6[1];  var_mean_6 = None
        add_26: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_6: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_25, getitem_13);  add_25 = getitem_13 = None
        mul_32: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        mul_33: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_32, primals_33)
        add_27: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_33, primals_34);  mul_33 = primals_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:176 in forward, code: outputs = self.fourier_transform(hidden_states).real
        convert_element_type_3: "c64[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_27, torch.complex64)
        _fft_c2c_3: "c64[32, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type_3, [1, 2], 0, True);  convert_element_type_3 = None
        view_as_real_3: "f32[32, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_3);  _fft_c2c_3 = None
        select_3: "f32[32, 512, 768]" = torch.ops.aten.select.int(view_as_real_3, 3, 0);  view_as_real_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        add_28: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_27, select_3);  add_27 = select_3 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
        getitem_14: "f32[32, 512, 1]" = var_mean_7[0]
        getitem_15: "f32[32, 512, 1]" = var_mean_7[1];  var_mean_7 = None
        add_29: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        sub_7: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_28, getitem_15);  add_28 = getitem_15 = None
        mul_34: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        mul_35: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_34, primals_35)
        add_30: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_35, primals_36);  mul_35 = primals_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_14: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_30, [16384, 768])
        permute_7: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_37, [1, 0])
        addmm_7: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_38, view_14, permute_7);  primals_38 = permute_7 = None
        view_15: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_7, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_36: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_15, 0.5)
        pow_4: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_15, 3.0)
        mul_37: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_31: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_15, mul_37);  view_15 = mul_37 = None
        mul_38: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_31, 0.7978845608028654);  add_31 = None
        tanh_3: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_32: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_39: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_36, add_32);  mul_36 = add_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        view_16: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_39, [16384, 3072]);  mul_39 = None
        permute_8: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_39, [1, 0])
        addmm_8: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_40, view_16, permute_8);  primals_40 = permute_8 = None
        view_17: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_8, [32, 512, 768]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:229 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_4: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_random_default_8: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        gt_4: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_8, 1e-30);  inductor_random_default_8 = None
        mul_40: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_4, view_17);  view_17 = None
        mul_41: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_40, 1.0);  mul_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_33: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_41, add_30);  mul_41 = add_30 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
        getitem_16: "f32[32, 512, 1]" = var_mean_8[0]
        getitem_17: "f32[32, 512, 1]" = var_mean_8[1];  var_mean_8 = None
        add_34: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        sub_8: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_33, getitem_17);  add_33 = getitem_17 = None
        mul_42: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        mul_43: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_42, primals_41)
        add_35: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_43, primals_42);  mul_43 = primals_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:176 in forward, code: outputs = self.fourier_transform(hidden_states).real
        convert_element_type_4: "c64[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_35, torch.complex64)
        _fft_c2c_4: "c64[32, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type_4, [1, 2], 0, True);  convert_element_type_4 = None
        view_as_real_4: "f32[32, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_4);  _fft_c2c_4 = None
        select_4: "f32[32, 512, 768]" = torch.ops.aten.select.int(view_as_real_4, 3, 0);  view_as_real_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        add_36: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_35, select_4);  add_35 = select_4 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_36, [2], correction = 0, keepdim = True)
        getitem_18: "f32[32, 512, 1]" = var_mean_9[0]
        getitem_19: "f32[32, 512, 1]" = var_mean_9[1];  var_mean_9 = None
        add_37: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        sub_9: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_36, getitem_19);  add_36 = getitem_19 = None
        mul_44: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        mul_45: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_44, primals_43)
        add_38: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_45, primals_44);  mul_45 = primals_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_18: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_38, [16384, 768])
        permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_45, [1, 0])
        addmm_9: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_46, view_18, permute_9);  primals_46 = permute_9 = None
        view_19: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_9, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_46: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.5)
        pow_5: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_19, 3.0)
        mul_47: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_39: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_19, mul_47);  view_19 = mul_47 = None
        mul_48: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_39, 0.7978845608028654);  add_39 = None
        tanh_4: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_48);  mul_48 = None
        add_40: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_49: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_46, add_40);  mul_46 = add_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        view_20: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_49, [16384, 3072]);  mul_49 = None
        permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_47, [1, 0])
        addmm_10: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_48, view_20, permute_10);  primals_48 = permute_10 = None
        view_21: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_10, [32, 512, 768]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:229 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_5: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_random_default_7: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_5, 'rand');  inductor_lookup_seed_default_5 = None
        gt_5: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_7, 1e-30);  inductor_random_default_7 = None
        mul_50: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_5, view_21);  view_21 = None
        mul_51: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_50, 1.0);  mul_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_41: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_51, add_38);  mul_51 = add_38 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
        getitem_20: "f32[32, 512, 1]" = var_mean_10[0]
        getitem_21: "f32[32, 512, 1]" = var_mean_10[1];  var_mean_10 = None
        add_42: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_10: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_41, getitem_21);  add_41 = getitem_21 = None
        mul_52: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        mul_53: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_52, primals_49)
        add_43: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_53, primals_50);  mul_53 = primals_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:176 in forward, code: outputs = self.fourier_transform(hidden_states).real
        convert_element_type_5: "c64[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_43, torch.complex64)
        _fft_c2c_5: "c64[32, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type_5, [1, 2], 0, True);  convert_element_type_5 = None
        view_as_real_5: "f32[32, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_5);  _fft_c2c_5 = None
        select_5: "f32[32, 512, 768]" = torch.ops.aten.select.int(view_as_real_5, 3, 0);  view_as_real_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        add_44: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_43, select_5);  add_43 = select_5 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_44, [2], correction = 0, keepdim = True)
        getitem_22: "f32[32, 512, 1]" = var_mean_11[0]
        getitem_23: "f32[32, 512, 1]" = var_mean_11[1];  var_mean_11 = None
        add_45: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        sub_11: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_44, getitem_23);  add_44 = getitem_23 = None
        mul_54: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        mul_55: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_54, primals_51)
        add_46: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_55, primals_52);  mul_55 = primals_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_22: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_46, [16384, 768])
        permute_11: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_53, [1, 0])
        addmm_11: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_54, view_22, permute_11);  primals_54 = permute_11 = None
        view_23: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_11, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_56: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_23, 0.5)
        pow_6: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_23, 3.0)
        mul_57: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_47: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_23, mul_57);  view_23 = mul_57 = None
        mul_58: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_47, 0.7978845608028654);  add_47 = None
        tanh_5: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_58);  mul_58 = None
        add_48: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_59: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_56, add_48);  mul_56 = add_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        view_24: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_59, [16384, 3072]);  mul_59 = None
        permute_12: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_55, [1, 0])
        addmm_12: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_56, view_24, permute_12);  primals_56 = permute_12 = None
        view_25: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_12, [32, 512, 768]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:229 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_6: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6)
        inductor_random_default_6: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_6, 'rand');  inductor_lookup_seed_default_6 = None
        gt_6: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_6, 1e-30);  inductor_random_default_6 = None
        mul_60: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_6, view_25);  view_25 = None
        mul_61: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_60, 1.0);  mul_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_49: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_61, add_46);  mul_61 = add_46 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
        getitem_24: "f32[32, 512, 1]" = var_mean_12[0]
        getitem_25: "f32[32, 512, 1]" = var_mean_12[1];  var_mean_12 = None
        add_50: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_12: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_49, getitem_25);  add_49 = getitem_25 = None
        mul_62: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        mul_63: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_62, primals_57)
        add_51: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_63, primals_58);  mul_63 = primals_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:176 in forward, code: outputs = self.fourier_transform(hidden_states).real
        convert_element_type_6: "c64[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_51, torch.complex64)
        _fft_c2c_6: "c64[32, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type_6, [1, 2], 0, True);  convert_element_type_6 = None
        view_as_real_6: "f32[32, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_6);  _fft_c2c_6 = None
        select_6: "f32[32, 512, 768]" = torch.ops.aten.select.int(view_as_real_6, 3, 0);  view_as_real_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        add_52: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_51, select_6);  add_51 = select_6 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
        getitem_26: "f32[32, 512, 1]" = var_mean_13[0]
        getitem_27: "f32[32, 512, 1]" = var_mean_13[1];  var_mean_13 = None
        add_53: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_13: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_52, getitem_27);  add_52 = getitem_27 = None
        mul_64: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        mul_65: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_64, primals_59)
        add_54: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_65, primals_60);  mul_65 = primals_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_26: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_54, [16384, 768])
        permute_13: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_61, [1, 0])
        addmm_13: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_62, view_26, permute_13);  primals_62 = permute_13 = None
        view_27: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_13, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_66: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_27, 0.5)
        pow_7: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_27, 3.0)
        mul_67: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_55: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_27, mul_67);  view_27 = mul_67 = None
        mul_68: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_55, 0.7978845608028654);  add_55 = None
        tanh_6: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_68);  mul_68 = None
        add_56: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_69: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_66, add_56);  mul_66 = add_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        view_28: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_69, [16384, 3072]);  mul_69 = None
        permute_14: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_63, [1, 0])
        addmm_14: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_64, view_28, permute_14);  primals_64 = permute_14 = None
        view_29: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_14, [32, 512, 768]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:229 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_7: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 7)
        inductor_random_default_5: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_7, 'rand');  inductor_lookup_seed_default_7 = None
        gt_7: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_5, 1e-30);  inductor_random_default_5 = None
        mul_70: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_7, view_29);  view_29 = None
        mul_71: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_70, 1.0);  mul_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_57: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_71, add_54);  mul_71 = add_54 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_28: "f32[32, 512, 1]" = var_mean_14[0]
        getitem_29: "f32[32, 512, 1]" = var_mean_14[1];  var_mean_14 = None
        add_58: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_14: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_57, getitem_29);  add_57 = getitem_29 = None
        mul_72: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        mul_73: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_72, primals_65)
        add_59: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_73, primals_66);  mul_73 = primals_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:176 in forward, code: outputs = self.fourier_transform(hidden_states).real
        convert_element_type_7: "c64[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_59, torch.complex64)
        _fft_c2c_7: "c64[32, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type_7, [1, 2], 0, True);  convert_element_type_7 = None
        view_as_real_7: "f32[32, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_7);  _fft_c2c_7 = None
        select_7: "f32[32, 512, 768]" = torch.ops.aten.select.int(view_as_real_7, 3, 0);  view_as_real_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        add_60: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_59, select_7);  add_59 = select_7 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_60, [2], correction = 0, keepdim = True)
        getitem_30: "f32[32, 512, 1]" = var_mean_15[0]
        getitem_31: "f32[32, 512, 1]" = var_mean_15[1];  var_mean_15 = None
        add_61: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        sub_15: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_60, getitem_31);  add_60 = getitem_31 = None
        mul_74: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        mul_75: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_74, primals_67)
        add_62: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_75, primals_68);  mul_75 = primals_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_30: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_62, [16384, 768])
        permute_15: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_69, [1, 0])
        addmm_15: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_70, view_30, permute_15);  primals_70 = permute_15 = None
        view_31: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_15, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_76: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_31, 0.5)
        pow_8: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_31, 3.0)
        mul_77: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_63: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_31, mul_77);  view_31 = mul_77 = None
        mul_78: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_63, 0.7978845608028654);  add_63 = None
        tanh_7: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_64: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_79: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_76, add_64);  mul_76 = add_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        view_32: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_79, [16384, 3072]);  mul_79 = None
        permute_16: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_71, [1, 0])
        addmm_16: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_72, view_32, permute_16);  primals_72 = permute_16 = None
        view_33: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_16, [32, 512, 768]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:229 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_8: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8)
        inductor_random_default_4: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_8, 'rand');  inductor_lookup_seed_default_8 = None
        gt_8: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_4, 1e-30);  inductor_random_default_4 = None
        mul_80: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_8, view_33);  view_33 = None
        mul_81: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_80, 1.0);  mul_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_65: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_81, add_62);  mul_81 = add_62 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
        getitem_32: "f32[32, 512, 1]" = var_mean_16[0]
        getitem_33: "f32[32, 512, 1]" = var_mean_16[1];  var_mean_16 = None
        add_66: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_16: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_65, getitem_33);  add_65 = getitem_33 = None
        mul_82: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        mul_83: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_82, primals_73)
        add_67: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_83, primals_74);  mul_83 = primals_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:176 in forward, code: outputs = self.fourier_transform(hidden_states).real
        convert_element_type_8: "c64[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_67, torch.complex64)
        _fft_c2c_8: "c64[32, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type_8, [1, 2], 0, True);  convert_element_type_8 = None
        view_as_real_8: "f32[32, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_8);  _fft_c2c_8 = None
        select_8: "f32[32, 512, 768]" = torch.ops.aten.select.int(view_as_real_8, 3, 0);  view_as_real_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        add_68: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_67, select_8);  add_67 = select_8 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_68, [2], correction = 0, keepdim = True)
        getitem_34: "f32[32, 512, 1]" = var_mean_17[0]
        getitem_35: "f32[32, 512, 1]" = var_mean_17[1];  var_mean_17 = None
        add_69: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_17: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_68, getitem_35);  add_68 = getitem_35 = None
        mul_84: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        mul_85: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_84, primals_75)
        add_70: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_85, primals_76);  mul_85 = primals_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_34: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_70, [16384, 768])
        permute_17: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_77, [1, 0])
        addmm_17: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_78, view_34, permute_17);  primals_78 = permute_17 = None
        view_35: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_17, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_86: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_35, 0.5)
        pow_9: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_35, 3.0)
        mul_87: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_71: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_35, mul_87);  view_35 = mul_87 = None
        mul_88: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_71, 0.7978845608028654);  add_71 = None
        tanh_8: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_88);  mul_88 = None
        add_72: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_89: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_86, add_72);  mul_86 = add_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        view_36: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_89, [16384, 3072]);  mul_89 = None
        permute_18: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_79, [1, 0])
        addmm_18: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_80, view_36, permute_18);  primals_80 = permute_18 = None
        view_37: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_18, [32, 512, 768]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:229 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_9: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 9)
        inductor_random_default_3: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_9, 'rand');  inductor_lookup_seed_default_9 = None
        gt_9: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_3, 1e-30);  inductor_random_default_3 = None
        mul_90: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_9, view_37);  view_37 = None
        mul_91: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_90, 1.0);  mul_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_73: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_91, add_70);  mul_91 = add_70 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
        getitem_36: "f32[32, 512, 1]" = var_mean_18[0]
        getitem_37: "f32[32, 512, 1]" = var_mean_18[1];  var_mean_18 = None
        add_74: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_18: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_73, getitem_37);  add_73 = getitem_37 = None
        mul_92: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        mul_93: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_92, primals_81)
        add_75: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_93, primals_82);  mul_93 = primals_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:176 in forward, code: outputs = self.fourier_transform(hidden_states).real
        convert_element_type_9: "c64[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_75, torch.complex64)
        _fft_c2c_9: "c64[32, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type_9, [1, 2], 0, True);  convert_element_type_9 = None
        view_as_real_9: "f32[32, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_9);  _fft_c2c_9 = None
        select_9: "f32[32, 512, 768]" = torch.ops.aten.select.int(view_as_real_9, 3, 0);  view_as_real_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        add_76: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_75, select_9);  add_75 = select_9 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_76, [2], correction = 0, keepdim = True)
        getitem_38: "f32[32, 512, 1]" = var_mean_19[0]
        getitem_39: "f32[32, 512, 1]" = var_mean_19[1];  var_mean_19 = None
        add_77: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        sub_19: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_76, getitem_39);  add_76 = getitem_39 = None
        mul_94: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        mul_95: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_94, primals_83)
        add_78: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_95, primals_84);  mul_95 = primals_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_38: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_78, [16384, 768])
        permute_19: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_85, [1, 0])
        addmm_19: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_86, view_38, permute_19);  primals_86 = permute_19 = None
        view_39: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_19, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_96: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_39, 0.5)
        pow_10: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_39, 3.0)
        mul_97: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_79: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_39, mul_97);  view_39 = mul_97 = None
        mul_98: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_79, 0.7978845608028654);  add_79 = None
        tanh_9: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_98);  mul_98 = None
        add_80: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_99: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_96, add_80);  mul_96 = add_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        view_40: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_99, [16384, 3072]);  mul_99 = None
        permute_20: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_87, [1, 0])
        addmm_20: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_88, view_40, permute_20);  primals_88 = permute_20 = None
        view_41: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_20, [32, 512, 768]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:229 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_10: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 10)
        inductor_random_default_2: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_10, 'rand');  inductor_lookup_seed_default_10 = None
        gt_10: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_2, 1e-30);  inductor_random_default_2 = None
        mul_100: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_10, view_41);  view_41 = None
        mul_101: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_100, 1.0);  mul_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_81: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_101, add_78);  mul_101 = add_78 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_40: "f32[32, 512, 1]" = var_mean_20[0]
        getitem_41: "f32[32, 512, 1]" = var_mean_20[1];  var_mean_20 = None
        add_82: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_20: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_81, getitem_41);  add_81 = getitem_41 = None
        mul_102: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
        mul_103: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_102, primals_89)
        add_83: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_103, primals_90);  mul_103 = primals_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:176 in forward, code: outputs = self.fourier_transform(hidden_states).real
        convert_element_type_10: "c64[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_83, torch.complex64)
        _fft_c2c_10: "c64[32, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type_10, [1, 2], 0, True);  convert_element_type_10 = None
        view_as_real_10: "f32[32, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_10);  _fft_c2c_10 = None
        select_10: "f32[32, 512, 768]" = torch.ops.aten.select.int(view_as_real_10, 3, 0);  view_as_real_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        add_84: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_83, select_10);  add_83 = select_10 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
        getitem_42: "f32[32, 512, 1]" = var_mean_21[0]
        getitem_43: "f32[32, 512, 1]" = var_mean_21[1];  var_mean_21 = None
        add_85: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        sub_21: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_84, getitem_43);  add_84 = getitem_43 = None
        mul_104: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        mul_105: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_104, primals_91)
        add_86: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_105, primals_92);  mul_105 = primals_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_42: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_86, [16384, 768])
        permute_21: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_93, [1, 0])
        addmm_21: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_94, view_42, permute_21);  primals_94 = permute_21 = None
        view_43: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_21, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_106: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
        pow_11: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_43, 3.0)
        mul_107: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_87: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_43, mul_107);  view_43 = mul_107 = None
        mul_108: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_87, 0.7978845608028654);  add_87 = None
        tanh_10: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_108);  mul_108 = None
        add_88: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_109: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_106, add_88);  mul_106 = add_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        view_44: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_109, [16384, 3072]);  mul_109 = None
        permute_22: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_95, [1, 0])
        addmm_22: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_96, view_44, permute_22);  primals_96 = permute_22 = None
        view_45: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_22, [32, 512, 768]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:229 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_11: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 11)
        inductor_random_default_1: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_11, 'rand');  inductor_lookup_seed_default_11 = None
        gt_11: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 1e-30);  inductor_random_default_1 = None
        mul_110: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_11, view_45);  view_45 = None
        mul_111: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_110, 1.0);  mul_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_89: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_111, add_86);  mul_111 = add_86 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
        getitem_44: "f32[32, 512, 1]" = var_mean_22[0]
        getitem_45: "f32[32, 512, 1]" = var_mean_22[1];  var_mean_22 = None
        add_90: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        sub_22: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_89, getitem_45);  add_89 = getitem_45 = None
        mul_112: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        mul_113: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_112, primals_97)
        add_91: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_113, primals_98);  mul_113 = primals_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:176 in forward, code: outputs = self.fourier_transform(hidden_states).real
        convert_element_type_11: "c64[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_91, torch.complex64)
        _fft_c2c_11: "c64[32, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type_11, [1, 2], 0, True);  convert_element_type_11 = None
        view_as_real_11: "f32[32, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_11);  _fft_c2c_11 = None
        select_11: "f32[32, 512, 768]" = torch.ops.aten.select.int(view_as_real_11, 3, 0);  view_as_real_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        add_92: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_91, select_11);  add_91 = select_11 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
        getitem_46: "f32[32, 512, 1]" = var_mean_23[0]
        getitem_47: "f32[32, 512, 1]" = var_mean_23[1];  var_mean_23 = None
        add_93: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        sub_23: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_92, getitem_47);  add_92 = getitem_47 = None
        mul_114: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
        mul_115: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_114, primals_99)
        add_94: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_115, primals_100);  mul_115 = primals_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:214 in forward, code: hidden_states = self.dense(hidden_states)
        view_46: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_94, [16384, 768])
        permute_23: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_101, [1, 0])
        addmm_23: "f32[16384, 3072]" = torch.ops.aten.addmm.default(primals_102, view_46, permute_23);  primals_102 = permute_23 = None
        view_47: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_23, [32, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_116: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_47, 0.5)
        pow_12: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_47, 3.0)
        mul_117: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_95: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_47, mul_117);  view_47 = mul_117 = None
        mul_118: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_95, 0.7978845608028654);  add_95 = None
        tanh_11: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
        add_96: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_119: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_116, add_96);  mul_116 = add_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:228 in forward, code: hidden_states = self.dense(hidden_states)
        view_48: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_119, [16384, 3072]);  mul_119 = None
        permute_24: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_103, [1, 0])
        addmm_24: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_104, view_48, permute_24);  primals_104 = permute_24 = None
        view_49: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_24, [32, 512, 768]);  addmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:229 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_12: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 12);  inductor_seeds_default = None
        inductor_random_default: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default_12, 'rand');  inductor_lookup_seed_default_12 = None
        gt_12: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default, 1e-30);  inductor_random_default = None
        mul_120: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_12, view_49);  view_49 = None
        mul_121: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_120, 1.0);  mul_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_97: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_121, add_94);  mul_121 = add_94 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
        getitem_48: "f32[32, 512, 1]" = var_mean_24[0]
        getitem_49: "f32[32, 512, 1]" = var_mean_24[1];  var_mean_24 = None
        add_98: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        sub_24: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_97, getitem_49);  add_97 = getitem_49 = None
        mul_122: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        mul_123: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_122, primals_105)
        add_99: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_123, primals_106);  mul_123 = primals_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:316 in forward, code: hidden_states = self.dense(hidden_states)
        view_50: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_99, [16384, 768]);  add_99 = None
        permute_26: "f32[768, 768]" = torch.ops.aten.permute.default(primals_109, [1, 0])
        addmm_26: "f32[16384, 768]" = torch.ops.aten.addmm.default(primals_110, view_50, permute_26);  primals_110 = permute_26 = None
        view_51: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_26, [32, 512, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_124: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_51, 0.5)
        pow_13: "f32[32, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(view_51, 3.0)
        mul_125: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
        add_100: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_51, mul_125);  view_51 = mul_125 = None
        mul_126: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_100, 0.7978845608028654);  add_100 = None
        tanh_13: "f32[32, 512, 768]" = torch.ops.aten.tanh.default(mul_126);  mul_126 = None
        add_101: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(tanh_13, 1.0);  tanh_13 = None
        mul_127: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_124, add_101);  mul_124 = add_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:318 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        var_mean_25 = torch.ops.aten.var_mean.correction(mul_127, [2], correction = 0, keepdim = True)
        getitem_50: "f32[32, 512, 1]" = var_mean_25[0]
        getitem_51: "f32[32, 512, 1]" = var_mean_25[1];  var_mean_25 = None
        add_102: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_25: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        sub_25: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_127, getitem_51);  mul_127 = None
        mul_128: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        mul_129: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_128, primals_111);  mul_128 = None
        add_103: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_129, primals_112);  mul_129 = primals_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:331 in forward, code: hidden_states = self.decoder(hidden_states)
        view_52: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_103, [16384, 768]);  add_103 = None
        permute_27: "f32[768, 32000]" = torch.ops.aten.permute.default(primals_4, [1, 0])
        addmm_27: "f32[16384, 32000]" = torch.ops.aten.addmm.default(primals_113, view_52, permute_27);  primals_113 = permute_27 = None
        view_53: "f32[32, 512, 32000]" = torch.ops.aten.reshape.default(addmm_27, [32, 512, 32000]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:666 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        view_54: "f32[16384, 32000]" = torch.ops.aten.reshape.default(view_53, [-1, 32000])
        view_55: "i64[16384]" = torch.ops.aten.reshape.default(primals_114, [-1])
        amax: "f32[16384, 1]" = torch.ops.aten.amax.default(view_54, [1], True)
        sub_26: "f32[16384, 32000]" = torch.ops.aten.sub.Tensor(view_54, amax);  view_54 = None
        exp: "f32[16384, 32000]" = torch.ops.aten.exp.default(sub_26)
        sum_1: "f32[16384, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[16384, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_27: "f32[16384, 32000]" = torch.ops.aten.sub.Tensor(sub_26, log);  sub_26 = None
        ne: "b8[16384]" = torch.ops.aten.ne.Scalar(view_55, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[16384]" = torch.ops.aten.where.self(ne, view_55, full_default);  view_55 = full_default = None
        unsqueeze: "i64[16384, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[16384, 1]" = torch.ops.aten.gather.default(sub_27, 1, unsqueeze);  sub_27 = unsqueeze = None
        squeeze: "f32[16384]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[16384]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[16384]" = torch.ops.aten.where.self(ne, neg, full_default_1);  neg = full_default_1 = None
        sum_2: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type_12: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
        div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_12);  sum_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_3: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        div_4: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_5: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        div_6: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_7: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        div_8: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_9: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        div_10: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_11: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        div_12: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_13: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        div_14: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_15: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        div_16: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_17: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        div_18: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_19: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        div_20: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_21: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        div_22: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_23: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        div_24: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:230 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_25: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:186 in forward, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
        div_26: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:136 in forward, code: embeddings = self.LayerNorm(embeddings)
        div_27: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        return (div, view_53, primals_1, primals_2, primals_3, primals_4, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_109, primals_111, primals_114, mul, view, gt, mul_4, view_2, addmm_1, view_4, gt_1, mul_12, mul_14, view_6, addmm_3, view_8, gt_2, mul_22, mul_24, view_10, addmm_5, view_12, gt_3, mul_32, mul_34, view_14, addmm_7, view_16, gt_4, mul_42, mul_44, view_18, addmm_9, view_20, gt_5, mul_52, mul_54, view_22, addmm_11, view_24, gt_6, mul_62, mul_64, view_26, addmm_13, view_28, gt_7, mul_72, mul_74, view_30, addmm_15, view_32, gt_8, mul_82, mul_84, view_34, addmm_17, view_36, gt_9, mul_92, mul_94, view_38, addmm_19, view_40, gt_10, mul_102, mul_104, view_42, addmm_21, view_44, gt_11, mul_112, mul_114, view_46, addmm_23, view_48, gt_12, mul_122, view_50, addmm_26, getitem_51, rsqrt_25, view_52, view_53, amax, log, convert_element_type_12, div_3, div_4, div_5, div_6, div_7, div_8, div_9, div_10, div_11, div_12, div_13, div_14, div_15, div_16, div_17, div_18, div_19, div_20, div_21, div_22, div_23, div_24, div_25, div_26, div_27)
