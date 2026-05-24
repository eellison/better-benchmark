import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[64, 512]", arg1_1: "i64[64, 512]", arg2_1: "i64[1, 512]", arg3_1: "i64[1, 512]", arg4_1: "f32[30522, 128]", arg5_1: "f32[2, 128]", arg6_1: "f32[512, 128]", arg7_1: "f32[128]", arg8_1: "f32[128]", arg9_1: "f32[256, 128]", arg10_1: "f32[256]", arg11_1: "f32[256, 256]", arg12_1: "f32[256]", arg13_1: "f32[256, 256]", arg14_1: "f32[256]", arg15_1: "f32[256, 256]", arg16_1: "f32[256]", arg17_1: "f32[256, 256]", arg18_1: "f32[256]", arg19_1: "f32[256]", arg20_1: "f32[256]", arg21_1: "f32[1024, 256]", arg22_1: "f32[1024]", arg23_1: "f32[256, 1024]", arg24_1: "f32[256]", arg25_1: "f32[256]", arg26_1: "f32[256]", arg27_1: "f32[256, 256]", arg28_1: "f32[256]", arg29_1: "f32[256, 256]", arg30_1: "f32[256]", arg31_1: "f32[256, 256]", arg32_1: "f32[256]", arg33_1: "f32[256, 256]", arg34_1: "f32[256]", arg35_1: "f32[256]", arg36_1: "f32[256]", arg37_1: "f32[1024, 256]", arg38_1: "f32[1024]", arg39_1: "f32[256, 1024]", arg40_1: "f32[256]", arg41_1: "f32[256]", arg42_1: "f32[256]", arg43_1: "f32[256, 256]", arg44_1: "f32[256]", arg45_1: "f32[256, 256]", arg46_1: "f32[256]", arg47_1: "f32[256, 256]", arg48_1: "f32[256]", arg49_1: "f32[256, 256]", arg50_1: "f32[256]", arg51_1: "f32[256]", arg52_1: "f32[256]", arg53_1: "f32[1024, 256]", arg54_1: "f32[1024]", arg55_1: "f32[256, 1024]", arg56_1: "f32[256]", arg57_1: "f32[256]", arg58_1: "f32[256]", arg59_1: "f32[256, 256]", arg60_1: "f32[256]", arg61_1: "f32[256, 256]", arg62_1: "f32[256]", arg63_1: "f32[256, 256]", arg64_1: "f32[256]", arg65_1: "f32[256, 256]", arg66_1: "f32[256]", arg67_1: "f32[256]", arg68_1: "f32[256]", arg69_1: "f32[1024, 256]", arg70_1: "f32[1024]", arg71_1: "f32[256, 1024]", arg72_1: "f32[256]", arg73_1: "f32[256]", arg74_1: "f32[256]", arg75_1: "f32[256, 256]", arg76_1: "f32[256]", arg77_1: "f32[256, 256]", arg78_1: "f32[256]", arg79_1: "f32[256, 256]", arg80_1: "f32[256]", arg81_1: "f32[256, 256]", arg82_1: "f32[256]", arg83_1: "f32[256]", arg84_1: "f32[256]", arg85_1: "f32[1024, 256]", arg86_1: "f32[1024]", arg87_1: "f32[256, 1024]", arg88_1: "f32[256]", arg89_1: "f32[256]", arg90_1: "f32[256]", arg91_1: "f32[256, 256]", arg92_1: "f32[256]", arg93_1: "f32[256, 256]", arg94_1: "f32[256]", arg95_1: "f32[256, 256]", arg96_1: "f32[256]", arg97_1: "f32[256, 256]", arg98_1: "f32[256]", arg99_1: "f32[256]", arg100_1: "f32[256]", arg101_1: "f32[1024, 256]", arg102_1: "f32[1024]", arg103_1: "f32[256, 1024]", arg104_1: "f32[256]", arg105_1: "f32[256]", arg106_1: "f32[256]", arg107_1: "f32[256, 256]", arg108_1: "f32[256]", arg109_1: "f32[256, 256]", arg110_1: "f32[256]", arg111_1: "f32[256, 256]", arg112_1: "f32[256]", arg113_1: "f32[256, 256]", arg114_1: "f32[256]", arg115_1: "f32[256]", arg116_1: "f32[256]", arg117_1: "f32[1024, 256]", arg118_1: "f32[1024]", arg119_1: "f32[256, 1024]", arg120_1: "f32[256]", arg121_1: "f32[256]", arg122_1: "f32[256]", arg123_1: "f32[256, 256]", arg124_1: "f32[256]", arg125_1: "f32[256, 256]", arg126_1: "f32[256]", arg127_1: "f32[256, 256]", arg128_1: "f32[256]", arg129_1: "f32[256, 256]", arg130_1: "f32[256]", arg131_1: "f32[256]", arg132_1: "f32[256]", arg133_1: "f32[1024, 256]", arg134_1: "f32[1024]", arg135_1: "f32[256, 1024]", arg136_1: "f32[256]", arg137_1: "f32[256]", arg138_1: "f32[256]", arg139_1: "f32[256, 256]", arg140_1: "f32[256]", arg141_1: "f32[256, 256]", arg142_1: "f32[256]", arg143_1: "f32[256, 256]", arg144_1: "f32[256]", arg145_1: "f32[256, 256]", arg146_1: "f32[256]", arg147_1: "f32[256]", arg148_1: "f32[256]", arg149_1: "f32[1024, 256]", arg150_1: "f32[1024]", arg151_1: "f32[256, 1024]", arg152_1: "f32[256]", arg153_1: "f32[256]", arg154_1: "f32[256]", arg155_1: "f32[256, 256]", arg156_1: "f32[256]", arg157_1: "f32[256, 256]", arg158_1: "f32[256]", arg159_1: "f32[256, 256]", arg160_1: "f32[256]", arg161_1: "f32[256, 256]", arg162_1: "f32[256]", arg163_1: "f32[256]", arg164_1: "f32[256]", arg165_1: "f32[1024, 256]", arg166_1: "f32[1024]", arg167_1: "f32[256, 1024]", arg168_1: "f32[256]", arg169_1: "f32[256]", arg170_1: "f32[256]", arg171_1: "f32[256, 256]", arg172_1: "f32[256]", arg173_1: "f32[256, 256]", arg174_1: "f32[256]", arg175_1: "f32[256, 256]", arg176_1: "f32[256]", arg177_1: "f32[256, 256]", arg178_1: "f32[256]", arg179_1: "f32[256]", arg180_1: "f32[256]", arg181_1: "f32[1024, 256]", arg182_1: "f32[1024]", arg183_1: "f32[256, 1024]", arg184_1: "f32[256]", arg185_1: "f32[256]", arg186_1: "f32[256]", arg187_1: "f32[256, 256]", arg188_1: "f32[256]", arg189_1: "f32[256, 256]", arg190_1: "f32[256]", arg191_1: "f32[256, 256]", arg192_1: "f32[256]", arg193_1: "f32[256, 256]", arg194_1: "f32[256]", arg195_1: "f32[256]", arg196_1: "f32[256]", arg197_1: "f32[1024, 256]", arg198_1: "f32[1024]", arg199_1: "f32[256, 1024]", arg200_1: "f32[256]", arg201_1: "f32[256]", arg202_1: "f32[256]", arg203_1: "f32[128, 256]", arg204_1: "f32[128]", arg205_1: "f32[128]", arg206_1: "f32[128]", arg207_1: "f32[30522]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:108 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding: "f32[64, 512, 128]" = torch.ops.aten.embedding.default(arg4_1, arg1_1, 0);  arg1_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:101 in forward, code: buffered_token_type_ids = self.token_type_ids.expand(position_ids.shape[0], -1)
        expand: "i64[1, 512]" = torch.ops.aten.expand.default(arg3_1, [1, -1]);  arg3_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:102 in forward, code: buffered_token_type_ids = torch.gather(buffered_token_type_ids, dim=1, index=position_ids)
        gather: "i64[1, 512]" = torch.ops.aten.gather.default(expand, 1, arg2_1);  expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:103 in forward, code: token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
        expand_1: "i64[64, 512]" = torch.ops.aten.expand.default(gather, [64, 512]);  gather = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:109 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_1: "f32[64, 512, 128]" = torch.ops.aten.embedding.default(arg5_1, expand_1);  arg5_1 = expand_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:110 in forward, code: embeddings = inputs_embeds + token_type_embeddings
        add: "f32[64, 512, 128]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:112 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_2: "f32[1, 512, 128]" = torch.ops.aten.embedding.default(arg6_1, arg2_1);  arg6_1 = arg2_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:113 in forward, code: embeddings = embeddings + position_embeddings
        add_1: "f32[64, 512, 128]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:115 in forward, code: embeddings = self.LayerNorm(embeddings)
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem: "f32[64, 512, 1]" = var_mean[0]
        getitem_1: "f32[64, 512, 1]" = var_mean[1];  var_mean = None
        sub: "f32[64, 512, 128]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        add_2: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul, arg7_1);  mul = arg7_1 = None
        add_3: "f32[64, 512, 128]" = torch.ops.aten.add.Tensor(mul_1, arg8_1);  mul_1 = arg8_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:610 in forward, code: embedding_output = self.embeddings_project(embedding_output)
        view: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_3, [32768, 128]);  add_3 = None
        permute: "f32[128, 256]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg10_1, view, permute);  arg10_1 = view = permute = None
        view_1: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm, [64, 512, 256]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_2: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_1, [32768, 256])
        permute_1: "f32[256, 256]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_1: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg12_1, view_2, permute_1);  arg12_1 = view_2 = permute_1 = None
        view_3: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_1, [64, 512, 256]);  addmm_1 = None
        view_4: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_3, [64, 512, -1, 64]);  view_3 = None
        permute_2: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_2: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(permute_2, 0.3535533905932738);  permute_2 = None
        expand_3: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(mul_2, [64, 4, 512, 64]);  mul_2 = None
        clone_1: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_11: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_1, [256, 512, 64]);  clone_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_5: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_1, [32768, 256])
        permute_3: "f32[256, 256]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_2: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg14_1, view_5, permute_3);  arg14_1 = view_5 = permute_3 = None
        view_6: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_2, [64, 512, 256]);  addmm_2 = None
        view_7: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_6, [64, 512, -1, 64]);  view_6 = None
        permute_4: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_7: "f32[64, 4, 64, 512]" = torch.ops.aten.permute.default(permute_4, [0, 1, 3, 2]);  permute_4 = None
        mul_3: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_7, 0.3535533905932738);  permute_7 = None
        expand_4: "f32[64, 4, 64, 512]" = torch.ops.aten.expand.default(mul_3, [64, 4, 64, 512]);  mul_3 = None
        clone_2: "f32[64, 4, 64, 512]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_12: "f32[256, 64, 512]" = torch.ops.aten.reshape.default(clone_2, [256, 64, 512]);  clone_2 = None
        bmm: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_11, view_12);  view_11 = view_12 = None
        view_13: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm, [64, 4, 512, 512]);  bmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_2: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_4: "i64[512]" = torch.ops.aten.add.Tensor(iota_2, 0);  iota_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_4, 0);  add_4 = None
        unsqueeze_1: "i64[1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:87 in bidirectional_mask_function, code: return q_idx >= 0
        ge: "b8[1, 1, 512, 1]" = torch.ops.aten.ge.Scalar(unsqueeze_2, 0);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_2: "b8[64, 1, 512, 512]" = torch.ops.aten.expand.default(ge, [64, -1, 512, 512]);  ge = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[64, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_1, full_default);  full_default_1 = full_default = None
        add_6: "f32[64, 4, 512, 512]" = torch.ops.aten.add.Tensor(view_13, where);  view_13 = where = None
        eq: "b8[64, 4, 512, 512]" = torch.ops.aten.eq.Scalar(add_6, -inf)
        logical_not: "b8[64, 4, 512, 512]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[64, 4, 512, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[64, 4, 512, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full_default_2: "f32[64, 4, 512, 512]" = torch.ops.aten.full.default([64, 4, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax: "f32[64, 4, 512, 1]" = torch.ops.aten.amax.default(add_6, [-1], True)
        sub_1: "f32[64, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_6, amax);  add_6 = amax = None
        exp: "f32[64, 4, 512, 512]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[64, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        where_1: "f32[64, 4, 512, 512]" = torch.ops.aten.where.self(logical_not_1, full_default_2, div);  logical_not_1 = full_default_2 = div = None
        expand_5: "f32[64, 4, 512, 512]" = torch.ops.aten.expand.default(where_1, [64, 4, 512, 512]);  where_1 = None
        view_14: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(expand_5, [256, 512, 512]);  expand_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_8: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_1, [32768, 256])
        permute_5: "f32[256, 256]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_3: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg16_1, view_8, permute_5);  arg16_1 = view_8 = permute_5 = None
        view_9: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_3, [64, 512, 256]);  addmm_3 = None
        view_10: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_9, [64, 512, -1, 64]);  view_9 = None
        permute_6: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_6: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(permute_6, [64, 4, 512, 64]);  permute_6 = None
        clone_3: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_15: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_3, [256, 512, 64]);  clone_3 = None
        bmm_1: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_14, view_15);  view_14 = view_15 = None
        view_16: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_1, [64, 4, 512, 64]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_8: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        clone_4: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_17: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_4, [64, 512, -1]);  clone_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_18: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_17, [32768, 256]);  view_17 = None
        permute_9: "f32[256, 256]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        addmm_4: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg18_1, view_18, permute_9);  arg18_1 = view_18 = permute_9 = None
        view_19: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_4, [64, 512, 256]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_7: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(view_19, view_1);  view_19 = view_1 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
        getitem_2: "f32[64, 512, 1]" = var_mean_1[0]
        getitem_3: "f32[64, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_2: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_7, getitem_3);  add_7 = getitem_3 = None
        add_8: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        mul_4: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_5: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_4, arg19_1);  mul_4 = arg19_1 = None
        add_9: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_5, arg20_1);  mul_5 = arg20_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_20: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_9, [32768, 256])
        permute_10: "f32[256, 1024]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_5: "f32[32768, 1024]" = torch.ops.aten.addmm.default(arg22_1, view_20, permute_10);  arg22_1 = view_20 = permute_10 = None
        view_21: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_5, [64, 512, 1024]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_6: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_21, 0.5)
        mul_7: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_21, 0.7071067811865476);  view_21 = None
        erf: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
        add_10: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_8: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_6, add_10);  mul_6 = add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_22: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_8, [32768, 1024]);  mul_8 = None
        permute_11: "f32[1024, 256]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_6: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg24_1, view_22, permute_11);  arg24_1 = view_22 = permute_11 = None
        view_23: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_6, [64, 512, 256]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_11: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(view_23, add_9);  view_23 = add_9 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_4: "f32[64, 512, 1]" = var_mean_2[0]
        getitem_5: "f32[64, 512, 1]" = var_mean_2[1];  var_mean_2 = None
        sub_3: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_11, getitem_5);  add_11 = getitem_5 = None
        add_12: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_9: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_10: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_9, arg25_1);  mul_9 = arg25_1 = None
        add_13: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_10, arg26_1);  mul_10 = arg26_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_24: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_13, [32768, 256])
        permute_12: "f32[256, 256]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_7: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg28_1, view_24, permute_12);  arg28_1 = view_24 = permute_12 = None
        view_25: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_7, [64, 512, 256]);  addmm_7 = None
        view_26: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_25, [64, 512, -1, 64]);  view_25 = None
        permute_13: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_11: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(permute_13, 0.3535533905932738);  permute_13 = None
        expand_7: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(mul_11, [64, 4, 512, 64]);  mul_11 = None
        clone_7: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_33: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_7, [256, 512, 64]);  clone_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_27: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_13, [32768, 256])
        permute_14: "f32[256, 256]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        addmm_8: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg30_1, view_27, permute_14);  arg30_1 = view_27 = permute_14 = None
        view_28: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_8, [64, 512, 256]);  addmm_8 = None
        view_29: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_28, [64, 512, -1, 64]);  view_28 = None
        permute_15: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_18: "f32[64, 4, 64, 512]" = torch.ops.aten.permute.default(permute_15, [0, 1, 3, 2]);  permute_15 = None
        mul_12: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_18, 0.3535533905932738);  permute_18 = None
        expand_8: "f32[64, 4, 64, 512]" = torch.ops.aten.expand.default(mul_12, [64, 4, 64, 512]);  mul_12 = None
        clone_8: "f32[64, 4, 64, 512]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_34: "f32[256, 64, 512]" = torch.ops.aten.reshape.default(clone_8, [256, 64, 512]);  clone_8 = None
        bmm_2: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_33, view_34);  view_33 = view_34 = None
        view_35: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_2, [64, 4, 512, 512]);  bmm_2 = None
        full_default_4: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[64, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_4, full_default_3);  full_default_4 = full_default_3 = None
        add_14: "f32[64, 4, 512, 512]" = torch.ops.aten.add.Tensor(view_35, where_2);  view_35 = where_2 = None
        eq_1: "b8[64, 4, 512, 512]" = torch.ops.aten.eq.Scalar(add_14, -inf)
        logical_not_2: "b8[64, 4, 512, 512]" = torch.ops.aten.logical_not.default(eq_1);  eq_1 = None
        any_2: "b8[64, 4, 512, 1]" = torch.ops.aten.any.dim(logical_not_2, -1, True);  logical_not_2 = None
        logical_not_3: "b8[64, 4, 512, 1]" = torch.ops.aten.logical_not.default(any_2);  any_2 = None
        full_default_5: "f32[64, 4, 512, 512]" = torch.ops.aten.full.default([64, 4, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_1: "f32[64, 4, 512, 1]" = torch.ops.aten.amax.default(add_14, [-1], True)
        sub_4: "f32[64, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_14, amax_1);  add_14 = amax_1 = None
        exp_1: "f32[64, 4, 512, 512]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[64, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        where_3: "f32[64, 4, 512, 512]" = torch.ops.aten.where.self(logical_not_3, full_default_5, div_1);  logical_not_3 = full_default_5 = div_1 = None
        expand_9: "f32[64, 4, 512, 512]" = torch.ops.aten.expand.default(where_3, [64, 4, 512, 512]);  where_3 = None
        view_36: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(expand_9, [256, 512, 512]);  expand_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_30: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_13, [32768, 256])
        permute_16: "f32[256, 256]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_9: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg32_1, view_30, permute_16);  arg32_1 = view_30 = permute_16 = None
        view_31: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_9, [64, 512, 256]);  addmm_9 = None
        view_32: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_31, [64, 512, -1, 64]);  view_31 = None
        permute_17: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_10: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(permute_17, [64, 4, 512, 64]);  permute_17 = None
        clone_9: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_37: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_9, [256, 512, 64]);  clone_9 = None
        bmm_3: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_36, view_37);  view_36 = view_37 = None
        view_38: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_3, [64, 4, 512, 64]);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_19: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
        clone_10: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_39: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_10, [64, 512, -1]);  clone_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_40: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_39, [32768, 256]);  view_39 = None
        permute_20: "f32[256, 256]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        addmm_10: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg34_1, view_40, permute_20);  arg34_1 = view_40 = permute_20 = None
        view_41: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_10, [64, 512, 256]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_15: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(view_41, add_13);  view_41 = add_13 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
        getitem_6: "f32[64, 512, 1]" = var_mean_3[0]
        getitem_7: "f32[64, 512, 1]" = var_mean_3[1];  var_mean_3 = None
        sub_5: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_15, getitem_7);  add_15 = getitem_7 = None
        add_16: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        mul_13: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_14: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_13, arg35_1);  mul_13 = arg35_1 = None
        add_17: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_14, arg36_1);  mul_14 = arg36_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_42: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_17, [32768, 256])
        permute_21: "f32[256, 1024]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_11: "f32[32768, 1024]" = torch.ops.aten.addmm.default(arg38_1, view_42, permute_21);  arg38_1 = view_42 = permute_21 = None
        view_43: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_11, [64, 512, 1024]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_15: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
        mul_16: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_43, 0.7071067811865476);  view_43 = None
        erf_1: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_16);  mul_16 = None
        add_18: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_17: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_15, add_18);  mul_15 = add_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_44: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_17, [32768, 1024]);  mul_17 = None
        permute_22: "f32[1024, 256]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_12: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg40_1, view_44, permute_22);  arg40_1 = view_44 = permute_22 = None
        view_45: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_12, [64, 512, 256]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_19: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(view_45, add_17);  view_45 = add_17 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
        getitem_8: "f32[64, 512, 1]" = var_mean_4[0]
        getitem_9: "f32[64, 512, 1]" = var_mean_4[1];  var_mean_4 = None
        sub_6: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_19, getitem_9);  add_19 = getitem_9 = None
        add_20: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_18: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_19: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_18, arg41_1);  mul_18 = arg41_1 = None
        add_21: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_19, arg42_1);  mul_19 = arg42_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_46: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_21, [32768, 256])
        permute_23: "f32[256, 256]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_13: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg44_1, view_46, permute_23);  arg44_1 = view_46 = permute_23 = None
        view_47: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_13, [64, 512, 256]);  addmm_13 = None
        view_48: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_47, [64, 512, -1, 64]);  view_47 = None
        permute_24: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_20: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(permute_24, 0.3535533905932738);  permute_24 = None
        expand_11: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(mul_20, [64, 4, 512, 64]);  mul_20 = None
        clone_13: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_55: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_13, [256, 512, 64]);  clone_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_49: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_21, [32768, 256])
        permute_25: "f32[256, 256]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        addmm_14: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg46_1, view_49, permute_25);  arg46_1 = view_49 = permute_25 = None
        view_50: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_14, [64, 512, 256]);  addmm_14 = None
        view_51: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_50, [64, 512, -1, 64]);  view_50 = None
        permute_26: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_29: "f32[64, 4, 64, 512]" = torch.ops.aten.permute.default(permute_26, [0, 1, 3, 2]);  permute_26 = None
        mul_21: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_29, 0.3535533905932738);  permute_29 = None
        expand_12: "f32[64, 4, 64, 512]" = torch.ops.aten.expand.default(mul_21, [64, 4, 64, 512]);  mul_21 = None
        clone_14: "f32[64, 4, 64, 512]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_56: "f32[256, 64, 512]" = torch.ops.aten.reshape.default(clone_14, [256, 64, 512]);  clone_14 = None
        bmm_4: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_55, view_56);  view_55 = view_56 = None
        view_57: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_4, [64, 4, 512, 512]);  bmm_4 = None
        full_default_7: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "f32[64, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_7, full_default_6);  full_default_7 = full_default_6 = None
        add_22: "f32[64, 4, 512, 512]" = torch.ops.aten.add.Tensor(view_57, where_4);  view_57 = where_4 = None
        eq_2: "b8[64, 4, 512, 512]" = torch.ops.aten.eq.Scalar(add_22, -inf)
        logical_not_4: "b8[64, 4, 512, 512]" = torch.ops.aten.logical_not.default(eq_2);  eq_2 = None
        any_3: "b8[64, 4, 512, 1]" = torch.ops.aten.any.dim(logical_not_4, -1, True);  logical_not_4 = None
        logical_not_5: "b8[64, 4, 512, 1]" = torch.ops.aten.logical_not.default(any_3);  any_3 = None
        full_default_8: "f32[64, 4, 512, 512]" = torch.ops.aten.full.default([64, 4, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_2: "f32[64, 4, 512, 1]" = torch.ops.aten.amax.default(add_22, [-1], True)
        sub_7: "f32[64, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_22, amax_2);  add_22 = amax_2 = None
        exp_2: "f32[64, 4, 512, 512]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[64, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        where_5: "f32[64, 4, 512, 512]" = torch.ops.aten.where.self(logical_not_5, full_default_8, div_2);  logical_not_5 = full_default_8 = div_2 = None
        expand_13: "f32[64, 4, 512, 512]" = torch.ops.aten.expand.default(where_5, [64, 4, 512, 512]);  where_5 = None
        view_58: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(expand_13, [256, 512, 512]);  expand_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_52: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_21, [32768, 256])
        permute_27: "f32[256, 256]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_15: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg48_1, view_52, permute_27);  arg48_1 = view_52 = permute_27 = None
        view_53: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_15, [64, 512, 256]);  addmm_15 = None
        view_54: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_53, [64, 512, -1, 64]);  view_53 = None
        permute_28: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_14: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(permute_28, [64, 4, 512, 64]);  permute_28 = None
        clone_15: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_59: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_15, [256, 512, 64]);  clone_15 = None
        bmm_5: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_58, view_59);  view_58 = view_59 = None
        view_60: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_5, [64, 4, 512, 64]);  bmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_30: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
        clone_16: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_61: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_16, [64, 512, -1]);  clone_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_62: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_61, [32768, 256]);  view_61 = None
        permute_31: "f32[256, 256]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_16: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg50_1, view_62, permute_31);  arg50_1 = view_62 = permute_31 = None
        view_63: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_16, [64, 512, 256]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_23: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(view_63, add_21);  view_63 = add_21 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_23, [2], correction = 0, keepdim = True)
        getitem_10: "f32[64, 512, 1]" = var_mean_5[0]
        getitem_11: "f32[64, 512, 1]" = var_mean_5[1];  var_mean_5 = None
        sub_8: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_23, getitem_11);  add_23 = getitem_11 = None
        add_24: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        mul_22: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
        mul_23: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_22, arg51_1);  mul_22 = arg51_1 = None
        add_25: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_23, arg52_1);  mul_23 = arg52_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_64: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_25, [32768, 256])
        permute_32: "f32[256, 1024]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_17: "f32[32768, 1024]" = torch.ops.aten.addmm.default(arg54_1, view_64, permute_32);  arg54_1 = view_64 = permute_32 = None
        view_65: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_17, [64, 512, 1024]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_24: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_65, 0.5)
        mul_25: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_65, 0.7071067811865476);  view_65 = None
        erf_2: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_25);  mul_25 = None
        add_26: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_26: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_24, add_26);  mul_24 = add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_66: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_26, [32768, 1024]);  mul_26 = None
        permute_33: "f32[1024, 256]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_18: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg56_1, view_66, permute_33);  arg56_1 = view_66 = permute_33 = None
        view_67: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_18, [64, 512, 256]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_27: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(view_67, add_25);  view_67 = add_25 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_12: "f32[64, 512, 1]" = var_mean_6[0]
        getitem_13: "f32[64, 512, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_9: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_27, getitem_13);  add_27 = getitem_13 = None
        add_28: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_27: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
        mul_28: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_27, arg57_1);  mul_27 = arg57_1 = None
        add_29: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_28, arg58_1);  mul_28 = arg58_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_68: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_29, [32768, 256])
        permute_34: "f32[256, 256]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_19: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg60_1, view_68, permute_34);  arg60_1 = view_68 = permute_34 = None
        view_69: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_19, [64, 512, 256]);  addmm_19 = None
        view_70: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_69, [64, 512, -1, 64]);  view_69 = None
        permute_35: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_29: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(permute_35, 0.3535533905932738);  permute_35 = None
        expand_15: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(mul_29, [64, 4, 512, 64]);  mul_29 = None
        clone_19: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_77: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_19, [256, 512, 64]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_71: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_29, [32768, 256])
        permute_36: "f32[256, 256]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_20: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg62_1, view_71, permute_36);  arg62_1 = view_71 = permute_36 = None
        view_72: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_20, [64, 512, 256]);  addmm_20 = None
        view_73: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_72, [64, 512, -1, 64]);  view_72 = None
        permute_37: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_40: "f32[64, 4, 64, 512]" = torch.ops.aten.permute.default(permute_37, [0, 1, 3, 2]);  permute_37 = None
        mul_30: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_40, 0.3535533905932738);  permute_40 = None
        expand_16: "f32[64, 4, 64, 512]" = torch.ops.aten.expand.default(mul_30, [64, 4, 64, 512]);  mul_30 = None
        clone_20: "f32[64, 4, 64, 512]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_78: "f32[256, 64, 512]" = torch.ops.aten.reshape.default(clone_20, [256, 64, 512]);  clone_20 = None
        bmm_6: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_77, view_78);  view_77 = view_78 = None
        view_79: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_6, [64, 4, 512, 512]);  bmm_6 = None
        full_default_10: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_9: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "f32[64, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_10, full_default_9);  full_default_10 = full_default_9 = None
        add_30: "f32[64, 4, 512, 512]" = torch.ops.aten.add.Tensor(view_79, where_6);  view_79 = where_6 = None
        eq_3: "b8[64, 4, 512, 512]" = torch.ops.aten.eq.Scalar(add_30, -inf)
        logical_not_6: "b8[64, 4, 512, 512]" = torch.ops.aten.logical_not.default(eq_3);  eq_3 = None
        any_4: "b8[64, 4, 512, 1]" = torch.ops.aten.any.dim(logical_not_6, -1, True);  logical_not_6 = None
        logical_not_7: "b8[64, 4, 512, 1]" = torch.ops.aten.logical_not.default(any_4);  any_4 = None
        full_default_11: "f32[64, 4, 512, 512]" = torch.ops.aten.full.default([64, 4, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_3: "f32[64, 4, 512, 1]" = torch.ops.aten.amax.default(add_30, [-1], True)
        sub_10: "f32[64, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_30, amax_3);  add_30 = amax_3 = None
        exp_3: "f32[64, 4, 512, 512]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[64, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        where_7: "f32[64, 4, 512, 512]" = torch.ops.aten.where.self(logical_not_7, full_default_11, div_3);  logical_not_7 = full_default_11 = div_3 = None
        expand_17: "f32[64, 4, 512, 512]" = torch.ops.aten.expand.default(where_7, [64, 4, 512, 512]);  where_7 = None
        view_80: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(expand_17, [256, 512, 512]);  expand_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_74: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_29, [32768, 256])
        permute_38: "f32[256, 256]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_21: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg64_1, view_74, permute_38);  arg64_1 = view_74 = permute_38 = None
        view_75: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_21, [64, 512, 256]);  addmm_21 = None
        view_76: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_75, [64, 512, -1, 64]);  view_75 = None
        permute_39: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_18: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(permute_39, [64, 4, 512, 64]);  permute_39 = None
        clone_21: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_81: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_21, [256, 512, 64]);  clone_21 = None
        bmm_7: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_80, view_81);  view_80 = view_81 = None
        view_82: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_7, [64, 4, 512, 64]);  bmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_41: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
        clone_22: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_83: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_22, [64, 512, -1]);  clone_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_84: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_83, [32768, 256]);  view_83 = None
        permute_42: "f32[256, 256]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        addmm_22: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg66_1, view_84, permute_42);  arg66_1 = view_84 = permute_42 = None
        view_85: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_22, [64, 512, 256]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_31: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(view_85, add_29);  view_85 = add_29 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
        getitem_14: "f32[64, 512, 1]" = var_mean_7[0]
        getitem_15: "f32[64, 512, 1]" = var_mean_7[1];  var_mean_7 = None
        sub_11: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_31, getitem_15);  add_31 = getitem_15 = None
        add_32: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        mul_31: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
        mul_32: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_31, arg67_1);  mul_31 = arg67_1 = None
        add_33: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_32, arg68_1);  mul_32 = arg68_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_86: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_33, [32768, 256])
        permute_43: "f32[256, 1024]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        addmm_23: "f32[32768, 1024]" = torch.ops.aten.addmm.default(arg70_1, view_86, permute_43);  arg70_1 = view_86 = permute_43 = None
        view_87: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_23, [64, 512, 1024]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_33: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_87, 0.5)
        mul_34: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_87, 0.7071067811865476);  view_87 = None
        erf_3: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_34);  mul_34 = None
        add_34: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_35: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_33, add_34);  mul_33 = add_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_88: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_35, [32768, 1024]);  mul_35 = None
        permute_44: "f32[1024, 256]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_24: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg72_1, view_88, permute_44);  arg72_1 = view_88 = permute_44 = None
        view_89: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_24, [64, 512, 256]);  addmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_35: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(view_89, add_33);  view_89 = add_33 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_16: "f32[64, 512, 1]" = var_mean_8[0]
        getitem_17: "f32[64, 512, 1]" = var_mean_8[1];  var_mean_8 = None
        sub_12: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_35, getitem_17);  add_35 = getitem_17 = None
        add_36: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        mul_36: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
        mul_37: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_36, arg73_1);  mul_36 = arg73_1 = None
        add_37: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_37, arg74_1);  mul_37 = arg74_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_90: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_37, [32768, 256])
        permute_45: "f32[256, 256]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_25: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg76_1, view_90, permute_45);  arg76_1 = view_90 = permute_45 = None
        view_91: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_25, [64, 512, 256]);  addmm_25 = None
        view_92: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_91, [64, 512, -1, 64]);  view_91 = None
        permute_46: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_38: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(permute_46, 0.3535533905932738);  permute_46 = None
        expand_19: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(mul_38, [64, 4, 512, 64]);  mul_38 = None
        clone_25: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_99: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_25, [256, 512, 64]);  clone_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_93: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_37, [32768, 256])
        permute_47: "f32[256, 256]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        addmm_26: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg78_1, view_93, permute_47);  arg78_1 = view_93 = permute_47 = None
        view_94: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_26, [64, 512, 256]);  addmm_26 = None
        view_95: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_94, [64, 512, -1, 64]);  view_94 = None
        permute_48: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_51: "f32[64, 4, 64, 512]" = torch.ops.aten.permute.default(permute_48, [0, 1, 3, 2]);  permute_48 = None
        mul_39: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_51, 0.3535533905932738);  permute_51 = None
        expand_20: "f32[64, 4, 64, 512]" = torch.ops.aten.expand.default(mul_39, [64, 4, 64, 512]);  mul_39 = None
        clone_26: "f32[64, 4, 64, 512]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_100: "f32[256, 64, 512]" = torch.ops.aten.reshape.default(clone_26, [256, 64, 512]);  clone_26 = None
        bmm_8: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_99, view_100);  view_99 = view_100 = None
        view_101: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_8, [64, 4, 512, 512]);  bmm_8 = None
        full_default_13: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_12: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8: "f32[64, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_13, full_default_12);  full_default_13 = full_default_12 = None
        add_38: "f32[64, 4, 512, 512]" = torch.ops.aten.add.Tensor(view_101, where_8);  view_101 = where_8 = None
        eq_4: "b8[64, 4, 512, 512]" = torch.ops.aten.eq.Scalar(add_38, -inf)
        logical_not_8: "b8[64, 4, 512, 512]" = torch.ops.aten.logical_not.default(eq_4);  eq_4 = None
        any_5: "b8[64, 4, 512, 1]" = torch.ops.aten.any.dim(logical_not_8, -1, True);  logical_not_8 = None
        logical_not_9: "b8[64, 4, 512, 1]" = torch.ops.aten.logical_not.default(any_5);  any_5 = None
        full_default_14: "f32[64, 4, 512, 512]" = torch.ops.aten.full.default([64, 4, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_4: "f32[64, 4, 512, 1]" = torch.ops.aten.amax.default(add_38, [-1], True)
        sub_13: "f32[64, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_38, amax_4);  add_38 = amax_4 = None
        exp_4: "f32[64, 4, 512, 512]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[64, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        where_9: "f32[64, 4, 512, 512]" = torch.ops.aten.where.self(logical_not_9, full_default_14, div_4);  logical_not_9 = full_default_14 = div_4 = None
        expand_21: "f32[64, 4, 512, 512]" = torch.ops.aten.expand.default(where_9, [64, 4, 512, 512]);  where_9 = None
        view_102: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(expand_21, [256, 512, 512]);  expand_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_96: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_37, [32768, 256])
        permute_49: "f32[256, 256]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_27: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg80_1, view_96, permute_49);  arg80_1 = view_96 = permute_49 = None
        view_97: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_27, [64, 512, 256]);  addmm_27 = None
        view_98: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_97, [64, 512, -1, 64]);  view_97 = None
        permute_50: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_22: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(permute_50, [64, 4, 512, 64]);  permute_50 = None
        clone_27: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_103: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_27, [256, 512, 64]);  clone_27 = None
        bmm_9: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_102, view_103);  view_102 = view_103 = None
        view_104: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_9, [64, 4, 512, 64]);  bmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_52: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
        clone_28: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_105: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_28, [64, 512, -1]);  clone_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_106: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_105, [32768, 256]);  view_105 = None
        permute_53: "f32[256, 256]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        addmm_28: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg82_1, view_106, permute_53);  arg82_1 = view_106 = permute_53 = None
        view_107: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_28, [64, 512, 256]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_39: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(view_107, add_37);  view_107 = add_37 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
        getitem_18: "f32[64, 512, 1]" = var_mean_9[0]
        getitem_19: "f32[64, 512, 1]" = var_mean_9[1];  var_mean_9 = None
        sub_14: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_39, getitem_19);  add_39 = getitem_19 = None
        add_40: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        mul_40: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
        mul_41: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_40, arg83_1);  mul_40 = arg83_1 = None
        add_41: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_41, arg84_1);  mul_41 = arg84_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_108: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_41, [32768, 256])
        permute_54: "f32[256, 1024]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_29: "f32[32768, 1024]" = torch.ops.aten.addmm.default(arg86_1, view_108, permute_54);  arg86_1 = view_108 = permute_54 = None
        view_109: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_29, [64, 512, 1024]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_42: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
        mul_43: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_109, 0.7071067811865476);  view_109 = None
        erf_4: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_43);  mul_43 = None
        add_42: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_44: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_42, add_42);  mul_42 = add_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_110: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_44, [32768, 1024]);  mul_44 = None
        permute_55: "f32[1024, 256]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_30: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg88_1, view_110, permute_55);  arg88_1 = view_110 = permute_55 = None
        view_111: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_30, [64, 512, 256]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_43: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(view_111, add_41);  view_111 = add_41 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_20: "f32[64, 512, 1]" = var_mean_10[0]
        getitem_21: "f32[64, 512, 1]" = var_mean_10[1];  var_mean_10 = None
        sub_15: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_43, getitem_21);  add_43 = getitem_21 = None
        add_44: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        mul_45: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
        mul_46: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_45, arg89_1);  mul_45 = arg89_1 = None
        add_45: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_46, arg90_1);  mul_46 = arg90_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_112: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_45, [32768, 256])
        permute_56: "f32[256, 256]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_31: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg92_1, view_112, permute_56);  arg92_1 = view_112 = permute_56 = None
        view_113: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_31, [64, 512, 256]);  addmm_31 = None
        view_114: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_113, [64, 512, -1, 64]);  view_113 = None
        permute_57: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_47: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(permute_57, 0.3535533905932738);  permute_57 = None
        expand_23: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(mul_47, [64, 4, 512, 64]);  mul_47 = None
        clone_31: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_121: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_31, [256, 512, 64]);  clone_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_115: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_45, [32768, 256])
        permute_58: "f32[256, 256]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        addmm_32: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg94_1, view_115, permute_58);  arg94_1 = view_115 = permute_58 = None
        view_116: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_32, [64, 512, 256]);  addmm_32 = None
        view_117: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_116, [64, 512, -1, 64]);  view_116 = None
        permute_59: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_62: "f32[64, 4, 64, 512]" = torch.ops.aten.permute.default(permute_59, [0, 1, 3, 2]);  permute_59 = None
        mul_48: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_62, 0.3535533905932738);  permute_62 = None
        expand_24: "f32[64, 4, 64, 512]" = torch.ops.aten.expand.default(mul_48, [64, 4, 64, 512]);  mul_48 = None
        clone_32: "f32[64, 4, 64, 512]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_122: "f32[256, 64, 512]" = torch.ops.aten.reshape.default(clone_32, [256, 64, 512]);  clone_32 = None
        bmm_10: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_121, view_122);  view_121 = view_122 = None
        view_123: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_10, [64, 4, 512, 512]);  bmm_10 = None
        full_default_16: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_15: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_10: "f32[64, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_16, full_default_15);  full_default_16 = full_default_15 = None
        add_46: "f32[64, 4, 512, 512]" = torch.ops.aten.add.Tensor(view_123, where_10);  view_123 = where_10 = None
        eq_5: "b8[64, 4, 512, 512]" = torch.ops.aten.eq.Scalar(add_46, -inf)
        logical_not_10: "b8[64, 4, 512, 512]" = torch.ops.aten.logical_not.default(eq_5);  eq_5 = None
        any_6: "b8[64, 4, 512, 1]" = torch.ops.aten.any.dim(logical_not_10, -1, True);  logical_not_10 = None
        logical_not_11: "b8[64, 4, 512, 1]" = torch.ops.aten.logical_not.default(any_6);  any_6 = None
        full_default_17: "f32[64, 4, 512, 512]" = torch.ops.aten.full.default([64, 4, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_5: "f32[64, 4, 512, 1]" = torch.ops.aten.amax.default(add_46, [-1], True)
        sub_16: "f32[64, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_46, amax_5);  add_46 = amax_5 = None
        exp_5: "f32[64, 4, 512, 512]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[64, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        where_11: "f32[64, 4, 512, 512]" = torch.ops.aten.where.self(logical_not_11, full_default_17, div_5);  logical_not_11 = full_default_17 = div_5 = None
        expand_25: "f32[64, 4, 512, 512]" = torch.ops.aten.expand.default(where_11, [64, 4, 512, 512]);  where_11 = None
        view_124: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(expand_25, [256, 512, 512]);  expand_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_118: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_45, [32768, 256])
        permute_60: "f32[256, 256]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_33: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg96_1, view_118, permute_60);  arg96_1 = view_118 = permute_60 = None
        view_119: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_33, [64, 512, 256]);  addmm_33 = None
        view_120: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_119, [64, 512, -1, 64]);  view_119 = None
        permute_61: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_26: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(permute_61, [64, 4, 512, 64]);  permute_61 = None
        clone_33: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_125: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_33, [256, 512, 64]);  clone_33 = None
        bmm_11: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_124, view_125);  view_124 = view_125 = None
        view_126: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_11, [64, 4, 512, 64]);  bmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_63: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
        clone_34: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_127: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_34, [64, 512, -1]);  clone_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_128: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_127, [32768, 256]);  view_127 = None
        permute_64: "f32[256, 256]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_34: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg98_1, view_128, permute_64);  arg98_1 = view_128 = permute_64 = None
        view_129: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_34, [64, 512, 256]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_47: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(view_129, add_45);  view_129 = add_45 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_47, [2], correction = 0, keepdim = True)
        getitem_22: "f32[64, 512, 1]" = var_mean_11[0]
        getitem_23: "f32[64, 512, 1]" = var_mean_11[1];  var_mean_11 = None
        sub_17: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_47, getitem_23);  add_47 = getitem_23 = None
        add_48: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        mul_49: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
        mul_50: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_49, arg99_1);  mul_49 = arg99_1 = None
        add_49: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_50, arg100_1);  mul_50 = arg100_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_130: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_49, [32768, 256])
        permute_65: "f32[256, 1024]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_35: "f32[32768, 1024]" = torch.ops.aten.addmm.default(arg102_1, view_130, permute_65);  arg102_1 = view_130 = permute_65 = None
        view_131: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_35, [64, 512, 1024]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_51: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_131, 0.5)
        mul_52: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_131, 0.7071067811865476);  view_131 = None
        erf_5: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_52);  mul_52 = None
        add_50: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_53: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_51, add_50);  mul_51 = add_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_132: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_53, [32768, 1024]);  mul_53 = None
        permute_66: "f32[1024, 256]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_36: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg104_1, view_132, permute_66);  arg104_1 = view_132 = permute_66 = None
        view_133: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_36, [64, 512, 256]);  addmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_51: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(view_133, add_49);  view_133 = add_49 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
        getitem_24: "f32[64, 512, 1]" = var_mean_12[0]
        getitem_25: "f32[64, 512, 1]" = var_mean_12[1];  var_mean_12 = None
        sub_18: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_51, getitem_25);  add_51 = getitem_25 = None
        add_52: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_54: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
        mul_55: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_54, arg105_1);  mul_54 = arg105_1 = None
        add_53: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_55, arg106_1);  mul_55 = arg106_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_134: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_53, [32768, 256])
        permute_67: "f32[256, 256]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_37: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg108_1, view_134, permute_67);  arg108_1 = view_134 = permute_67 = None
        view_135: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_37, [64, 512, 256]);  addmm_37 = None
        view_136: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_135, [64, 512, -1, 64]);  view_135 = None
        permute_68: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_56: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(permute_68, 0.3535533905932738);  permute_68 = None
        expand_27: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(mul_56, [64, 4, 512, 64]);  mul_56 = None
        clone_37: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_143: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_37, [256, 512, 64]);  clone_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_137: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_53, [32768, 256])
        permute_69: "f32[256, 256]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_38: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg110_1, view_137, permute_69);  arg110_1 = view_137 = permute_69 = None
        view_138: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_38, [64, 512, 256]);  addmm_38 = None
        view_139: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_138, [64, 512, -1, 64]);  view_138 = None
        permute_70: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_73: "f32[64, 4, 64, 512]" = torch.ops.aten.permute.default(permute_70, [0, 1, 3, 2]);  permute_70 = None
        mul_57: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_73, 0.3535533905932738);  permute_73 = None
        expand_28: "f32[64, 4, 64, 512]" = torch.ops.aten.expand.default(mul_57, [64, 4, 64, 512]);  mul_57 = None
        clone_38: "f32[64, 4, 64, 512]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_144: "f32[256, 64, 512]" = torch.ops.aten.reshape.default(clone_38, [256, 64, 512]);  clone_38 = None
        bmm_12: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_143, view_144);  view_143 = view_144 = None
        view_145: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_12, [64, 4, 512, 512]);  bmm_12 = None
        full_default_19: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_18: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: "f32[64, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_19, full_default_18);  full_default_19 = full_default_18 = None
        add_54: "f32[64, 4, 512, 512]" = torch.ops.aten.add.Tensor(view_145, where_12);  view_145 = where_12 = None
        eq_6: "b8[64, 4, 512, 512]" = torch.ops.aten.eq.Scalar(add_54, -inf)
        logical_not_12: "b8[64, 4, 512, 512]" = torch.ops.aten.logical_not.default(eq_6);  eq_6 = None
        any_7: "b8[64, 4, 512, 1]" = torch.ops.aten.any.dim(logical_not_12, -1, True);  logical_not_12 = None
        logical_not_13: "b8[64, 4, 512, 1]" = torch.ops.aten.logical_not.default(any_7);  any_7 = None
        full_default_20: "f32[64, 4, 512, 512]" = torch.ops.aten.full.default([64, 4, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_6: "f32[64, 4, 512, 1]" = torch.ops.aten.amax.default(add_54, [-1], True)
        sub_19: "f32[64, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_54, amax_6);  add_54 = amax_6 = None
        exp_6: "f32[64, 4, 512, 512]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_7: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: "f32[64, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        where_13: "f32[64, 4, 512, 512]" = torch.ops.aten.where.self(logical_not_13, full_default_20, div_6);  logical_not_13 = full_default_20 = div_6 = None
        expand_29: "f32[64, 4, 512, 512]" = torch.ops.aten.expand.default(where_13, [64, 4, 512, 512]);  where_13 = None
        view_146: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(expand_29, [256, 512, 512]);  expand_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_140: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_53, [32768, 256])
        permute_71: "f32[256, 256]" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        addmm_39: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg112_1, view_140, permute_71);  arg112_1 = view_140 = permute_71 = None
        view_141: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_39, [64, 512, 256]);  addmm_39 = None
        view_142: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_141, [64, 512, -1, 64]);  view_141 = None
        permute_72: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_30: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(permute_72, [64, 4, 512, 64]);  permute_72 = None
        clone_39: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_147: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_39, [256, 512, 64]);  clone_39 = None
        bmm_13: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_146, view_147);  view_146 = view_147 = None
        view_148: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_13, [64, 4, 512, 64]);  bmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_74: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
        clone_40: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_149: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_40, [64, 512, -1]);  clone_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_150: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_149, [32768, 256]);  view_149 = None
        permute_75: "f32[256, 256]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        addmm_40: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg114_1, view_150, permute_75);  arg114_1 = view_150 = permute_75 = None
        view_151: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_40, [64, 512, 256]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_55: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(view_151, add_53);  view_151 = add_53 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
        getitem_26: "f32[64, 512, 1]" = var_mean_13[0]
        getitem_27: "f32[64, 512, 1]" = var_mean_13[1];  var_mean_13 = None
        sub_20: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_55, getitem_27);  add_55 = getitem_27 = None
        add_56: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        mul_58: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
        mul_59: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_58, arg115_1);  mul_58 = arg115_1 = None
        add_57: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_59, arg116_1);  mul_59 = arg116_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_152: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_57, [32768, 256])
        permute_76: "f32[256, 1024]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        addmm_41: "f32[32768, 1024]" = torch.ops.aten.addmm.default(arg118_1, view_152, permute_76);  arg118_1 = view_152 = permute_76 = None
        view_153: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_41, [64, 512, 1024]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_60: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_153, 0.5)
        mul_61: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_153, 0.7071067811865476);  view_153 = None
        erf_6: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
        add_58: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_62: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_60, add_58);  mul_60 = add_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_154: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_62, [32768, 1024]);  mul_62 = None
        permute_77: "f32[1024, 256]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_42: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg120_1, view_154, permute_77);  arg120_1 = view_154 = permute_77 = None
        view_155: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_42, [64, 512, 256]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_59: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(view_155, add_57);  view_155 = add_57 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
        getitem_28: "f32[64, 512, 1]" = var_mean_14[0]
        getitem_29: "f32[64, 512, 1]" = var_mean_14[1];  var_mean_14 = None
        sub_21: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_59, getitem_29);  add_59 = getitem_29 = None
        add_60: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_63: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = rsqrt_14 = None
        mul_64: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_63, arg121_1);  mul_63 = arg121_1 = None
        add_61: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_64, arg122_1);  mul_64 = arg122_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_156: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_61, [32768, 256])
        permute_78: "f32[256, 256]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_43: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg124_1, view_156, permute_78);  arg124_1 = view_156 = permute_78 = None
        view_157: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_43, [64, 512, 256]);  addmm_43 = None
        view_158: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_157, [64, 512, -1, 64]);  view_157 = None
        permute_79: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_158, [0, 2, 1, 3]);  view_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_65: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(permute_79, 0.3535533905932738);  permute_79 = None
        expand_31: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(mul_65, [64, 4, 512, 64]);  mul_65 = None
        clone_43: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_165: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_43, [256, 512, 64]);  clone_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_159: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_61, [32768, 256])
        permute_80: "f32[256, 256]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        addmm_44: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg126_1, view_159, permute_80);  arg126_1 = view_159 = permute_80 = None
        view_160: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_44, [64, 512, 256]);  addmm_44 = None
        view_161: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_160, [64, 512, -1, 64]);  view_160 = None
        permute_81: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_161, [0, 2, 1, 3]);  view_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_84: "f32[64, 4, 64, 512]" = torch.ops.aten.permute.default(permute_81, [0, 1, 3, 2]);  permute_81 = None
        mul_66: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_84, 0.3535533905932738);  permute_84 = None
        expand_32: "f32[64, 4, 64, 512]" = torch.ops.aten.expand.default(mul_66, [64, 4, 64, 512]);  mul_66 = None
        clone_44: "f32[64, 4, 64, 512]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_166: "f32[256, 64, 512]" = torch.ops.aten.reshape.default(clone_44, [256, 64, 512]);  clone_44 = None
        bmm_14: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_165, view_166);  view_165 = view_166 = None
        view_167: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_14, [64, 4, 512, 512]);  bmm_14 = None
        full_default_22: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_21: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_14: "f32[64, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_22, full_default_21);  full_default_22 = full_default_21 = None
        add_62: "f32[64, 4, 512, 512]" = torch.ops.aten.add.Tensor(view_167, where_14);  view_167 = where_14 = None
        eq_7: "b8[64, 4, 512, 512]" = torch.ops.aten.eq.Scalar(add_62, -inf)
        logical_not_14: "b8[64, 4, 512, 512]" = torch.ops.aten.logical_not.default(eq_7);  eq_7 = None
        any_8: "b8[64, 4, 512, 1]" = torch.ops.aten.any.dim(logical_not_14, -1, True);  logical_not_14 = None
        logical_not_15: "b8[64, 4, 512, 1]" = torch.ops.aten.logical_not.default(any_8);  any_8 = None
        full_default_23: "f32[64, 4, 512, 512]" = torch.ops.aten.full.default([64, 4, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_7: "f32[64, 4, 512, 1]" = torch.ops.aten.amax.default(add_62, [-1], True)
        sub_22: "f32[64, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_62, amax_7);  add_62 = amax_7 = None
        exp_7: "f32[64, 4, 512, 512]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_8: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: "f32[64, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        where_15: "f32[64, 4, 512, 512]" = torch.ops.aten.where.self(logical_not_15, full_default_23, div_7);  logical_not_15 = full_default_23 = div_7 = None
        expand_33: "f32[64, 4, 512, 512]" = torch.ops.aten.expand.default(where_15, [64, 4, 512, 512]);  where_15 = None
        view_168: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(expand_33, [256, 512, 512]);  expand_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_162: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_61, [32768, 256])
        permute_82: "f32[256, 256]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_45: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg128_1, view_162, permute_82);  arg128_1 = view_162 = permute_82 = None
        view_163: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_45, [64, 512, 256]);  addmm_45 = None
        view_164: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_163, [64, 512, -1, 64]);  view_163 = None
        permute_83: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_34: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(permute_83, [64, 4, 512, 64]);  permute_83 = None
        clone_45: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_169: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_45, [256, 512, 64]);  clone_45 = None
        bmm_15: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_168, view_169);  view_168 = view_169 = None
        view_170: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_15, [64, 4, 512, 64]);  bmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_85: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
        clone_46: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_171: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_46, [64, 512, -1]);  clone_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_172: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_171, [32768, 256]);  view_171 = None
        permute_86: "f32[256, 256]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        addmm_46: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg130_1, view_172, permute_86);  arg130_1 = view_172 = permute_86 = None
        view_173: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_46, [64, 512, 256]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_63: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(view_173, add_61);  view_173 = add_61 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
        getitem_30: "f32[64, 512, 1]" = var_mean_15[0]
        getitem_31: "f32[64, 512, 1]" = var_mean_15[1];  var_mean_15 = None
        sub_23: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_63, getitem_31);  add_63 = getitem_31 = None
        add_64: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        mul_67: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
        mul_68: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_67, arg131_1);  mul_67 = arg131_1 = None
        add_65: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_68, arg132_1);  mul_68 = arg132_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_174: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_65, [32768, 256])
        permute_87: "f32[256, 1024]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_47: "f32[32768, 1024]" = torch.ops.aten.addmm.default(arg134_1, view_174, permute_87);  arg134_1 = view_174 = permute_87 = None
        view_175: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_47, [64, 512, 1024]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_69: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_175, 0.5)
        mul_70: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_175, 0.7071067811865476);  view_175 = None
        erf_7: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_70);  mul_70 = None
        add_66: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_71: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_69, add_66);  mul_69 = add_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_176: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_71, [32768, 1024]);  mul_71 = None
        permute_88: "f32[1024, 256]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_48: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg136_1, view_176, permute_88);  arg136_1 = view_176 = permute_88 = None
        view_177: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_48, [64, 512, 256]);  addmm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_67: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(view_177, add_65);  view_177 = add_65 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
        getitem_32: "f32[64, 512, 1]" = var_mean_16[0]
        getitem_33: "f32[64, 512, 1]" = var_mean_16[1];  var_mean_16 = None
        sub_24: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_67, getitem_33);  add_67 = getitem_33 = None
        add_68: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        mul_72: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = rsqrt_16 = None
        mul_73: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_72, arg137_1);  mul_72 = arg137_1 = None
        add_69: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_73, arg138_1);  mul_73 = arg138_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_178: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_69, [32768, 256])
        permute_89: "f32[256, 256]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_49: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg140_1, view_178, permute_89);  arg140_1 = view_178 = permute_89 = None
        view_179: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_49, [64, 512, 256]);  addmm_49 = None
        view_180: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_179, [64, 512, -1, 64]);  view_179 = None
        permute_90: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_180, [0, 2, 1, 3]);  view_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_74: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(permute_90, 0.3535533905932738);  permute_90 = None
        expand_35: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(mul_74, [64, 4, 512, 64]);  mul_74 = None
        clone_49: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_187: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_49, [256, 512, 64]);  clone_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_181: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_69, [32768, 256])
        permute_91: "f32[256, 256]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        addmm_50: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg142_1, view_181, permute_91);  arg142_1 = view_181 = permute_91 = None
        view_182: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_50, [64, 512, 256]);  addmm_50 = None
        view_183: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_182, [64, 512, -1, 64]);  view_182 = None
        permute_92: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_95: "f32[64, 4, 64, 512]" = torch.ops.aten.permute.default(permute_92, [0, 1, 3, 2]);  permute_92 = None
        mul_75: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_95, 0.3535533905932738);  permute_95 = None
        expand_36: "f32[64, 4, 64, 512]" = torch.ops.aten.expand.default(mul_75, [64, 4, 64, 512]);  mul_75 = None
        clone_50: "f32[64, 4, 64, 512]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_188: "f32[256, 64, 512]" = torch.ops.aten.reshape.default(clone_50, [256, 64, 512]);  clone_50 = None
        bmm_16: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_187, view_188);  view_187 = view_188 = None
        view_189: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_16, [64, 4, 512, 512]);  bmm_16 = None
        full_default_25: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_24: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_16: "f32[64, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_25, full_default_24);  full_default_25 = full_default_24 = None
        add_70: "f32[64, 4, 512, 512]" = torch.ops.aten.add.Tensor(view_189, where_16);  view_189 = where_16 = None
        eq_8: "b8[64, 4, 512, 512]" = torch.ops.aten.eq.Scalar(add_70, -inf)
        logical_not_16: "b8[64, 4, 512, 512]" = torch.ops.aten.logical_not.default(eq_8);  eq_8 = None
        any_9: "b8[64, 4, 512, 1]" = torch.ops.aten.any.dim(logical_not_16, -1, True);  logical_not_16 = None
        logical_not_17: "b8[64, 4, 512, 1]" = torch.ops.aten.logical_not.default(any_9);  any_9 = None
        full_default_26: "f32[64, 4, 512, 512]" = torch.ops.aten.full.default([64, 4, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_8: "f32[64, 4, 512, 1]" = torch.ops.aten.amax.default(add_70, [-1], True)
        sub_25: "f32[64, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_70, amax_8);  add_70 = amax_8 = None
        exp_8: "f32[64, 4, 512, 512]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_9: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8: "f32[64, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        where_17: "f32[64, 4, 512, 512]" = torch.ops.aten.where.self(logical_not_17, full_default_26, div_8);  logical_not_17 = full_default_26 = div_8 = None
        expand_37: "f32[64, 4, 512, 512]" = torch.ops.aten.expand.default(where_17, [64, 4, 512, 512]);  where_17 = None
        view_190: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(expand_37, [256, 512, 512]);  expand_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_184: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_69, [32768, 256])
        permute_93: "f32[256, 256]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_51: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg144_1, view_184, permute_93);  arg144_1 = view_184 = permute_93 = None
        view_185: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_51, [64, 512, 256]);  addmm_51 = None
        view_186: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_185, [64, 512, -1, 64]);  view_185 = None
        permute_94: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_38: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(permute_94, [64, 4, 512, 64]);  permute_94 = None
        clone_51: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_191: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_51, [256, 512, 64]);  clone_51 = None
        bmm_17: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_190, view_191);  view_190 = view_191 = None
        view_192: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_17, [64, 4, 512, 64]);  bmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_96: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
        clone_52: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_96, memory_format = torch.contiguous_format);  permute_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_193: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_52, [64, 512, -1]);  clone_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_194: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_193, [32768, 256]);  view_193 = None
        permute_97: "f32[256, 256]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_52: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg146_1, view_194, permute_97);  arg146_1 = view_194 = permute_97 = None
        view_195: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_52, [64, 512, 256]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_71: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(view_195, add_69);  view_195 = add_69 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
        getitem_34: "f32[64, 512, 1]" = var_mean_17[0]
        getitem_35: "f32[64, 512, 1]" = var_mean_17[1];  var_mean_17 = None
        sub_26: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_71, getitem_35);  add_71 = getitem_35 = None
        add_72: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        mul_76: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = rsqrt_17 = None
        mul_77: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_76, arg147_1);  mul_76 = arg147_1 = None
        add_73: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_77, arg148_1);  mul_77 = arg148_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_196: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_73, [32768, 256])
        permute_98: "f32[256, 1024]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        addmm_53: "f32[32768, 1024]" = torch.ops.aten.addmm.default(arg150_1, view_196, permute_98);  arg150_1 = view_196 = permute_98 = None
        view_197: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_53, [64, 512, 1024]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_78: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_197, 0.5)
        mul_79: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_197, 0.7071067811865476);  view_197 = None
        erf_8: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_79);  mul_79 = None
        add_74: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_80: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_78, add_74);  mul_78 = add_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_198: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_80, [32768, 1024]);  mul_80 = None
        permute_99: "f32[1024, 256]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_54: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg152_1, view_198, permute_99);  arg152_1 = view_198 = permute_99 = None
        view_199: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_54, [64, 512, 256]);  addmm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_75: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(view_199, add_73);  view_199 = add_73 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
        getitem_36: "f32[64, 512, 1]" = var_mean_18[0]
        getitem_37: "f32[64, 512, 1]" = var_mean_18[1];  var_mean_18 = None
        sub_27: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_75, getitem_37);  add_75 = getitem_37 = None
        add_76: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        mul_81: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = rsqrt_18 = None
        mul_82: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_81, arg153_1);  mul_81 = arg153_1 = None
        add_77: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_82, arg154_1);  mul_82 = arg154_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_200: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_77, [32768, 256])
        permute_100: "f32[256, 256]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        addmm_55: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg156_1, view_200, permute_100);  arg156_1 = view_200 = permute_100 = None
        view_201: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_55, [64, 512, 256]);  addmm_55 = None
        view_202: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_201, [64, 512, -1, 64]);  view_201 = None
        permute_101: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_83: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(permute_101, 0.3535533905932738);  permute_101 = None
        expand_39: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(mul_83, [64, 4, 512, 64]);  mul_83 = None
        clone_55: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_209: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_55, [256, 512, 64]);  clone_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_203: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_77, [32768, 256])
        permute_102: "f32[256, 256]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        addmm_56: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg158_1, view_203, permute_102);  arg158_1 = view_203 = permute_102 = None
        view_204: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_56, [64, 512, 256]);  addmm_56 = None
        view_205: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_204, [64, 512, -1, 64]);  view_204 = None
        permute_103: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_106: "f32[64, 4, 64, 512]" = torch.ops.aten.permute.default(permute_103, [0, 1, 3, 2]);  permute_103 = None
        mul_84: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_106, 0.3535533905932738);  permute_106 = None
        expand_40: "f32[64, 4, 64, 512]" = torch.ops.aten.expand.default(mul_84, [64, 4, 64, 512]);  mul_84 = None
        clone_56: "f32[64, 4, 64, 512]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_210: "f32[256, 64, 512]" = torch.ops.aten.reshape.default(clone_56, [256, 64, 512]);  clone_56 = None
        bmm_18: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_209, view_210);  view_209 = view_210 = None
        view_211: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_18, [64, 4, 512, 512]);  bmm_18 = None
        full_default_28: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_27: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_18: "f32[64, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_28, full_default_27);  full_default_28 = full_default_27 = None
        add_78: "f32[64, 4, 512, 512]" = torch.ops.aten.add.Tensor(view_211, where_18);  view_211 = where_18 = None
        eq_9: "b8[64, 4, 512, 512]" = torch.ops.aten.eq.Scalar(add_78, -inf)
        logical_not_18: "b8[64, 4, 512, 512]" = torch.ops.aten.logical_not.default(eq_9);  eq_9 = None
        any_10: "b8[64, 4, 512, 1]" = torch.ops.aten.any.dim(logical_not_18, -1, True);  logical_not_18 = None
        logical_not_19: "b8[64, 4, 512, 1]" = torch.ops.aten.logical_not.default(any_10);  any_10 = None
        full_default_29: "f32[64, 4, 512, 512]" = torch.ops.aten.full.default([64, 4, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_9: "f32[64, 4, 512, 1]" = torch.ops.aten.amax.default(add_78, [-1], True)
        sub_28: "f32[64, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_78, amax_9);  add_78 = amax_9 = None
        exp_9: "f32[64, 4, 512, 512]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_10: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9: "f32[64, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        where_19: "f32[64, 4, 512, 512]" = torch.ops.aten.where.self(logical_not_19, full_default_29, div_9);  logical_not_19 = full_default_29 = div_9 = None
        expand_41: "f32[64, 4, 512, 512]" = torch.ops.aten.expand.default(where_19, [64, 4, 512, 512]);  where_19 = None
        view_212: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(expand_41, [256, 512, 512]);  expand_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_206: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_77, [32768, 256])
        permute_104: "f32[256, 256]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm_57: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg160_1, view_206, permute_104);  arg160_1 = view_206 = permute_104 = None
        view_207: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_57, [64, 512, 256]);  addmm_57 = None
        view_208: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_207, [64, 512, -1, 64]);  view_207 = None
        permute_105: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_42: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(permute_105, [64, 4, 512, 64]);  permute_105 = None
        clone_57: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_213: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_57, [256, 512, 64]);  clone_57 = None
        bmm_19: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_212, view_213);  view_212 = view_213 = None
        view_214: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_19, [64, 4, 512, 64]);  bmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_107: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_214, [0, 2, 1, 3]);  view_214 = None
        clone_58: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_107, memory_format = torch.contiguous_format);  permute_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_215: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_58, [64, 512, -1]);  clone_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_216: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_215, [32768, 256]);  view_215 = None
        permute_108: "f32[256, 256]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        addmm_58: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg162_1, view_216, permute_108);  arg162_1 = view_216 = permute_108 = None
        view_217: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_58, [64, 512, 256]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_79: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(view_217, add_77);  view_217 = add_77 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_79, [2], correction = 0, keepdim = True)
        getitem_38: "f32[64, 512, 1]" = var_mean_19[0]
        getitem_39: "f32[64, 512, 1]" = var_mean_19[1];  var_mean_19 = None
        sub_29: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_79, getitem_39);  add_79 = getitem_39 = None
        add_80: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        mul_85: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
        mul_86: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_85, arg163_1);  mul_85 = arg163_1 = None
        add_81: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_86, arg164_1);  mul_86 = arg164_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_218: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_81, [32768, 256])
        permute_109: "f32[256, 1024]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_59: "f32[32768, 1024]" = torch.ops.aten.addmm.default(arg166_1, view_218, permute_109);  arg166_1 = view_218 = permute_109 = None
        view_219: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_59, [64, 512, 1024]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_87: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_219, 0.5)
        mul_88: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_219, 0.7071067811865476);  view_219 = None
        erf_9: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
        add_82: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_89: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_87, add_82);  mul_87 = add_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_220: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_89, [32768, 1024]);  mul_89 = None
        permute_110: "f32[1024, 256]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        addmm_60: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg168_1, view_220, permute_110);  arg168_1 = view_220 = permute_110 = None
        view_221: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_60, [64, 512, 256]);  addmm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_83: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(view_221, add_81);  view_221 = add_81 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
        getitem_40: "f32[64, 512, 1]" = var_mean_20[0]
        getitem_41: "f32[64, 512, 1]" = var_mean_20[1];  var_mean_20 = None
        sub_30: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_83, getitem_41);  add_83 = getitem_41 = None
        add_84: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        mul_90: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = rsqrt_20 = None
        mul_91: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_90, arg169_1);  mul_90 = arg169_1 = None
        add_85: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_91, arg170_1);  mul_91 = arg170_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_222: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_85, [32768, 256])
        permute_111: "f32[256, 256]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_61: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg172_1, view_222, permute_111);  arg172_1 = view_222 = permute_111 = None
        view_223: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_61, [64, 512, 256]);  addmm_61 = None
        view_224: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_223, [64, 512, -1, 64]);  view_223 = None
        permute_112: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_92: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(permute_112, 0.3535533905932738);  permute_112 = None
        expand_43: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(mul_92, [64, 4, 512, 64]);  mul_92 = None
        clone_61: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_231: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_61, [256, 512, 64]);  clone_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_225: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_85, [32768, 256])
        permute_113: "f32[256, 256]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        addmm_62: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg174_1, view_225, permute_113);  arg174_1 = view_225 = permute_113 = None
        view_226: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_62, [64, 512, 256]);  addmm_62 = None
        view_227: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_226, [64, 512, -1, 64]);  view_226 = None
        permute_114: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_117: "f32[64, 4, 64, 512]" = torch.ops.aten.permute.default(permute_114, [0, 1, 3, 2]);  permute_114 = None
        mul_93: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_117, 0.3535533905932738);  permute_117 = None
        expand_44: "f32[64, 4, 64, 512]" = torch.ops.aten.expand.default(mul_93, [64, 4, 64, 512]);  mul_93 = None
        clone_62: "f32[64, 4, 64, 512]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_232: "f32[256, 64, 512]" = torch.ops.aten.reshape.default(clone_62, [256, 64, 512]);  clone_62 = None
        bmm_20: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_231, view_232);  view_231 = view_232 = None
        view_233: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_20, [64, 4, 512, 512]);  bmm_20 = None
        full_default_31: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_30: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_20: "f32[64, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_31, full_default_30);  full_default_31 = full_default_30 = None
        add_86: "f32[64, 4, 512, 512]" = torch.ops.aten.add.Tensor(view_233, where_20);  view_233 = where_20 = None
        eq_10: "b8[64, 4, 512, 512]" = torch.ops.aten.eq.Scalar(add_86, -inf)
        logical_not_20: "b8[64, 4, 512, 512]" = torch.ops.aten.logical_not.default(eq_10);  eq_10 = None
        any_11: "b8[64, 4, 512, 1]" = torch.ops.aten.any.dim(logical_not_20, -1, True);  logical_not_20 = None
        logical_not_21: "b8[64, 4, 512, 1]" = torch.ops.aten.logical_not.default(any_11);  any_11 = None
        full_default_32: "f32[64, 4, 512, 512]" = torch.ops.aten.full.default([64, 4, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_10: "f32[64, 4, 512, 1]" = torch.ops.aten.amax.default(add_86, [-1], True)
        sub_31: "f32[64, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_86, amax_10);  add_86 = amax_10 = None
        exp_10: "f32[64, 4, 512, 512]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_11: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10: "f32[64, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        where_21: "f32[64, 4, 512, 512]" = torch.ops.aten.where.self(logical_not_21, full_default_32, div_10);  logical_not_21 = full_default_32 = div_10 = None
        expand_45: "f32[64, 4, 512, 512]" = torch.ops.aten.expand.default(where_21, [64, 4, 512, 512]);  where_21 = None
        view_234: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(expand_45, [256, 512, 512]);  expand_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_228: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_85, [32768, 256])
        permute_115: "f32[256, 256]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        addmm_63: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg176_1, view_228, permute_115);  arg176_1 = view_228 = permute_115 = None
        view_229: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_63, [64, 512, 256]);  addmm_63 = None
        view_230: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_229, [64, 512, -1, 64]);  view_229 = None
        permute_116: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1, 3]);  view_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_46: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(permute_116, [64, 4, 512, 64]);  permute_116 = None
        clone_63: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_235: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_63, [256, 512, 64]);  clone_63 = None
        bmm_21: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_234, view_235);  view_234 = view_235 = None
        view_236: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_21, [64, 4, 512, 64]);  bmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_118: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
        clone_64: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_118, memory_format = torch.contiguous_format);  permute_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_237: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_64, [64, 512, -1]);  clone_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_238: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_237, [32768, 256]);  view_237 = None
        permute_119: "f32[256, 256]" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        addmm_64: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg178_1, view_238, permute_119);  arg178_1 = view_238 = permute_119 = None
        view_239: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_64, [64, 512, 256]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_87: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(view_239, add_85);  view_239 = add_85 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
        getitem_42: "f32[64, 512, 1]" = var_mean_21[0]
        getitem_43: "f32[64, 512, 1]" = var_mean_21[1];  var_mean_21 = None
        sub_32: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_87, getitem_43);  add_87 = getitem_43 = None
        add_88: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        mul_94: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
        mul_95: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_94, arg179_1);  mul_94 = arg179_1 = None
        add_89: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_95, arg180_1);  mul_95 = arg180_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_240: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_89, [32768, 256])
        permute_120: "f32[256, 1024]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        addmm_65: "f32[32768, 1024]" = torch.ops.aten.addmm.default(arg182_1, view_240, permute_120);  arg182_1 = view_240 = permute_120 = None
        view_241: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_65, [64, 512, 1024]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_96: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_241, 0.5)
        mul_97: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_241, 0.7071067811865476);  view_241 = None
        erf_10: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_97);  mul_97 = None
        add_90: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_98: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_96, add_90);  mul_96 = add_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_242: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_98, [32768, 1024]);  mul_98 = None
        permute_121: "f32[1024, 256]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        addmm_66: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg184_1, view_242, permute_121);  arg184_1 = view_242 = permute_121 = None
        view_243: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_66, [64, 512, 256]);  addmm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_91: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(view_243, add_89);  view_243 = add_89 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
        getitem_44: "f32[64, 512, 1]" = var_mean_22[0]
        getitem_45: "f32[64, 512, 1]" = var_mean_22[1];  var_mean_22 = None
        sub_33: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_91, getitem_45);  add_91 = getitem_45 = None
        add_92: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        mul_99: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = rsqrt_22 = None
        mul_100: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_99, arg185_1);  mul_99 = arg185_1 = None
        add_93: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_100, arg186_1);  mul_100 = arg186_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_244: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_93, [32768, 256])
        permute_122: "f32[256, 256]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_67: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg188_1, view_244, permute_122);  arg188_1 = view_244 = permute_122 = None
        view_245: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_67, [64, 512, 256]);  addmm_67 = None
        view_246: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_245, [64, 512, -1, 64]);  view_245 = None
        permute_123: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_101: "f32[64, 4, 512, 64]" = torch.ops.aten.mul.Scalar(permute_123, 0.3535533905932738);  permute_123 = None
        expand_47: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(mul_101, [64, 4, 512, 64]);  mul_101 = None
        clone_67: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_253: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_67, [256, 512, 64]);  clone_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_247: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_93, [32768, 256])
        permute_124: "f32[256, 256]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        addmm_68: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg190_1, view_247, permute_124);  arg190_1 = view_247 = permute_124 = None
        view_248: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_68, [64, 512, 256]);  addmm_68 = None
        view_249: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_248, [64, 512, -1, 64]);  view_248 = None
        permute_125: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_128: "f32[64, 4, 64, 512]" = torch.ops.aten.permute.default(permute_125, [0, 1, 3, 2]);  permute_125 = None
        mul_102: "f32[64, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_128, 0.3535533905932738);  permute_128 = None
        expand_48: "f32[64, 4, 64, 512]" = torch.ops.aten.expand.default(mul_102, [64, 4, 64, 512]);  mul_102 = None
        clone_68: "f32[64, 4, 64, 512]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_254: "f32[256, 64, 512]" = torch.ops.aten.reshape.default(clone_68, [256, 64, 512]);  clone_68 = None
        bmm_22: "f32[256, 512, 512]" = torch.ops.aten.bmm.default(view_253, view_254);  view_253 = view_254 = None
        view_255: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_22, [64, 4, 512, 512]);  bmm_22 = None
        full_default_34: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_33: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_22: "f32[64, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_34, full_default_33);  expand_2 = full_default_34 = full_default_33 = None
        add_94: "f32[64, 4, 512, 512]" = torch.ops.aten.add.Tensor(view_255, where_22);  view_255 = where_22 = None
        eq_11: "b8[64, 4, 512, 512]" = torch.ops.aten.eq.Scalar(add_94, -inf)
        logical_not_22: "b8[64, 4, 512, 512]" = torch.ops.aten.logical_not.default(eq_11);  eq_11 = None
        any_12: "b8[64, 4, 512, 1]" = torch.ops.aten.any.dim(logical_not_22, -1, True);  logical_not_22 = None
        logical_not_23: "b8[64, 4, 512, 1]" = torch.ops.aten.logical_not.default(any_12);  any_12 = None
        full_default_35: "f32[64, 4, 512, 512]" = torch.ops.aten.full.default([64, 4, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_11: "f32[64, 4, 512, 1]" = torch.ops.aten.amax.default(add_94, [-1], True)
        sub_34: "f32[64, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_94, amax_11);  add_94 = amax_11 = None
        exp_11: "f32[64, 4, 512, 512]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_12: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11: "f32[64, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        where_23: "f32[64, 4, 512, 512]" = torch.ops.aten.where.self(logical_not_23, full_default_35, div_11);  logical_not_23 = full_default_35 = div_11 = None
        expand_49: "f32[64, 4, 512, 512]" = torch.ops.aten.expand.default(where_23, [64, 4, 512, 512]);  where_23 = None
        view_256: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(expand_49, [256, 512, 512]);  expand_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        view_250: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_93, [32768, 256])
        permute_126: "f32[256, 256]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_69: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg192_1, view_250, permute_126);  arg192_1 = view_250 = permute_126 = None
        view_251: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_69, [64, 512, 256]);  addmm_69 = None
        view_252: "f32[64, 512, 4, 64]" = torch.ops.aten.reshape.default(view_251, [64, 512, -1, 64]);  view_251 = None
        permute_127: "f32[64, 4, 512, 64]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_50: "f32[64, 4, 512, 64]" = torch.ops.aten.expand.default(permute_127, [64, 4, 512, 64]);  permute_127 = None
        clone_69: "f32[64, 4, 512, 64]" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_257: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_69, [256, 512, 64]);  clone_69 = None
        bmm_23: "f32[256, 512, 64]" = torch.ops.aten.bmm.default(view_256, view_257);  view_256 = view_257 = None
        view_258: "f32[64, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_23, [64, 4, 512, 64]);  bmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_129: "f32[64, 512, 4, 64]" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
        clone_70: "f32[64, 512, 4, 64]" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_259: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(clone_70, [64, 512, -1]);  clone_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_260: "f32[32768, 256]" = torch.ops.aten.reshape.default(view_259, [32768, 256]);  view_259 = None
        permute_130: "f32[256, 256]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        addmm_70: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg194_1, view_260, permute_130);  arg194_1 = view_260 = permute_130 = None
        view_261: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_70, [64, 512, 256]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_95: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(view_261, add_93);  view_261 = add_93 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_95, [2], correction = 0, keepdim = True)
        getitem_46: "f32[64, 512, 1]" = var_mean_23[0]
        getitem_47: "f32[64, 512, 1]" = var_mean_23[1];  var_mean_23 = None
        sub_35: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_95, getitem_47);  add_95 = getitem_47 = None
        add_96: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        mul_103: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
        mul_104: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_103, arg195_1);  mul_103 = arg195_1 = None
        add_97: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_104, arg196_1);  mul_104 = arg196_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_262: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_97, [32768, 256])
        permute_131: "f32[256, 1024]" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        addmm_71: "f32[32768, 1024]" = torch.ops.aten.addmm.default(arg198_1, view_262, permute_131);  arg198_1 = view_262 = permute_131 = None
        view_263: "f32[64, 512, 1024]" = torch.ops.aten.reshape.default(addmm_71, [64, 512, 1024]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_105: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_263, 0.5)
        mul_106: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(view_263, 0.7071067811865476);  view_263 = None
        erf_11: "f32[64, 512, 1024]" = torch.ops.aten.erf.default(mul_106);  mul_106 = None
        add_98: "f32[64, 512, 1024]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_107: "f32[64, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_105, add_98);  mul_105 = add_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_264: "f32[32768, 1024]" = torch.ops.aten.reshape.default(mul_107, [32768, 1024]);  mul_107 = None
        permute_132: "f32[1024, 256]" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
        addmm_72: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg200_1, view_264, permute_132);  arg200_1 = view_264 = permute_132 = None
        view_265: "f32[64, 512, 256]" = torch.ops.aten.reshape.default(addmm_72, [64, 512, 256]);  addmm_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_99: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(view_265, add_97);  view_265 = add_97 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_99, [2], correction = 0, keepdim = True)
        getitem_48: "f32[64, 512, 1]" = var_mean_24[0]
        getitem_49: "f32[64, 512, 1]" = var_mean_24[1];  var_mean_24 = None
        sub_36: "f32[64, 512, 256]" = torch.ops.aten.sub.Tensor(add_99, getitem_49);  add_99 = getitem_49 = None
        add_100: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        mul_108: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = rsqrt_24 = None
        mul_109: "f32[64, 512, 256]" = torch.ops.aten.mul.Tensor(mul_108, arg201_1);  mul_108 = arg201_1 = None
        add_101: "f32[64, 512, 256]" = torch.ops.aten.add.Tensor(mul_109, arg202_1);  mul_109 = arg202_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:499 in forward, code: hidden_states = self.dense(generator_hidden_states)
        view_266: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_101, [32768, 256]);  add_101 = None
        permute_133: "f32[256, 128]" = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
        addmm_73: "f32[32768, 128]" = torch.ops.aten.addmm.default(arg204_1, view_266, permute_133);  arg204_1 = view_266 = permute_133 = None
        view_267: "f32[64, 512, 128]" = torch.ops.aten.reshape.default(addmm_73, [64, 512, 128]);  addmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_110: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(view_267, 0.5)
        mul_111: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(view_267, 0.7071067811865476);  view_267 = None
        erf_12: "f32[64, 512, 128]" = torch.ops.aten.erf.default(mul_111);  mul_111 = None
        add_102: "f32[64, 512, 128]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_112: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_110, add_102);  mul_110 = add_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:501 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        var_mean_25 = torch.ops.aten.var_mean.correction(mul_112, [2], correction = 0, keepdim = True)
        getitem_50: "f32[64, 512, 1]" = var_mean_25[0]
        getitem_51: "f32[64, 512, 1]" = var_mean_25[1];  var_mean_25 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "i64[64, 513]" = torch.ops.aten.constant_pad_nd.default(arg0_1, [0, 1], -100.0);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:61 in ForCausalLMLoss, code: shift_labels = labels[..., 1:].contiguous()
        slice_1: "i64[64, 512]" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807);  constant_pad_nd = None
        clone_73: "i64[64, 512]" = torch.ops.aten.clone.default(slice_1, memory_format = torch.contiguous_format);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:65 in ForCausalLMLoss, code: shift_labels = shift_labels.view(-1)
        view_271: "i64[32768]" = torch.ops.aten.reshape.default(clone_73, [-1]);  clone_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        ne_1: "b8[32768]" = torch.ops.aten.ne.Scalar(view_271, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:1351 in forward, code: logits = self.generator_lm_head(self.generator_predictions(hidden_states[:, slice_indices, :]))
        full_default_38: "f32[2]" = torch.ops.aten.full.default([2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default: "f32[30524]" = torch.ops.aten.cat.default([arg207_1, full_default_38]);  arg207_1 = full_default_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:501 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        sub_37: "f32[64, 512, 128]" = torch.ops.aten.sub.Tensor(mul_112, getitem_51);  mul_112 = getitem_51 = None
        add_103: "f32[64, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_25: "f32[64, 512, 1]" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
        mul_113: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_25);  sub_37 = rsqrt_25 = None
        mul_114: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_113, arg205_1);  mul_113 = arg205_1 = None
        add_104: "f32[64, 512, 128]" = torch.ops.aten.add.Tensor(mul_114, arg206_1);  mul_114 = arg206_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:1351 in forward, code: logits = self.generator_lm_head(self.generator_predictions(hidden_states[:, slice_indices, :]))
        view_268: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_104, [32768, 128]);  add_104 = None
        permute_134: "f32[128, 30522]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        constant_pad_nd_default: "f32[128, 30524]" = torch.ops.aten.constant_pad_nd.default(permute_134, [0, 2, 0, 0]);  permute_134 = None
        addmm_default: "f32[32768, 30524]" = torch.ops.aten.addmm.default(cat_default, view_268, constant_pad_nd_default);  cat_default = view_268 = constant_pad_nd_default = None
        slice_tensor: "f32[32768, 30522]" = torch.ops.aten.slice.Tensor(addmm_default, 1, 0, -2);  addmm_default = None
        view_269: "f32[64, 512, 30522]" = torch.ops.aten.reshape.default(slice_tensor, [64, 512, 30522]);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_270: "f32[32768, 30522]" = torch.ops.aten.reshape.default(view_269, [-1, 30522])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        amax_12: "f32[32768, 1]" = torch.ops.aten.amax.default(view_270, [1], True)
        sub_38: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(view_270, amax_12);  view_270 = amax_12 = None
        exp_12: "f32[32768, 30522]" = torch.ops.aten.exp.default(sub_38)
        sum_13: "f32[32768, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
        log: "f32[32768, 1]" = torch.ops.aten.log.default(sum_13);  sum_13 = None
        sub_39: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(sub_38, log);  sub_38 = log = None
        ne: "b8[32768]" = torch.ops.aten.ne.Scalar(view_271, -100)
        full_default_36: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_24: "i64[32768]" = torch.ops.aten.where.self(ne, view_271, full_default_36);  ne = full_default_36 = None
        unsqueeze_3: "i64[32768, 1]" = torch.ops.aten.unsqueeze.default(where_24, 1);  where_24 = None
        gather_1: "f32[32768, 1]" = torch.ops.aten.gather.default(sub_39, 1, unsqueeze_3);  sub_39 = unsqueeze_3 = None
        squeeze: "f32[32768]" = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
        neg: "f32[32768]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_37: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_25: "f32[32768]" = torch.ops.aten.where.self(ne_1, neg, full_default_37);  ne_1 = neg = full_default_37 = None
        sum_15: "f32[]" = torch.ops.aten.sum.default(where_25);  where_25 = None
        ne_2: "b8[32768]" = torch.ops.aten.ne.Scalar(view_271, -100);  view_271 = None
        sum_14: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
        div_12: "f32[]" = torch.ops.aten.div.Tensor(sum_15, convert_element_type);  sum_15 = convert_element_type = None
        return (div_12, view_269)
