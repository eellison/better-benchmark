import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[32, 512]", arg1_1: "f32[30522, 768]", arg2_1: "i64[1, 512]", arg3_1: "f32[512, 768]", arg4_1: "f32[1024, 768]", arg5_1: "f32[1024, 768]", arg6_1: "f32[1024, 768]", arg7_1: "f32[1024, 768]", arg8_1: "f32[2, 768]", arg9_1: "f32[768]", arg10_1: "f32[768]", arg11_1: "f32[768, 768]", arg12_1: "f32[768]", arg13_1: "f32[768, 768]", arg14_1: "f32[768]", arg15_1: "f32[768, 768]", arg16_1: "f32[768]", arg17_1: "f32[768, 768]", arg18_1: "f32[768]", arg19_1: "f32[768]", arg20_1: "f32[768]", arg21_1: "f32[3072, 768]", arg22_1: "f32[3072]", arg23_1: "f32[768, 3072]", arg24_1: "f32[768]", arg25_1: "f32[768]", arg26_1: "f32[768]", arg27_1: "f32[768, 768]", arg28_1: "f32[768]", arg29_1: "f32[768, 768]", arg30_1: "f32[768]", arg31_1: "f32[768, 768]", arg32_1: "f32[768]", arg33_1: "f32[768, 768]", arg34_1: "f32[768]", arg35_1: "f32[768]", arg36_1: "f32[768]", arg37_1: "f32[3072, 768]", arg38_1: "f32[3072]", arg39_1: "f32[768, 3072]", arg40_1: "f32[768]", arg41_1: "f32[768]", arg42_1: "f32[768]", arg43_1: "f32[768, 768]", arg44_1: "f32[768]", arg45_1: "f32[768, 768]", arg46_1: "f32[768]", arg47_1: "f32[768, 768]", arg48_1: "f32[768]", arg49_1: "f32[768, 768]", arg50_1: "f32[768]", arg51_1: "f32[768]", arg52_1: "f32[768]", arg53_1: "f32[3072, 768]", arg54_1: "f32[3072]", arg55_1: "f32[768, 3072]", arg56_1: "f32[768]", arg57_1: "f32[768]", arg58_1: "f32[768]", arg59_1: "f32[768, 768]", arg60_1: "f32[768]", arg61_1: "f32[768, 768]", arg62_1: "f32[768]", arg63_1: "f32[768, 768]", arg64_1: "f32[768]", arg65_1: "f32[768, 768]", arg66_1: "f32[768]", arg67_1: "f32[768]", arg68_1: "f32[768]", arg69_1: "f32[3072, 768]", arg70_1: "f32[3072]", arg71_1: "f32[768, 3072]", arg72_1: "f32[768]", arg73_1: "f32[768]", arg74_1: "f32[768]", arg75_1: "f32[768, 768]", arg76_1: "f32[768]", arg77_1: "f32[768, 768]", arg78_1: "f32[768]", arg79_1: "f32[768, 768]", arg80_1: "f32[768]", arg81_1: "f32[768, 768]", arg82_1: "f32[768]", arg83_1: "f32[768]", arg84_1: "f32[768]", arg85_1: "f32[3072, 768]", arg86_1: "f32[3072]", arg87_1: "f32[768, 3072]", arg88_1: "f32[768]", arg89_1: "f32[768]", arg90_1: "f32[768]", arg91_1: "f32[768, 768]", arg92_1: "f32[768]", arg93_1: "f32[768, 768]", arg94_1: "f32[768]", arg95_1: "f32[768, 768]", arg96_1: "f32[768]", arg97_1: "f32[768, 768]", arg98_1: "f32[768]", arg99_1: "f32[768]", arg100_1: "f32[768]", arg101_1: "f32[3072, 768]", arg102_1: "f32[3072]", arg103_1: "f32[768, 3072]", arg104_1: "f32[768]", arg105_1: "f32[768]", arg106_1: "f32[768]", arg107_1: "f32[768, 768]", arg108_1: "f32[768]", arg109_1: "f32[768, 768]", arg110_1: "f32[768]", arg111_1: "f32[768, 768]", arg112_1: "f32[768]", arg113_1: "f32[768, 768]", arg114_1: "f32[768]", arg115_1: "f32[768]", arg116_1: "f32[768]", arg117_1: "f32[3072, 768]", arg118_1: "f32[3072]", arg119_1: "f32[768, 3072]", arg120_1: "f32[768]", arg121_1: "f32[768]", arg122_1: "f32[768]", arg123_1: "f32[768, 768]", arg124_1: "f32[768]", arg125_1: "f32[768, 768]", arg126_1: "f32[768]", arg127_1: "f32[768, 768]", arg128_1: "f32[768]", arg129_1: "f32[768, 768]", arg130_1: "f32[768]", arg131_1: "f32[768]", arg132_1: "f32[768]", arg133_1: "f32[3072, 768]", arg134_1: "f32[3072]", arg135_1: "f32[768, 3072]", arg136_1: "f32[768]", arg137_1: "f32[768]", arg138_1: "f32[768]", arg139_1: "f32[768, 768]", arg140_1: "f32[768]", arg141_1: "f32[768, 768]", arg142_1: "f32[768]", arg143_1: "f32[768, 768]", arg144_1: "f32[768]", arg145_1: "f32[768, 768]", arg146_1: "f32[768]", arg147_1: "f32[768]", arg148_1: "f32[768]", arg149_1: "f32[3072, 768]", arg150_1: "f32[3072]", arg151_1: "f32[768, 3072]", arg152_1: "f32[768]", arg153_1: "f32[768]", arg154_1: "f32[768]", arg155_1: "f32[768, 768]", arg156_1: "f32[768]", arg157_1: "f32[768, 768]", arg158_1: "f32[768]", arg159_1: "f32[768, 768]", arg160_1: "f32[768]", arg161_1: "f32[768, 768]", arg162_1: "f32[768]", arg163_1: "f32[768]", arg164_1: "f32[768]", arg165_1: "f32[3072, 768]", arg166_1: "f32[3072]", arg167_1: "f32[768, 3072]", arg168_1: "f32[768]", arg169_1: "f32[768]", arg170_1: "f32[768]", arg171_1: "f32[768, 768]", arg172_1: "f32[768]", arg173_1: "f32[768, 768]", arg174_1: "f32[768]", arg175_1: "f32[768, 768]", arg176_1: "f32[768]", arg177_1: "f32[768, 768]", arg178_1: "f32[768]", arg179_1: "f32[768]", arg180_1: "f32[768]", arg181_1: "f32[3072, 768]", arg182_1: "f32[3072]", arg183_1: "f32[768, 3072]", arg184_1: "f32[768]", arg185_1: "f32[768]", arg186_1: "f32[768]", arg187_1: "f32[768, 768]", arg188_1: "f32[768]", arg189_1: "f32[768, 768]", arg190_1: "f32[768]", arg191_1: "f32[768, 768]", arg192_1: "f32[768]", arg193_1: "f32[768, 768]", arg194_1: "f32[768]", arg195_1: "f32[768]", arg196_1: "f32[768]", arg197_1: "f32[3072, 768]", arg198_1: "f32[3072]", arg199_1: "f32[768, 3072]", arg200_1: "f32[768]", arg201_1: "f32[768]", arg202_1: "f32[768]", arg203_1: "f32[768, 768]", arg204_1: "f32[768]", arg205_1: "f32[768, 768]", arg206_1: "f32[768]", arg207_1: "f32[768]", arg208_1: "f32[768]", arg209_1: "f32[30522]", arg210_1: "i64[32, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:496 in forward, code: attention_mask = torch.ones(input_shape, device=device)
        full: "f32[32, 512]" = torch.ops.aten.full.default([32, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:503 in forward, code: extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        unsqueeze: "f32[32, 1, 512]" = torch.ops.aten.unsqueeze.default(full, 1);  full = None
        unsqueeze_1: "f32[32, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:506 in forward, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min
        sub: "f32[32, 1, 1, 512]" = torch.ops.aten.sub.Tensor(1.0, unsqueeze_1);  unsqueeze_1 = sub = None
        full_default_2: "f32[32, 1, 1, 512]" = torch.ops.aten.full.default([32, 1, 1, 512], -0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:92 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 0);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:95 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_1: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg3_1, arg2_1);  arg3_1 = arg2_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:109 in forward, code: words_embeddings
        add: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:501 in forward, code: bbox = torch.zeros(input_shape + (4,), dtype=torch.long, device=device)
        full_default_1: "i64[32, 512, 4]" = torch.ops.aten.full.default([32, 512, 4], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:97 in forward, code: left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
        select: "i64[32, 512]" = torch.ops.aten.select.int(full_default_1, 2, 0)
        embedding_2: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg4_1, select);  select = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:109 in forward, code: words_embeddings
        add_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:98 in forward, code: upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
        select_1: "i64[32, 512]" = torch.ops.aten.select.int(full_default_1, 2, 1)
        embedding_3: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg5_1, select_1);  select_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:109 in forward, code: words_embeddings
        add_2: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_1, embedding_3);  add_1 = embedding_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:99 in forward, code: right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
        select_2: "i64[32, 512]" = torch.ops.aten.select.int(full_default_1, 2, 2)
        embedding_4: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg4_1, select_2);  arg4_1 = select_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:109 in forward, code: words_embeddings
        add_3: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_2, embedding_4);  add_2 = embedding_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:100 in forward, code: lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        select_3: "i64[32, 512]" = torch.ops.aten.select.int(full_default_1, 2, 3)
        embedding_5: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg5_1, select_3);  arg5_1 = select_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:109 in forward, code: words_embeddings
        add_4: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_3, embedding_5);  add_3 = embedding_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:104 in forward, code: h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        select_4: "i64[32, 512]" = torch.ops.aten.select.int(full_default_1, 2, 3)
        select_5: "i64[32, 512]" = torch.ops.aten.select.int(full_default_1, 2, 1)
        sub_1: "i64[32, 512]" = torch.ops.aten.sub.Tensor(select_4, select_5);  select_4 = select_5 = None
        embedding_6: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg6_1, sub_1);  arg6_1 = sub_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:109 in forward, code: words_embeddings
        add_5: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_4, embedding_6);  add_4 = embedding_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:105 in forward, code: w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])
        select_6: "i64[32, 512]" = torch.ops.aten.select.int(full_default_1, 2, 2)
        select_7: "i64[32, 512]" = torch.ops.aten.select.int(full_default_1, 2, 0);  full_default_1 = None
        sub_2: "i64[32, 512]" = torch.ops.aten.sub.Tensor(select_6, select_7);  select_6 = select_7 = None
        embedding_7: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg7_1, sub_2);  arg7_1 = sub_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:109 in forward, code: words_embeddings
        add_6: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_5, embedding_7);  add_5 = embedding_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:498 in forward, code: token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        full_default: "i64[32, 512]" = torch.ops.aten.full.default([32, 512], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:106 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_8: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg8_1, full_default);  arg8_1 = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:109 in forward, code: words_embeddings
        add_7: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_6, embedding_8);  add_6 = embedding_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:119 in forward, code: embeddings = self.LayerNorm(embeddings)
        var_mean = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean[0]
        getitem_1: "f32[32, 512, 1]" = var_mean[1];  var_mean = None
        sub_3: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_7, getitem_1);  add_7 = getitem_1 = None
        add_8: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt);  sub_3 = rsqrt = None
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_1, arg9_1);  mul_1 = arg9_1 = None
        add_9: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_2, arg10_1);  mul_2 = arg10_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        view: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_9, [16384, 768])
        permute: "f32[768, 768]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg12_1, view, permute);  arg12_1 = view = permute = None
        view_1: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm, [32, 512, 768]);  addmm = None
        view_2: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_1, [32, 512, -1, 64]);  view_1 = None
        permute_1: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None

        # No stacktrace found for following nodes
        clone_default_33: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        view_3: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_9, [16384, 768])
        permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_1: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg14_1, view_3, permute_2);  arg14_1 = view_3 = permute_2 = None
        view_4: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_1, [32, 512, 768]);  addmm_1 = None
        view_5: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_4, [32, 512, -1, 64]);  view_4 = None
        permute_3: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None

        # No stacktrace found for following nodes
        clone_default_34: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        view_6: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_9, [16384, 768])
        permute_4: "f32[768, 768]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_2: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg16_1, view_6, permute_4);  arg16_1 = view_6 = permute_4 = None
        view_7: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_2, [32, 512, 768]);  addmm_2 = None
        view_8: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_7, [32, 512, -1, 64]);  view_7 = None
        permute_5: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # No stacktrace found for following nodes
        clone_default_35: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        _scaled_dot_product_efficient_attention_default_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_33, clone_default_34, clone_default_35, None, False, scale = 0.125);  clone_default_33 = clone_default_34 = clone_default_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_63: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_11[0];  _scaled_dot_product_efficient_attention_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_63, [0, 2, 1, 3]);  getitem_63 = None
        clone_5: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_15: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_5, [32, 512, -1]);  clone_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_16: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_15, [16384, 768]);  view_15 = None
        permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        addmm_3: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg18_1, view_16, permute_8);  arg18_1 = view_16 = permute_8 = None
        view_17: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_3, [32, 512, 768]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_11: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_17, add_9);  view_17 = add_9 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_2: "f32[32, 512, 1]" = var_mean_1[0]
        getitem_3: "f32[32, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_5: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_3);  add_11 = getitem_3 = None
        add_12: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_1);  sub_5 = rsqrt_1 = None
        mul_5: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_4, arg19_1);  mul_4 = arg19_1 = None
        add_13: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_5, arg20_1);  mul_5 = arg20_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_18: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_13, [16384, 768])
        permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_4: "f32[16384, 3072]" = torch.ops.aten.addmm.default(arg22_1, view_18, permute_9);  arg22_1 = view_18 = permute_9 = None
        view_19: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_4, [32, 512, 3072]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_6: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.5)
        mul_7: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476);  view_19 = None
        erf: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
        add_14: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_8: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_6, add_14);  mul_6 = add_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_20: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_8, [16384, 3072]);  mul_8 = None
        permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_5: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg24_1, view_20, permute_10);  arg24_1 = view_20 = permute_10 = None
        view_21: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_5, [32, 512, 768]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_15: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_21, add_13);  view_21 = add_13 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
        getitem_4: "f32[32, 512, 1]" = var_mean_2[0]
        getitem_5: "f32[32, 512, 1]" = var_mean_2[1];  var_mean_2 = None
        sub_6: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_15, getitem_5);  add_15 = getitem_5 = None
        add_16: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        mul_9: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_2);  sub_6 = rsqrt_2 = None
        mul_10: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_9, arg25_1);  mul_9 = arg25_1 = None
        add_17: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_10, arg26_1);  mul_10 = arg26_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        view_22: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_17, [16384, 768])
        permute_11: "f32[768, 768]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_6: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg28_1, view_22, permute_11);  arg28_1 = view_22 = permute_11 = None
        view_23: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_6, [32, 512, 768]);  addmm_6 = None
        view_24: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_23, [32, 512, -1, 64]);  view_23 = None
        permute_12: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None

        # No stacktrace found for following nodes
        clone_default_30: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        view_25: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_17, [16384, 768])
        permute_13: "f32[768, 768]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        addmm_7: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg30_1, view_25, permute_13);  arg30_1 = view_25 = permute_13 = None
        view_26: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_7, [32, 512, 768]);  addmm_7 = None
        view_27: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_26, [32, 512, -1, 64]);  view_26 = None
        permute_14: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None

        # No stacktrace found for following nodes
        clone_default_31: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        view_28: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_17, [16384, 768])
        permute_15: "f32[768, 768]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_8: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg32_1, view_28, permute_15);  arg32_1 = view_28 = permute_15 = None
        view_29: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_8, [32, 512, 768]);  addmm_8 = None
        view_30: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_29, [32, 512, -1, 64]);  view_29 = None
        permute_16: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None

        # No stacktrace found for following nodes
        clone_default_32: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        _scaled_dot_product_efficient_attention_default_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_30, clone_default_31, clone_default_32, None, False, scale = 0.125);  clone_default_30 = clone_default_31 = clone_default_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_62: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_10[0];  _scaled_dot_product_efficient_attention_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_18: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_62, [0, 2, 1, 3]);  getitem_62 = None
        clone_12: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_37: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_12, [32, 512, -1]);  clone_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_38: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_37, [16384, 768]);  view_37 = None
        permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        addmm_9: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg34_1, view_38, permute_19);  arg34_1 = view_38 = permute_19 = None
        view_39: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_9, [32, 512, 768]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_19: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_39, add_17);  view_39 = add_17 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
        getitem_6: "f32[32, 512, 1]" = var_mean_3[0]
        getitem_7: "f32[32, 512, 1]" = var_mean_3[1];  var_mean_3 = None
        sub_8: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_7);  add_19 = getitem_7 = None
        add_20: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_12: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_3);  sub_8 = rsqrt_3 = None
        mul_13: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_12, arg35_1);  mul_12 = arg35_1 = None
        add_21: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_13, arg36_1);  mul_13 = arg36_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_40: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_21, [16384, 768])
        permute_20: "f32[768, 3072]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_10: "f32[16384, 3072]" = torch.ops.aten.addmm.default(arg38_1, view_40, permute_20);  arg38_1 = view_40 = permute_20 = None
        view_41: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_10, [32, 512, 3072]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_14: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.5)
        mul_15: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476);  view_41 = None
        erf_1: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
        add_22: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_16: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_14, add_22);  mul_14 = add_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_42: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_16, [16384, 3072]);  mul_16 = None
        permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_11: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg40_1, view_42, permute_21);  arg40_1 = view_42 = permute_21 = None
        view_43: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_11, [32, 512, 768]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_23: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_43, add_21);  view_43 = add_21 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_23, [2], correction = 0, keepdim = True)
        getitem_8: "f32[32, 512, 1]" = var_mean_4[0]
        getitem_9: "f32[32, 512, 1]" = var_mean_4[1];  var_mean_4 = None
        sub_9: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_23, getitem_9);  add_23 = getitem_9 = None
        add_24: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        mul_17: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_4);  sub_9 = rsqrt_4 = None
        mul_18: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_17, arg41_1);  mul_17 = arg41_1 = None
        add_25: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_18, arg42_1);  mul_18 = arg42_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        view_44: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_25, [16384, 768])
        permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_12: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg44_1, view_44, permute_22);  arg44_1 = view_44 = permute_22 = None
        view_45: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_12, [32, 512, 768]);  addmm_12 = None
        view_46: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_45, [32, 512, -1, 64]);  view_45 = None
        permute_23: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None

        # No stacktrace found for following nodes
        clone_default_27: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        view_47: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_25, [16384, 768])
        permute_24: "f32[768, 768]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        addmm_13: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg46_1, view_47, permute_24);  arg46_1 = view_47 = permute_24 = None
        view_48: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_13, [32, 512, 768]);  addmm_13 = None
        view_49: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_48, [32, 512, -1, 64]);  view_48 = None
        permute_25: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None

        # No stacktrace found for following nodes
        clone_default_28: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_25, memory_format = torch.contiguous_format);  permute_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        view_50: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_25, [16384, 768])
        permute_26: "f32[768, 768]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_14: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg48_1, view_50, permute_26);  arg48_1 = view_50 = permute_26 = None
        view_51: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_14, [32, 512, 768]);  addmm_14 = None
        view_52: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_51, [32, 512, -1, 64]);  view_51 = None
        permute_27: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None

        # No stacktrace found for following nodes
        clone_default_29: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
        _scaled_dot_product_efficient_attention_default_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_27, clone_default_28, clone_default_29, None, False, scale = 0.125);  clone_default_27 = clone_default_28 = clone_default_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_61: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_9[0];  _scaled_dot_product_efficient_attention_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_29: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_61, [0, 2, 1, 3]);  getitem_61 = None
        clone_19: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_59: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_19, [32, 512, -1]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_60: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_59, [16384, 768]);  view_59 = None
        permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_15: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg50_1, view_60, permute_30);  arg50_1 = view_60 = permute_30 = None
        view_61: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_15, [32, 512, 768]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_27: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_61, add_25);  view_61 = add_25 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_10: "f32[32, 512, 1]" = var_mean_5[0]
        getitem_11: "f32[32, 512, 1]" = var_mean_5[1];  var_mean_5 = None
        sub_11: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_11);  add_27 = getitem_11 = None
        add_28: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_20: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_5);  sub_11 = rsqrt_5 = None
        mul_21: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_20, arg51_1);  mul_20 = arg51_1 = None
        add_29: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_21, arg52_1);  mul_21 = arg52_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_62: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_29, [16384, 768])
        permute_31: "f32[768, 3072]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_16: "f32[16384, 3072]" = torch.ops.aten.addmm.default(arg54_1, view_62, permute_31);  arg54_1 = view_62 = permute_31 = None
        view_63: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_16, [32, 512, 3072]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_22: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.5)
        mul_23: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476);  view_63 = None
        erf_2: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_23);  mul_23 = None
        add_30: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_24: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_22, add_30);  mul_22 = add_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_64: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_24, [16384, 3072]);  mul_24 = None
        permute_32: "f32[3072, 768]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_17: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg56_1, view_64, permute_32);  arg56_1 = view_64 = permute_32 = None
        view_65: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_17, [32, 512, 768]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_31: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_65, add_29);  view_65 = add_29 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
        getitem_12: "f32[32, 512, 1]" = var_mean_6[0]
        getitem_13: "f32[32, 512, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_12: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_31, getitem_13);  add_31 = getitem_13 = None
        add_32: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        mul_25: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_6);  sub_12 = rsqrt_6 = None
        mul_26: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_25, arg57_1);  mul_25 = arg57_1 = None
        add_33: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_26, arg58_1);  mul_26 = arg58_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        view_66: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_33, [16384, 768])
        permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_18: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg60_1, view_66, permute_33);  arg60_1 = view_66 = permute_33 = None
        view_67: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_18, [32, 512, 768]);  addmm_18 = None
        view_68: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_67, [32, 512, -1, 64]);  view_67 = None
        permute_34: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None

        # No stacktrace found for following nodes
        clone_default_24: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        view_69: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_33, [16384, 768])
        permute_35: "f32[768, 768]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_19: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg62_1, view_69, permute_35);  arg62_1 = view_69 = permute_35 = None
        view_70: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_19, [32, 512, 768]);  addmm_19 = None
        view_71: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_70, [32, 512, -1, 64]);  view_70 = None
        permute_36: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None

        # No stacktrace found for following nodes
        clone_default_25: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_36, memory_format = torch.contiguous_format);  permute_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        view_72: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_33, [16384, 768])
        permute_37: "f32[768, 768]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_20: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg64_1, view_72, permute_37);  arg64_1 = view_72 = permute_37 = None
        view_73: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_20, [32, 512, 768]);  addmm_20 = None
        view_74: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_73, [32, 512, -1, 64]);  view_73 = None
        permute_38: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None

        # No stacktrace found for following nodes
        clone_default_26: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
        _scaled_dot_product_efficient_attention_default_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_24, clone_default_25, clone_default_26, None, False, scale = 0.125);  clone_default_24 = clone_default_25 = clone_default_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_60: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_8[0];  _scaled_dot_product_efficient_attention_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_40: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None
        clone_26: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_81: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_26, [32, 512, -1]);  clone_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_82: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_81, [16384, 768]);  view_81 = None
        permute_41: "f32[768, 768]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        addmm_21: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg66_1, view_82, permute_41);  arg66_1 = view_82 = permute_41 = None
        view_83: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_21, [32, 512, 768]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_35: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_83, add_33);  view_83 = add_33 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_14: "f32[32, 512, 1]" = var_mean_7[0]
        getitem_15: "f32[32, 512, 1]" = var_mean_7[1];  var_mean_7 = None
        sub_14: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_15);  add_35 = getitem_15 = None
        add_36: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        mul_28: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_7);  sub_14 = rsqrt_7 = None
        mul_29: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_28, arg67_1);  mul_28 = arg67_1 = None
        add_37: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_29, arg68_1);  mul_29 = arg68_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_84: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_37, [16384, 768])
        permute_42: "f32[768, 3072]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        addmm_22: "f32[16384, 3072]" = torch.ops.aten.addmm.default(arg70_1, view_84, permute_42);  arg70_1 = view_84 = permute_42 = None
        view_85: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_22, [32, 512, 3072]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_30: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.5)
        mul_31: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476);  view_85 = None
        erf_3: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_31);  mul_31 = None
        add_38: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_32: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_30, add_38);  mul_30 = add_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_86: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_32, [16384, 3072]);  mul_32 = None
        permute_43: "f32[3072, 768]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_23: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg72_1, view_86, permute_43);  arg72_1 = view_86 = permute_43 = None
        view_87: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_23, [32, 512, 768]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_39: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_87, add_37);  view_87 = add_37 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
        getitem_16: "f32[32, 512, 1]" = var_mean_8[0]
        getitem_17: "f32[32, 512, 1]" = var_mean_8[1];  var_mean_8 = None
        sub_15: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_39, getitem_17);  add_39 = getitem_17 = None
        add_40: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        mul_33: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_8);  sub_15 = rsqrt_8 = None
        mul_34: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_33, arg73_1);  mul_33 = arg73_1 = None
        add_41: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_34, arg74_1);  mul_34 = arg74_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        view_88: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_41, [16384, 768])
        permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_24: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg76_1, view_88, permute_44);  arg76_1 = view_88 = permute_44 = None
        view_89: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_24, [32, 512, 768]);  addmm_24 = None
        view_90: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_89, [32, 512, -1, 64]);  view_89 = None
        permute_45: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None

        # No stacktrace found for following nodes
        clone_default_21: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        view_91: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_41, [16384, 768])
        permute_46: "f32[768, 768]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        addmm_25: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg78_1, view_91, permute_46);  arg78_1 = view_91 = permute_46 = None
        view_92: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_25, [32, 512, 768]);  addmm_25 = None
        view_93: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_92, [32, 512, -1, 64]);  view_92 = None
        permute_47: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None

        # No stacktrace found for following nodes
        clone_default_22: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        view_94: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_41, [16384, 768])
        permute_48: "f32[768, 768]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_26: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg80_1, view_94, permute_48);  arg80_1 = view_94 = permute_48 = None
        view_95: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_26, [32, 512, 768]);  addmm_26 = None
        view_96: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_95, [32, 512, -1, 64]);  view_95 = None
        permute_49: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None

        # No stacktrace found for following nodes
        clone_default_23: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        _scaled_dot_product_efficient_attention_default_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_21, clone_default_22, clone_default_23, None, False, scale = 0.125);  clone_default_21 = clone_default_22 = clone_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_59: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_7[0];  _scaled_dot_product_efficient_attention_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_51: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_59, [0, 2, 1, 3]);  getitem_59 = None
        clone_33: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_103: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_33, [32, 512, -1]);  clone_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_104: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_103, [16384, 768]);  view_103 = None
        permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        addmm_27: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg82_1, view_104, permute_52);  arg82_1 = view_104 = permute_52 = None
        view_105: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_27, [32, 512, 768]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_43: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_105, add_41);  view_105 = add_41 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_18: "f32[32, 512, 1]" = var_mean_9[0]
        getitem_19: "f32[32, 512, 1]" = var_mean_9[1];  var_mean_9 = None
        sub_17: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_19);  add_43 = getitem_19 = None
        add_44: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        mul_36: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_9);  sub_17 = rsqrt_9 = None
        mul_37: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_36, arg83_1);  mul_36 = arg83_1 = None
        add_45: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_37, arg84_1);  mul_37 = arg84_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_106: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_45, [16384, 768])
        permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_28: "f32[16384, 3072]" = torch.ops.aten.addmm.default(arg86_1, view_106, permute_53);  arg86_1 = view_106 = permute_53 = None
        view_107: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_28, [32, 512, 3072]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_38: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
        mul_39: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476);  view_107 = None
        erf_4: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
        add_46: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_40: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_38, add_46);  mul_38 = add_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_108: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_40, [16384, 3072]);  mul_40 = None
        permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_29: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg88_1, view_108, permute_54);  arg88_1 = view_108 = permute_54 = None
        view_109: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_29, [32, 512, 768]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_47: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_109, add_45);  view_109 = add_45 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_47, [2], correction = 0, keepdim = True)
        getitem_20: "f32[32, 512, 1]" = var_mean_10[0]
        getitem_21: "f32[32, 512, 1]" = var_mean_10[1];  var_mean_10 = None
        sub_18: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_47, getitem_21);  add_47 = getitem_21 = None
        add_48: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        mul_41: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_10);  sub_18 = rsqrt_10 = None
        mul_42: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_41, arg89_1);  mul_41 = arg89_1 = None
        add_49: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_42, arg90_1);  mul_42 = arg90_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        view_110: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_49, [16384, 768])
        permute_55: "f32[768, 768]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_30: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg92_1, view_110, permute_55);  arg92_1 = view_110 = permute_55 = None
        view_111: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_30, [32, 512, 768]);  addmm_30 = None
        view_112: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_111, [32, 512, -1, 64]);  view_111 = None
        permute_56: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None

        # No stacktrace found for following nodes
        clone_default_18: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        view_113: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_49, [16384, 768])
        permute_57: "f32[768, 768]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        addmm_31: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg94_1, view_113, permute_57);  arg94_1 = view_113 = permute_57 = None
        view_114: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_31, [32, 512, 768]);  addmm_31 = None
        view_115: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_114, [32, 512, -1, 64]);  view_114 = None
        permute_58: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_115, [0, 2, 1, 3]);  view_115 = None

        # No stacktrace found for following nodes
        clone_default_19: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_58, memory_format = torch.contiguous_format);  permute_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        view_116: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_49, [16384, 768])
        permute_59: "f32[768, 768]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_32: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg96_1, view_116, permute_59);  arg96_1 = view_116 = permute_59 = None
        view_117: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_32, [32, 512, 768]);  addmm_32 = None
        view_118: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_117, [32, 512, -1, 64]);  view_117 = None
        permute_60: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None

        # No stacktrace found for following nodes
        clone_default_20: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
        _scaled_dot_product_efficient_attention_default_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_18, clone_default_19, clone_default_20, None, False, scale = 0.125);  clone_default_18 = clone_default_19 = clone_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_58: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_6[0];  _scaled_dot_product_efficient_attention_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_62: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_58, [0, 2, 1, 3]);  getitem_58 = None
        clone_40: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_125: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_40, [32, 512, -1]);  clone_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_126: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_125, [16384, 768]);  view_125 = None
        permute_63: "f32[768, 768]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_33: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg98_1, view_126, permute_63);  arg98_1 = view_126 = permute_63 = None
        view_127: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_33, [32, 512, 768]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_51: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_127, add_49);  view_127 = add_49 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
        getitem_22: "f32[32, 512, 1]" = var_mean_11[0]
        getitem_23: "f32[32, 512, 1]" = var_mean_11[1];  var_mean_11 = None
        sub_20: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_51, getitem_23);  add_51 = getitem_23 = None
        add_52: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_44: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_11);  sub_20 = rsqrt_11 = None
        mul_45: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_44, arg99_1);  mul_44 = arg99_1 = None
        add_53: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_45, arg100_1);  mul_45 = arg100_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_128: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_53, [16384, 768])
        permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_34: "f32[16384, 3072]" = torch.ops.aten.addmm.default(arg102_1, view_128, permute_64);  arg102_1 = view_128 = permute_64 = None
        view_129: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_34, [32, 512, 3072]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_46: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.5)
        mul_47: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476);  view_129 = None
        erf_5: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
        add_54: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_48: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_46, add_54);  mul_46 = add_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_130: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_48, [16384, 3072]);  mul_48 = None
        permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_35: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg104_1, view_130, permute_65);  arg104_1 = view_130 = permute_65 = None
        view_131: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_35, [32, 512, 768]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_55: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_131, add_53);  view_131 = add_53 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
        getitem_24: "f32[32, 512, 1]" = var_mean_12[0]
        getitem_25: "f32[32, 512, 1]" = var_mean_12[1];  var_mean_12 = None
        sub_21: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_55, getitem_25);  add_55 = getitem_25 = None
        add_56: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        mul_49: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_12);  sub_21 = rsqrt_12 = None
        mul_50: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_49, arg105_1);  mul_49 = arg105_1 = None
        add_57: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_50, arg106_1);  mul_50 = arg106_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        view_132: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_57, [16384, 768])
        permute_66: "f32[768, 768]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_36: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg108_1, view_132, permute_66);  arg108_1 = view_132 = permute_66 = None
        view_133: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_36, [32, 512, 768]);  addmm_36 = None
        view_134: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_133, [32, 512, -1, 64]);  view_133 = None
        permute_67: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None

        # No stacktrace found for following nodes
        clone_default_15: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        view_135: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_57, [16384, 768])
        permute_68: "f32[768, 768]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_37: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg110_1, view_135, permute_68);  arg110_1 = view_135 = permute_68 = None
        view_136: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_37, [32, 512, 768]);  addmm_37 = None
        view_137: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_136, [32, 512, -1, 64]);  view_136 = None
        permute_69: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None

        # No stacktrace found for following nodes
        clone_default_16: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_69, memory_format = torch.contiguous_format);  permute_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        view_138: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_57, [16384, 768])
        permute_70: "f32[768, 768]" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        addmm_38: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg112_1, view_138, permute_70);  arg112_1 = view_138 = permute_70 = None
        view_139: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_38, [32, 512, 768]);  addmm_38 = None
        view_140: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_139, [32, 512, -1, 64]);  view_139 = None
        permute_71: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None

        # No stacktrace found for following nodes
        clone_default_17: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
        _scaled_dot_product_efficient_attention_default_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_15, clone_default_16, clone_default_17, None, False, scale = 0.125);  clone_default_15 = clone_default_16 = clone_default_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_57: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_5[0];  _scaled_dot_product_efficient_attention_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_73: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_57, [0, 2, 1, 3]);  getitem_57 = None
        clone_47: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_147: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_47, [32, 512, -1]);  clone_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_148: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_147, [16384, 768]);  view_147 = None
        permute_74: "f32[768, 768]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        addmm_39: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg114_1, view_148, permute_74);  arg114_1 = view_148 = permute_74 = None
        view_149: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_39, [32, 512, 768]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_59: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_149, add_57);  view_149 = add_57 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
        getitem_26: "f32[32, 512, 1]" = var_mean_13[0]
        getitem_27: "f32[32, 512, 1]" = var_mean_13[1];  var_mean_13 = None
        sub_23: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_27);  add_59 = getitem_27 = None
        add_60: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_52: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_13);  sub_23 = rsqrt_13 = None
        mul_53: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_52, arg115_1);  mul_52 = arg115_1 = None
        add_61: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_53, arg116_1);  mul_53 = arg116_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_150: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_61, [16384, 768])
        permute_75: "f32[768, 3072]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        addmm_40: "f32[16384, 3072]" = torch.ops.aten.addmm.default(arg118_1, view_150, permute_75);  arg118_1 = view_150 = permute_75 = None
        view_151: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_40, [32, 512, 3072]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_54: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.5)
        mul_55: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476);  view_151 = None
        erf_6: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
        add_62: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_56: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_54, add_62);  mul_54 = add_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_152: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_56, [16384, 3072]);  mul_56 = None
        permute_76: "f32[3072, 768]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_41: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg120_1, view_152, permute_76);  arg120_1 = view_152 = permute_76 = None
        view_153: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_41, [32, 512, 768]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_63: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_153, add_61);  view_153 = add_61 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
        getitem_28: "f32[32, 512, 1]" = var_mean_14[0]
        getitem_29: "f32[32, 512, 1]" = var_mean_14[1];  var_mean_14 = None
        sub_24: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_63, getitem_29);  add_63 = getitem_29 = None
        add_64: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        mul_57: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_14);  sub_24 = rsqrt_14 = None
        mul_58: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_57, arg121_1);  mul_57 = arg121_1 = None
        add_65: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_58, arg122_1);  mul_58 = arg122_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        view_154: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_65, [16384, 768])
        permute_77: "f32[768, 768]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_42: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg124_1, view_154, permute_77);  arg124_1 = view_154 = permute_77 = None
        view_155: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_42, [32, 512, 768]);  addmm_42 = None
        view_156: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_155, [32, 512, -1, 64]);  view_155 = None
        permute_78: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None

        # No stacktrace found for following nodes
        clone_default_12: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_78, memory_format = torch.contiguous_format);  permute_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        view_157: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_65, [16384, 768])
        permute_79: "f32[768, 768]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        addmm_43: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg126_1, view_157, permute_79);  arg126_1 = view_157 = permute_79 = None
        view_158: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_43, [32, 512, 768]);  addmm_43 = None
        view_159: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_158, [32, 512, -1, 64]);  view_158 = None
        permute_80: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_159, [0, 2, 1, 3]);  view_159 = None

        # No stacktrace found for following nodes
        clone_default_13: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        view_160: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_65, [16384, 768])
        permute_81: "f32[768, 768]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_44: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg128_1, view_160, permute_81);  arg128_1 = view_160 = permute_81 = None
        view_161: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_44, [32, 512, 768]);  addmm_44 = None
        view_162: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_161, [32, 512, -1, 64]);  view_161 = None
        permute_82: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None

        # No stacktrace found for following nodes
        clone_default_14: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        _scaled_dot_product_efficient_attention_default_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_12, clone_default_13, clone_default_14, None, False, scale = 0.125);  clone_default_12 = clone_default_13 = clone_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_56: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_4[0];  _scaled_dot_product_efficient_attention_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_84: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_56, [0, 2, 1, 3]);  getitem_56 = None
        clone_54: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_169: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_54, [32, 512, -1]);  clone_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_170: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_169, [16384, 768]);  view_169 = None
        permute_85: "f32[768, 768]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        addmm_45: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg130_1, view_170, permute_85);  arg130_1 = view_170 = permute_85 = None
        view_171: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_45, [32, 512, 768]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_67: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_171, add_65);  view_171 = add_65 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
        getitem_30: "f32[32, 512, 1]" = var_mean_15[0]
        getitem_31: "f32[32, 512, 1]" = var_mean_15[1];  var_mean_15 = None
        sub_26: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_67, getitem_31);  add_67 = getitem_31 = None
        add_68: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        mul_60: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_15);  sub_26 = rsqrt_15 = None
        mul_61: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_60, arg131_1);  mul_60 = arg131_1 = None
        add_69: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_61, arg132_1);  mul_61 = arg132_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_172: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_69, [16384, 768])
        permute_86: "f32[768, 3072]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_46: "f32[16384, 3072]" = torch.ops.aten.addmm.default(arg134_1, view_172, permute_86);  arg134_1 = view_172 = permute_86 = None
        view_173: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_46, [32, 512, 3072]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_62: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.5)
        mul_63: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476);  view_173 = None
        erf_7: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
        add_70: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_64: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_62, add_70);  mul_62 = add_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_174: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_64, [16384, 3072]);  mul_64 = None
        permute_87: "f32[3072, 768]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_47: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg136_1, view_174, permute_87);  arg136_1 = view_174 = permute_87 = None
        view_175: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_47, [32, 512, 768]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_71: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_175, add_69);  view_175 = add_69 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
        getitem_32: "f32[32, 512, 1]" = var_mean_16[0]
        getitem_33: "f32[32, 512, 1]" = var_mean_16[1];  var_mean_16 = None
        sub_27: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_71, getitem_33);  add_71 = getitem_33 = None
        add_72: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        mul_65: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_16);  sub_27 = rsqrt_16 = None
        mul_66: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_65, arg137_1);  mul_65 = arg137_1 = None
        add_73: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_66, arg138_1);  mul_66 = arg138_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        view_176: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_73, [16384, 768])
        permute_88: "f32[768, 768]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_48: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg140_1, view_176, permute_88);  arg140_1 = view_176 = permute_88 = None
        view_177: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_48, [32, 512, 768]);  addmm_48 = None
        view_178: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_177, [32, 512, -1, 64]);  view_177 = None
        permute_89: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None

        # No stacktrace found for following nodes
        clone_default_9: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_89, memory_format = torch.contiguous_format);  permute_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        view_179: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_73, [16384, 768])
        permute_90: "f32[768, 768]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        addmm_49: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg142_1, view_179, permute_90);  arg142_1 = view_179 = permute_90 = None
        view_180: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_49, [32, 512, 768]);  addmm_49 = None
        view_181: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_180, [32, 512, -1, 64]);  view_180 = None
        permute_91: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None

        # No stacktrace found for following nodes
        clone_default_10: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        view_182: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_73, [16384, 768])
        permute_92: "f32[768, 768]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_50: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg144_1, view_182, permute_92);  arg144_1 = view_182 = permute_92 = None
        view_183: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_50, [32, 512, 768]);  addmm_50 = None
        view_184: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_183, [32, 512, -1, 64]);  view_183 = None
        permute_93: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None

        # No stacktrace found for following nodes
        clone_default_11: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
        _scaled_dot_product_efficient_attention_default_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_9, clone_default_10, clone_default_11, None, False, scale = 0.125);  clone_default_9 = clone_default_10 = clone_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_55: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_3[0];  _scaled_dot_product_efficient_attention_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_95: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_55, [0, 2, 1, 3]);  getitem_55 = None
        clone_61: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_191: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_61, [32, 512, -1]);  clone_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_192: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_191, [16384, 768]);  view_191 = None
        permute_96: "f32[768, 768]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_51: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg146_1, view_192, permute_96);  arg146_1 = view_192 = permute_96 = None
        view_193: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_51, [32, 512, 768]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_75: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_193, add_73);  view_193 = add_73 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
        getitem_34: "f32[32, 512, 1]" = var_mean_17[0]
        getitem_35: "f32[32, 512, 1]" = var_mean_17[1];  var_mean_17 = None
        sub_29: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_75, getitem_35);  add_75 = getitem_35 = None
        add_76: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        mul_68: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_17);  sub_29 = rsqrt_17 = None
        mul_69: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_68, arg147_1);  mul_68 = arg147_1 = None
        add_77: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_69, arg148_1);  mul_69 = arg148_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_194: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_77, [16384, 768])
        permute_97: "f32[768, 3072]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        addmm_52: "f32[16384, 3072]" = torch.ops.aten.addmm.default(arg150_1, view_194, permute_97);  arg150_1 = view_194 = permute_97 = None
        view_195: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_52, [32, 512, 3072]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_70: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.5)
        mul_71: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476);  view_195 = None
        erf_8: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_71);  mul_71 = None
        add_78: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_72: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_70, add_78);  mul_70 = add_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_196: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_72, [16384, 3072]);  mul_72 = None
        permute_98: "f32[3072, 768]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_53: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg152_1, view_196, permute_98);  arg152_1 = view_196 = permute_98 = None
        view_197: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_53, [32, 512, 768]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_79: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_197, add_77);  view_197 = add_77 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_79, [2], correction = 0, keepdim = True)
        getitem_36: "f32[32, 512, 1]" = var_mean_18[0]
        getitem_37: "f32[32, 512, 1]" = var_mean_18[1];  var_mean_18 = None
        sub_30: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_79, getitem_37);  add_79 = getitem_37 = None
        add_80: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        mul_73: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_18);  sub_30 = rsqrt_18 = None
        mul_74: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_73, arg153_1);  mul_73 = arg153_1 = None
        add_81: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_74, arg154_1);  mul_74 = arg154_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        view_198: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_81, [16384, 768])
        permute_99: "f32[768, 768]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        addmm_54: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg156_1, view_198, permute_99);  arg156_1 = view_198 = permute_99 = None
        view_199: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_54, [32, 512, 768]);  addmm_54 = None
        view_200: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_199, [32, 512, -1, 64]);  view_199 = None
        permute_100: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None

        # No stacktrace found for following nodes
        clone_default_6: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_100, memory_format = torch.contiguous_format);  permute_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        view_201: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_81, [16384, 768])
        permute_101: "f32[768, 768]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        addmm_55: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg158_1, view_201, permute_101);  arg158_1 = view_201 = permute_101 = None
        view_202: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_55, [32, 512, 768]);  addmm_55 = None
        view_203: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_202, [32, 512, -1, 64]);  view_202 = None
        permute_102: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None

        # No stacktrace found for following nodes
        clone_default_7: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        view_204: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_81, [16384, 768])
        permute_103: "f32[768, 768]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm_56: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg160_1, view_204, permute_103);  arg160_1 = view_204 = permute_103 = None
        view_205: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_56, [32, 512, 768]);  addmm_56 = None
        view_206: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_205, [32, 512, -1, 64]);  view_205 = None
        permute_104: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None

        # No stacktrace found for following nodes
        clone_default_8: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        _scaled_dot_product_efficient_attention_default_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_6, clone_default_7, clone_default_8, None, False, scale = 0.125);  clone_default_6 = clone_default_7 = clone_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_54: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_2[0];  _scaled_dot_product_efficient_attention_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_106: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_54, [0, 2, 1, 3]);  getitem_54 = None
        clone_68: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_213: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_68, [32, 512, -1]);  clone_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_214: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_213, [16384, 768]);  view_213 = None
        permute_107: "f32[768, 768]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        addmm_57: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg162_1, view_214, permute_107);  arg162_1 = view_214 = permute_107 = None
        view_215: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_57, [32, 512, 768]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_83: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_215, add_81);  view_215 = add_81 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
        getitem_38: "f32[32, 512, 1]" = var_mean_19[0]
        getitem_39: "f32[32, 512, 1]" = var_mean_19[1];  var_mean_19 = None
        sub_32: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_83, getitem_39);  add_83 = getitem_39 = None
        add_84: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        mul_76: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_19);  sub_32 = rsqrt_19 = None
        mul_77: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_76, arg163_1);  mul_76 = arg163_1 = None
        add_85: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_77, arg164_1);  mul_77 = arg164_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_216: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_85, [16384, 768])
        permute_108: "f32[768, 3072]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_58: "f32[16384, 3072]" = torch.ops.aten.addmm.default(arg166_1, view_216, permute_108);  arg166_1 = view_216 = permute_108 = None
        view_217: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_58, [32, 512, 3072]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_78: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.5)
        mul_79: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476);  view_217 = None
        erf_9: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_79);  mul_79 = None
        add_86: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_80: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_78, add_86);  mul_78 = add_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_218: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_80, [16384, 3072]);  mul_80 = None
        permute_109: "f32[3072, 768]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        addmm_59: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg168_1, view_218, permute_109);  arg168_1 = view_218 = permute_109 = None
        view_219: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_59, [32, 512, 768]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_87: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_219, add_85);  view_219 = add_85 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
        getitem_40: "f32[32, 512, 1]" = var_mean_20[0]
        getitem_41: "f32[32, 512, 1]" = var_mean_20[1];  var_mean_20 = None
        sub_33: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_87, getitem_41);  add_87 = getitem_41 = None
        add_88: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        mul_81: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_20);  sub_33 = rsqrt_20 = None
        mul_82: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_81, arg169_1);  mul_81 = arg169_1 = None
        add_89: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_82, arg170_1);  mul_82 = arg170_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        view_220: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_89, [16384, 768])
        permute_110: "f32[768, 768]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_60: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg172_1, view_220, permute_110);  arg172_1 = view_220 = permute_110 = None
        view_221: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_60, [32, 512, 768]);  addmm_60 = None
        view_222: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_221, [32, 512, -1, 64]);  view_221 = None
        permute_111: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None

        # No stacktrace found for following nodes
        clone_default_3: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_111, memory_format = torch.contiguous_format);  permute_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        view_223: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_89, [16384, 768])
        permute_112: "f32[768, 768]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        addmm_61: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg174_1, view_223, permute_112);  arg174_1 = view_223 = permute_112 = None
        view_224: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_61, [32, 512, 768]);  addmm_61 = None
        view_225: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_224, [32, 512, -1, 64]);  view_224 = None
        permute_113: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None

        # No stacktrace found for following nodes
        clone_default_4: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_113, memory_format = torch.contiguous_format);  permute_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        view_226: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_89, [16384, 768])
        permute_114: "f32[768, 768]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        addmm_62: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg176_1, view_226, permute_114);  arg176_1 = view_226 = permute_114 = None
        view_227: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_62, [32, 512, 768]);  addmm_62 = None
        view_228: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_227, [32, 512, -1, 64]);  view_227 = None
        permute_115: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None

        # No stacktrace found for following nodes
        clone_default_5: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
        _scaled_dot_product_efficient_attention_default_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_3, clone_default_4, clone_default_5, None, False, scale = 0.125);  clone_default_3 = clone_default_4 = clone_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_53: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_1[0];  _scaled_dot_product_efficient_attention_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_117: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_53, [0, 2, 1, 3]);  getitem_53 = None
        clone_75: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_235: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_75, [32, 512, -1]);  clone_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_236: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_235, [16384, 768]);  view_235 = None
        permute_118: "f32[768, 768]" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        addmm_63: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg178_1, view_236, permute_118);  arg178_1 = view_236 = permute_118 = None
        view_237: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_63, [32, 512, 768]);  addmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_91: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_237, add_89);  view_237 = add_89 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
        getitem_42: "f32[32, 512, 1]" = var_mean_21[0]
        getitem_43: "f32[32, 512, 1]" = var_mean_21[1];  var_mean_21 = None
        sub_35: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_91, getitem_43);  add_91 = getitem_43 = None
        add_92: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        mul_84: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_21);  sub_35 = rsqrt_21 = None
        mul_85: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_84, arg179_1);  mul_84 = arg179_1 = None
        add_93: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_85, arg180_1);  mul_85 = arg180_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_238: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_93, [16384, 768])
        permute_119: "f32[768, 3072]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        addmm_64: "f32[16384, 3072]" = torch.ops.aten.addmm.default(arg182_1, view_238, permute_119);  arg182_1 = view_238 = permute_119 = None
        view_239: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_64, [32, 512, 3072]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_86: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.5)
        mul_87: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476);  view_239 = None
        erf_10: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
        add_94: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_88: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_86, add_94);  mul_86 = add_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_240: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_88, [16384, 3072]);  mul_88 = None
        permute_120: "f32[3072, 768]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        addmm_65: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg184_1, view_240, permute_120);  arg184_1 = view_240 = permute_120 = None
        view_241: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_65, [32, 512, 768]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_95: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_241, add_93);  view_241 = add_93 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_95, [2], correction = 0, keepdim = True)
        getitem_44: "f32[32, 512, 1]" = var_mean_22[0]
        getitem_45: "f32[32, 512, 1]" = var_mean_22[1];  var_mean_22 = None
        sub_36: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_95, getitem_45);  add_95 = getitem_45 = None
        add_96: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        mul_89: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_22);  sub_36 = rsqrt_22 = None
        mul_90: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_89, arg185_1);  mul_89 = arg185_1 = None
        add_97: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_90, arg186_1);  mul_90 = arg186_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        view_242: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_97, [16384, 768])
        permute_121: "f32[768, 768]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_66: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg188_1, view_242, permute_121);  arg188_1 = view_242 = permute_121 = None
        view_243: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_66, [32, 512, 768]);  addmm_66 = None
        view_244: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_243, [32, 512, -1, 64]);  view_243 = None
        permute_122: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None

        # No stacktrace found for following nodes
        clone_default: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        view_245: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_97, [16384, 768])
        permute_123: "f32[768, 768]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        addmm_67: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg190_1, view_245, permute_123);  arg190_1 = view_245 = permute_123 = None
        view_246: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_67, [32, 512, 768]);  addmm_67 = None
        view_247: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_246, [32, 512, -1, 64]);  view_246 = None
        permute_124: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_247, [0, 2, 1, 3]);  view_247 = None

        # No stacktrace found for following nodes
        clone_default_1: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        view_248: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_97, [16384, 768])
        permute_125: "f32[768, 768]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_68: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg192_1, view_248, permute_125);  arg192_1 = view_248 = permute_125 = None
        view_249: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_68, [32, 512, 768]);  addmm_68 = None
        view_250: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(view_249, [32, 512, -1, 64]);  view_249 = None
        permute_126: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None

        # No stacktrace found for following nodes
        clone_default_2: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
        _scaled_dot_product_efficient_attention_default = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default, clone_default_1, clone_default_2, None, False, scale = 0.125);  clone_default = clone_default_1 = clone_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        getitem_52: "f32[32, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default[0];  _scaled_dot_product_efficient_attention_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_128: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_52, [0, 2, 1, 3]);  getitem_52 = None
        clone_82: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_257: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_82, [32, 512, -1]);  clone_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_258: "f32[16384, 768]" = torch.ops.aten.reshape.default(view_257, [16384, 768]);  view_257 = None
        permute_129: "f32[768, 768]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        addmm_69: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg194_1, view_258, permute_129);  arg194_1 = view_258 = permute_129 = None
        view_259: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_69, [32, 512, 768]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_99: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_259, add_97);  view_259 = add_97 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_99, [2], correction = 0, keepdim = True)
        getitem_46: "f32[32, 512, 1]" = var_mean_23[0]
        getitem_47: "f32[32, 512, 1]" = var_mean_23[1];  var_mean_23 = None
        sub_38: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_99, getitem_47);  add_99 = getitem_47 = None
        add_100: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        mul_92: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_23);  sub_38 = rsqrt_23 = None
        mul_93: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_92, arg195_1);  mul_92 = arg195_1 = None
        add_101: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_93, arg196_1);  mul_93 = arg196_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_260: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_101, [16384, 768])
        permute_130: "f32[768, 3072]" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        addmm_70: "f32[16384, 3072]" = torch.ops.aten.addmm.default(arg198_1, view_260, permute_130);  arg198_1 = view_260 = permute_130 = None
        view_261: "f32[32, 512, 3072]" = torch.ops.aten.reshape.default(addmm_70, [32, 512, 3072]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_94: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.5)
        mul_95: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476);  view_261 = None
        erf_11: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_95);  mul_95 = None
        add_102: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_96: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_94, add_102);  mul_94 = add_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_262: "f32[16384, 3072]" = torch.ops.aten.reshape.default(mul_96, [16384, 3072]);  mul_96 = None
        permute_131: "f32[3072, 768]" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
        addmm_71: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg200_1, view_262, permute_131);  arg200_1 = view_262 = permute_131 = None
        view_263: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_71, [32, 512, 768]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_103: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_263, add_101);  view_263 = add_101 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_103, [2], correction = 0, keepdim = True)
        getitem_48: "f32[32, 512, 1]" = var_mean_24[0]
        getitem_49: "f32[32, 512, 1]" = var_mean_24[1];  var_mean_24 = None
        sub_39: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_103, getitem_49);  add_103 = getitem_49 = None
        add_104: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        mul_97: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_24);  sub_39 = rsqrt_24 = None
        mul_98: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_97, arg201_1);  mul_97 = arg201_1 = None
        add_105: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_98, arg202_1);  mul_98 = arg202_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:359 in forward, code: hidden_states = self.dense(hidden_states)
        view_264: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_105, [16384, 768]);  add_105 = None
        permute_133: "f32[768, 768]" = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
        addmm_73: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg206_1, view_264, permute_133);  arg206_1 = view_264 = permute_133 = None
        view_265: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_73, [32, 512, 768]);  addmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_99: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_265, 0.5)
        mul_100: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_265, 0.7071067811865476);  view_265 = None
        erf_12: "f32[32, 512, 768]" = torch.ops.aten.erf.default(mul_100);  mul_100 = None
        add_106: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_101: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_99, add_106);  mul_99 = add_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:361 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        var_mean_25 = torch.ops.aten.var_mean.correction(mul_101, [2], correction = 0, keepdim = True)
        getitem_50: "f32[32, 512, 1]" = var_mean_25[0]
        getitem_51: "f32[32, 512, 1]" = var_mean_25[1];  var_mean_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:634 in forward, code: labels.view(-1),
        view_269: "i64[16384]" = torch.ops.aten.reshape.default(arg210_1, [-1]);  arg210_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:632 in forward, code: masked_lm_loss = loss_fct(
        ne_1: "b8[16384]" = torch.ops.aten.ne.Scalar(view_269, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:378 in forward, code: hidden_states = self.decoder(hidden_states)
        full_default_5: "f32[2]" = torch.ops.aten.full.default([2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default: "f32[30524]" = torch.ops.aten.cat.default([arg209_1, full_default_5]);  arg209_1 = full_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:361 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        sub_40: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_101, getitem_51);  mul_101 = getitem_51 = None
        add_107: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_25: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
        mul_102: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_25);  sub_40 = rsqrt_25 = None
        mul_103: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_102, arg207_1);  mul_102 = arg207_1 = None
        add_108: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_103, arg208_1);  mul_103 = arg208_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:378 in forward, code: hidden_states = self.decoder(hidden_states)
        view_266: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_108, [16384, 768]);  add_108 = None
        permute_134: "f32[768, 30522]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        constant_pad_nd_default: "f32[768, 30524]" = torch.ops.aten.constant_pad_nd.default(permute_134, [0, 2, 0, 0]);  permute_134 = None
        addmm_default: "f32[16384, 30524]" = torch.ops.aten.addmm.default(cat_default, view_266, constant_pad_nd_default);  cat_default = view_266 = constant_pad_nd_default = None
        slice_tensor: "f32[16384, 30522]" = torch.ops.aten.slice.Tensor(addmm_default, 1, 0, -2);  addmm_default = None
        view_267: "f32[32, 512, 30522]" = torch.ops.aten.reshape.default(slice_tensor, [32, 512, 30522]);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:633 in forward, code: prediction_scores.view(-1, self.config.vocab_size),
        view_268: "f32[16384, 30522]" = torch.ops.aten.reshape.default(view_267, [-1, 30522])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:632 in forward, code: masked_lm_loss = loss_fct(
        amax_12: "f32[16384, 1]" = torch.ops.aten.amax.default(view_268, [1], True)
        sub_41: "f32[16384, 30522]" = torch.ops.aten.sub.Tensor(view_268, amax_12);  view_268 = amax_12 = None
        exp_12: "f32[16384, 30522]" = torch.ops.aten.exp.default(sub_41)
        sum_13: "f32[16384, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
        log: "f32[16384, 1]" = torch.ops.aten.log.default(sum_13);  sum_13 = None
        sub_42: "f32[16384, 30522]" = torch.ops.aten.sub.Tensor(sub_41, log);  sub_41 = log = None
        ne: "b8[16384]" = torch.ops.aten.ne.Scalar(view_269, -100)
        full_default_3: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[16384]" = torch.ops.aten.where.self(ne, view_269, full_default_3);  ne = full_default_3 = None
        unsqueeze_2: "i64[16384, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[16384, 1]" = torch.ops.aten.gather.default(sub_42, 1, unsqueeze_2);  sub_42 = unsqueeze_2 = None
        squeeze: "f32[16384]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[16384]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_4: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[16384]" = torch.ops.aten.where.self(ne_1, neg, full_default_4);  ne_1 = neg = full_default_4 = None
        sum_15: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
        ne_2: "b8[16384]" = torch.ops.aten.ne.Scalar(view_269, -100);  view_269 = None
        sum_14: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
        div_12: "f32[]" = torch.ops.aten.div.Tensor(sum_15, convert_element_type);  sum_15 = convert_element_type = None
        return (div_12, view_267)
