class GraphModule(torch.nn.Module):
    def forward(self, arg1_1: "f32[30522, 768]", arg0_1: "i64[32, 512]", arg3_1: "f32[512, 768]", arg2_1: "i64[1, 512]", arg4_1: "f32[1024, 768]", arg5_1: "f32[1024, 768]", arg6_1: "f32[1024, 768]", arg7_1: "f32[1024, 768]", arg8_1: "f32[2, 768]", arg9_1: "f32[768]", arg10_1: "f32[768]", arg11_1: "f32[768, 768]", arg13_1: "f32[768, 768]", arg15_1: "f32[768, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:92 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding_default: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 0);  arg1_1 = arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:95 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_default_1: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg3_1, arg2_1);  arg3_1 = arg2_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:109 in forward, code: words_embeddings
        add_tensor: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:501 in forward, code: bbox = torch.zeros(input_shape + (4,), dtype=torch.long, device=device)
        full_default: "i64[32, 512, 4]" = torch.ops.aten.full.default([32, 512, 4], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:97 in forward, code: left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
        select_int: "i64[32, 512]" = torch.ops.aten.select.int(full_default, 2, 0)
        embedding_default_2: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg4_1, select_int);  select_int = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:109 in forward, code: words_embeddings
        add_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor, embedding_default_2);  add_tensor = embedding_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:98 in forward, code: upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
        select_int_1: "i64[32, 512]" = torch.ops.aten.select.int(full_default, 2, 1)
        embedding_default_3: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg5_1, select_int_1);  select_int_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:109 in forward, code: words_embeddings
        add_tensor_2: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_1, embedding_default_3);  add_tensor_1 = embedding_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:99 in forward, code: right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
        select_int_2: "i64[32, 512]" = torch.ops.aten.select.int(full_default, 2, 2)
        embedding_default_4: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg4_1, select_int_2);  arg4_1 = select_int_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:109 in forward, code: words_embeddings
        add_tensor_3: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_2, embedding_default_4);  add_tensor_2 = embedding_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:100 in forward, code: lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        select_int_3: "i64[32, 512]" = torch.ops.aten.select.int(full_default, 2, 3)
        embedding_default_5: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg5_1, select_int_3);  arg5_1 = select_int_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:109 in forward, code: words_embeddings
        add_tensor_4: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_3, embedding_default_5);  add_tensor_3 = embedding_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:104 in forward, code: h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        select_int_4: "i64[32, 512]" = torch.ops.aten.select.int(full_default, 2, 3)
        select_int_5: "i64[32, 512]" = torch.ops.aten.select.int(full_default, 2, 1)
        sub_tensor: "i64[32, 512]" = torch.ops.aten.sub.Tensor(select_int_4, select_int_5);  select_int_4 = select_int_5 = None
        embedding_default_6: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg6_1, sub_tensor);  arg6_1 = sub_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:109 in forward, code: words_embeddings
        add_tensor_5: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_4, embedding_default_6);  add_tensor_4 = embedding_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:105 in forward, code: w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])
        select_int_6: "i64[32, 512]" = torch.ops.aten.select.int(full_default, 2, 2)
        select_int_7: "i64[32, 512]" = torch.ops.aten.select.int(full_default, 2, 0);  full_default = None
        sub_tensor_1: "i64[32, 512]" = torch.ops.aten.sub.Tensor(select_int_6, select_int_7);  select_int_6 = select_int_7 = None
        embedding_default_7: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg7_1, sub_tensor_1);  arg7_1 = sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:109 in forward, code: words_embeddings
        add_tensor_6: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_5, embedding_default_7);  add_tensor_5 = embedding_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:498 in forward, code: token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        full_default_1: "i64[32, 512]" = torch.ops.aten.full.default([32, 512], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:106 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_default_8: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg8_1, full_default_1);  arg8_1 = full_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:109 in forward, code: words_embeddings
        add_tensor_7: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_6, embedding_default_8);  add_tensor_6 = embedding_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:119 in forward, code: embeddings = self.LayerNorm(embeddings)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_7, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[32, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor_2: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_tensor_7, getitem_1);  add_tensor_7 = getitem_1 = None
        add_tensor_8: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_8);  add_tensor_8 = None
        mul_tensor: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default);  sub_tensor_2 = rsqrt_default = None
        mul_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg9_1);  mul_tensor = arg9_1 = None
        add_tensor_9: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg10_1);  mul_tensor_1 = arg10_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_tensor_9, _shape_param_0);  _shape_param_0 = None
        permute_default: "f32[768, 768]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_1: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_tensor_9, _shape_param_1);  _shape_param_1 = None
        permute_default_1: "f32[768, 768]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_2: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_tensor_9, _shape_param_2);  add_tensor_9 = _shape_param_2 = None
        permute_default_2: "f32[768, 768]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        return (reshape_default, permute_default, reshape_default_1, permute_default_1, reshape_default_2, permute_default_2)
