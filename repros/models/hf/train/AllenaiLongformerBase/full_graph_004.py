class GraphModule(torch.nn.Module):
    def forward(self, primals_4: "f32[768][1]cuda:0", primals_8: "i64[8, 1024][1024, 1]cuda:0", view: "bf16[8192, 768][768, 1]cuda:0", addmm: "bf16[8192, 768][768, 1]cuda:0", getitem_1: "f32[8, 1024, 1][1024, 1, 1]cuda:0", rsqrt: "f32[8, 1024, 1][1024, 1, 1]cuda:0", view_2: "bf16[8192, 768][768, 1]cuda:0", view_3: "bf16[8, 1024, 50265][51478528, 50272, 1]cuda:0", amax: "f32[8192, 1][1, 1]cuda:0", log: "f32[8192, 1][1, 1]cuda:0", convert_element_type_18: "f32[][]cuda:0", permute_2: "bf16[50265, 768][768, 1]cuda:0", permute_6: "bf16[768, 768][768, 1]cuda:0", tangents_1: "f32[][]cuda:0", tangents_2: "bf16[8, 1024, 50265][51471360, 50265, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1635 in torch_dynamo_resume_in_forward_at_1616, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        div_1: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_18);  tangents_1 = convert_element_type_18 = None
        view_5: "i64[8192][1]cuda:0" = torch.ops.aten.reshape.default(primals_8, [-1]);  primals_8 = None
        unsqueeze_1: "i64[8192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_5, 1);  view_5 = None
        ne_3: "b8[8192, 1][1, 1]cuda:0" = torch.ops.aten.ne.Scalar(unsqueeze_1, -100)
        full_default: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "i64[8192, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_3, unsqueeze_1, full_default);  unsqueeze_1 = full_default = None

        # No stacktrace found for following nodes
        iota_default: "i64[50265][1]cuda:0" = torch.ops.prims.iota.default(50265, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 50265][50265, 1]cuda:0" = torch.ops.aten.reshape.default(iota_default, [1, 50265]);  iota_default = None
        expand_default: "i64[8192, 50265][1, 0]cuda:0" = torch.ops.aten.expand.default(where_2, [8192, 50265]);  where_2 = None
        eq_tensor: "b8[8192, 50265][50265, 1]cuda:0" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1635 in torch_dynamo_resume_in_forward_at_1616, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        where_self: "f32[8192, 50265][50265, 1]cuda:0" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        full_default_1: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_3, div_1, full_default_1);  ne_3 = div_1 = full_default_1 = None
        mul_5: "f32[8192, 50265][50265, 1]cuda:0" = torch.ops.aten.mul.Tensor(where_self, where_3);  where_self = where_3 = None
        convert_element_type_19: "bf16[8192, 50265][50265, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        convert_element_type_20: "f32[8192, 50265][50265, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_19, torch.float32);  convert_element_type_19 = None
        view_4: "bf16[8192, 50265][50272, 1]cuda:0" = torch.ops.aten.reshape.default(view_3, [-1, 50265]);  view_3 = None
        convert_element_type_15: "f32[8192, 50265][50265, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        sub_1: "f32[8192, 50265][50265, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_15, amax);  convert_element_type_15 = amax = None
        sub_2: "f32[8192, 50265][50265, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1, log);  sub_1 = log = None
        convert_element_type_16: "bf16[8192, 50265][50265, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_2, torch.bfloat16);  sub_2 = None
        convert_element_type_17: "f32[8192, 50265][50265, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_16, torch.float32);  convert_element_type_16 = None
        exp_1: "f32[8192, 50265][50265, 1]cuda:0" = torch.ops.aten.exp.default(convert_element_type_17);  convert_element_type_17 = None
        sum_4: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_20, [1], True)
        mul_6: "f32[8192, 50265][50265, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_1, sum_4);  exp_1 = sum_4 = None
        sub_3: "f32[8192, 50265][50265, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_20, mul_6);  convert_element_type_20 = mul_6 = None
        convert_element_type_22: "bf16[8192, 50265][50265, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_3, torch.bfloat16);  sub_3 = None
        view_6: "bf16[8, 1024, 50265][51471360, 50265, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_22, [8, 1024, 50265]);  convert_element_type_22 = None
        add_3: "bf16[8, 1024, 50265][51471360, 50265, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_2, view_6);  tangents_2 = view_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1281 in forward, code: x = self.decoder(x)
        view_7: "bf16[8192, 50265][50265, 1]cuda:0" = torch.ops.aten.reshape.default(add_3, [8192, 50265]);  add_3 = None
        constant_pad_nd_default_1: "bf16[8192, 50272][50272, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_7, [0, 7, 0, 0])
        constant_pad_nd_default_2: "bf16[50272, 768][768, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(permute_2, [0, 0, 0, 7]);  permute_2 = None
        mm_default_1: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(constant_pad_nd_default_1, constant_pad_nd_default_2);  constant_pad_nd_default_1 = constant_pad_nd_default_2 = None
        permute_3: "bf16[50265, 8192][1, 50265]cuda:0" = torch.ops.aten.permute.default(view_7, [1, 0])
        constant_pad_nd_default: "bf16[50272, 8192][8192, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(permute_3, [0, 0, 0, 7]);  permute_3 = None
        mm_default: "bf16[50272, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(constant_pad_nd_default, view_2);  constant_pad_nd_default = view_2 = None
        slice_tensor: "bf16[50265, 768][768, 1]cuda:0" = torch.ops.aten.slice.Tensor(mm_default, 0, 0, -7);  mm_default = None
        sum_5: "f32[1, 50265][50265, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_7, [0], True, dtype = torch.float32);  view_7 = None
        view_8: "f32[50265][1]cuda:0" = torch.ops.aten.reshape.default(sum_5, [50265]);  sum_5 = None
        convert_element_type_27: "bf16[50265][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_8, torch.bfloat16);  view_8 = None
        view_9: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_default_1, [8, 1024, 768]);  mm_default_1 = None
        convert_element_type_28: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_9, torch.float32);  view_9 = None
        convert_element_type_29: "f32[50265, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(slice_tensor, torch.float32);  slice_tensor = None
        convert_element_type_30: "f32[50265][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_27, torch.float32);  convert_element_type_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1278 in forward, code: x = self.layer_norm(x)
        mul_8: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_28, primals_4);  primals_4 = None
        mul_9: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_8, 768)
        sum_6: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_8, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1276 in forward, code: x = self.dense(features)
        view_1: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [8, 1024, 768]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_6: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        mul: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.5)
        mul_1: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.7071067811865476)
        erf: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.erf.default(mul_1);  mul_1 = None
        add: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_2: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, add);  mul = None
        convert_element_type_7: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1278 in forward, code: x = self.layer_norm(x)
        convert_element_type_8: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_7, torch.float32);  convert_element_type_7 = None
        sub: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_8, getitem_1);  convert_element_type_8 = getitem_1 = None
        mul_3: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_10: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_8, mul_3);  mul_8 = None
        sum_7: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_10, [2], True);  mul_10 = None
        mul_11: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, sum_7);  sum_7 = None
        sub_5: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_9, sum_6);  mul_9 = sum_6 = None
        sub_6: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_5, mul_11);  sub_5 = mul_11 = None
        div_2: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        mul_12: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_2, sub_6);  div_2 = sub_6 = None
        mul_13: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_28, mul_3);  mul_3 = None
        sum_8: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_13, [0, 1]);  mul_13 = None
        sum_9: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_28, [0, 1]);  convert_element_type_28 = None
        convert_element_type_31: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_12, torch.bfloat16);  mul_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_32: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_31, torch.float32);  convert_element_type_31 = None
        mul_15: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add, 0.5);  add = None
        mul_16: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_6, convert_element_type_6)
        mul_17: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_16, -0.5);  mul_16 = None
        exp_2: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.exp.default(mul_17);  mul_17 = None
        mul_18: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
        mul_19: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_6, mul_18);  convert_element_type_6 = mul_18 = None
        add_5: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_15, mul_19);  mul_15 = mul_19 = None
        mul_20: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_32, add_5);  convert_element_type_32 = add_5 = None
        convert_element_type_34: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_20, torch.bfloat16);  mul_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1276 in forward, code: x = self.dense(features)
        view_10: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_34, [8192, 768]);  convert_element_type_34 = None
        mm_2: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_10, permute_6);  permute_6 = None
        permute_7: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_10, [1, 0])
        mm_3: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_7, view);  permute_7 = view = None
        sum_10: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_10, [0], True, dtype = torch.float32);  view_10 = None
        view_11: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_10, [768]);  sum_10 = None
        convert_element_type_39: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_11, torch.bfloat16);  view_11 = None
        view_12: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [8, 1024, 768]);  mm_2 = None
        convert_element_type_40: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_12, torch.float32);  view_12 = None
        convert_element_type_41: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_42: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_39, torch.float32);  convert_element_type_39 = None
        return (convert_element_type_41, convert_element_type_42, convert_element_type_40, sum_8, sum_9, convert_element_type_29, convert_element_type_30, None)
