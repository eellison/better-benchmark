class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[768, 768]", primals_4: "f32[768]", primals_6: "f32[50265, 768]", primals_8: "i64[2, 1024]", view: "f32[2048, 768]", addmm: "f32[2048, 768]", getitem_1: "f32[2, 1024, 1]", rsqrt: "f32[2, 1024, 1]", view_2: "f32[2048, 768]", view_3: "f32[2, 1024, 50265]", amax: "f32[2048, 1]", log: "f32[2048, 1]", convert_element_type: "f32[]", tangents_1: "f32[]", tangents_2: "f32[2, 1024, 50265]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1635 in torch_dynamo_resume_in_forward_at_1616, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        div_1: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type);  tangents_1 = convert_element_type = None
        view_5: "i64[2048]" = torch.ops.aten.reshape.default(primals_8, [-1]);  primals_8 = None
        unsqueeze_1: "i64[2048, 1]" = torch.ops.aten.unsqueeze.default(view_5, 1);  view_5 = None
        ne_3: "b8[2048, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_1, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "i64[2048, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_1, full_default);  unsqueeze_1 = full_default = None

        # No stacktrace found for following nodes
        iota_default: "i64[50265]" = torch.ops.prims.iota.default(50265, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 50265]" = torch.ops.aten.reshape.default(iota_default, [1, 50265]);  iota_default = None
        expand_default: "i64[2048, 50265]" = torch.ops.aten.expand.default(where_2, [2048, 50265]);  where_2 = None
        eq_tensor: "b8[2048, 50265]" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1635 in torch_dynamo_resume_in_forward_at_1616, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        where_self: "f32[2048, 50265]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "f32[2048, 1]" = torch.ops.aten.where.self(ne_3, div_1, full_default_1);  ne_3 = div_1 = full_default_1 = None
        mul_5: "f32[2048, 50265]" = torch.ops.aten.mul.Tensor(where_self, where_3);  where_self = where_3 = None
        view_4: "f32[2048, 50265]" = torch.ops.aten.reshape.default(view_3, [-1, 50265]);  view_3 = None
        sub_1: "f32[2048, 50265]" = torch.ops.aten.sub.Tensor(view_4, amax);  view_4 = amax = None
        sub_2: "f32[2048, 50265]" = torch.ops.aten.sub.Tensor(sub_1, log);  sub_1 = log = None
        exp_1: "f32[2048, 50265]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_4: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_5, [1], True)
        mul_6: "f32[2048, 50265]" = torch.ops.aten.mul.Tensor(exp_1, sum_4);  exp_1 = sum_4 = None
        sub_3: "f32[2048, 50265]" = torch.ops.aten.sub.Tensor(mul_5, mul_6);  mul_5 = mul_6 = None
        view_6: "f32[2, 1024, 50265]" = torch.ops.aten.reshape.default(sub_3, [2, 1024, 50265]);  sub_3 = None
        add_3: "f32[2, 1024, 50265]" = torch.ops.aten.add.Tensor(tangents_2, view_6);  tangents_2 = view_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1281 in forward, code: x = self.decoder(x)
        view_7: "f32[2048, 50265]" = torch.ops.aten.reshape.default(add_3, [2048, 50265]);  add_3 = None
        permute_1: "f32[768, 50265]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_2: "f32[50265, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        mm: "f32[2048, 768]" = torch.ops.aten.mm.default(view_7, permute_2);  permute_2 = None
        permute_3: "f32[50265, 2048]" = torch.ops.aten.permute.default(view_7, [1, 0])
        mm_1: "f32[50265, 768]" = torch.ops.aten.mm.default(permute_3, view_2);  permute_3 = view_2 = None
        sum_5: "f32[1, 50265]" = torch.ops.aten.sum.dim_IntList(view_7, [0], True);  view_7 = None
        view_8: "f32[50265]" = torch.ops.aten.reshape.default(sum_5, [50265]);  sum_5 = None
        view_9: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(mm, [2, 1024, 768]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1278 in forward, code: x = self.layer_norm(x)
        mul_8: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(view_9, primals_4);  primals_4 = None
        mul_9: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_8, 768)
        sum_6: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_8, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1276 in forward, code: x = self.dense(features)
        view_1: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(addmm, [2, 1024, 768]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1, 0.5)
        mul_1: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1, 0.7071067811865476)
        erf: "f32[2, 1024, 768]" = torch.ops.aten.erf.default(mul_1);  mul_1 = None
        add: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_2: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, add);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1278 in forward, code: x = self.layer_norm(x)
        sub: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_2, getitem_1);  mul_2 = getitem_1 = None
        mul_3: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_10: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_8, mul_3);  mul_8 = None
        sum_7: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_10, [2], True);  mul_10 = None
        mul_11: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_3, sum_7);  sum_7 = None
        sub_5: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_9, sum_6);  mul_9 = sum_6 = None
        sub_6: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_5, mul_11);  sub_5 = mul_11 = None
        div_2: "f32[2, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        mul_12: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(div_2, sub_6);  div_2 = sub_6 = None
        mul_13: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(view_9, mul_3);  mul_3 = None
        sum_8: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_13, [0, 1]);  mul_13 = None
        sum_9: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_9, [0, 1]);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_15: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add, 0.5);  add = None
        mul_16: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1, view_1)
        mul_17: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_16, -0.5);  mul_16 = None
        exp_2: "f32[2, 1024, 768]" = torch.ops.aten.exp.default(mul_17);  mul_17 = None
        mul_18: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
        mul_19: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1, mul_18);  view_1 = mul_18 = None
        add_5: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_15, mul_19);  mul_15 = mul_19 = None
        mul_20: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_12, add_5);  mul_12 = add_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1276 in forward, code: x = self.dense(features)
        view_10: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_20, [2048, 768]);  mul_20 = None
        permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        permute_6: "f32[768, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_2: "f32[2048, 768]" = torch.ops.aten.mm.default(view_10, permute_6);  permute_6 = None
        permute_7: "f32[768, 2048]" = torch.ops.aten.permute.default(view_10, [1, 0])
        mm_3: "f32[768, 768]" = torch.ops.aten.mm.default(permute_7, view);  permute_7 = view = None
        sum_10: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_10, [0], True);  view_10 = None
        view_11: "f32[768]" = torch.ops.aten.reshape.default(sum_10, [768]);  sum_10 = None
        view_12: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(mm_2, [2, 1024, 768]);  mm_2 = None
        return (mm_3, view_11, view_12, sum_8, sum_9, mm_1, view_8, None)
