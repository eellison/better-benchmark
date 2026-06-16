class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[768, 768][768, 1]cuda:0", primals_2: "f32[768][1]cuda:0", primals_3: "f32[8, 1024, 768][786432, 768, 1]cuda:0", primals_4: "f32[768][1]cuda:0", primals_5: "f32[768][1]cuda:0", primals_6: "f32[50265, 768][768, 1]cuda:0", primals_7: "f32[50265][1]cuda:0", primals_8: "i64[8, 1024][1024, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1276 in forward, code: x = self.dense(features)
        convert_element_type: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
        convert_element_type_1: "bf16[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_1, torch.bfloat16);  primals_1 = None
        convert_element_type_2: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_3, torch.bfloat16);  primals_3 = None
        view: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2, [8192, 768]);  convert_element_type_2 = None
        permute: "bf16[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type, view, permute);  convert_element_type = None
        view_1: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [8, 1024, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_6: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        mul: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.5)
        mul_1: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.7071067811865476);  convert_element_type_6 = None
        erf: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.erf.default(mul_1);  mul_1 = None
        add: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_2: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, add);  mul = add = None
        convert_element_type_7: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1278 in forward, code: x = self.layer_norm(x)
        convert_element_type_8: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_7, torch.float32);  convert_element_type_7 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_8, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        add_1: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_8, getitem_1);  convert_element_type_8 = None
        mul_3: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_4: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, primals_4);  mul_3 = None
        add_2: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_4, primals_5);  mul_4 = primals_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1281 in forward, code: x = self.decoder(x)
        convert_element_type_9: "bf16[50265][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_7, torch.bfloat16);  primals_7 = None
        convert_element_type_10: "bf16[50265, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_6, torch.bfloat16);  primals_6 = None
        convert_element_type_11: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        view_2: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_11, [8192, 768]);  convert_element_type_11 = None
        permute_1: "bf16[768, 50265][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_10, [1, 0]);  convert_element_type_10 = None
        constant_pad_nd_default_3: "bf16[768, 50272][50272, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(permute_1, [0, 7, 0, 0])
        full_default_5: "bf16[7][1]cuda:0" = torch.ops.aten.full.default([7], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default: "bf16[50272][1]cuda:0" = torch.ops.aten.cat.default([convert_element_type_9, full_default_5]);  convert_element_type_9 = full_default_5 = None
        addmm_default: "bf16[8192, 50272][50272, 1]cuda:0" = torch.ops.aten.addmm.default(cat_default, view_2, constant_pad_nd_default_3);  cat_default = constant_pad_nd_default_3 = None
        slice_tensor_1: "bf16[8192, 50265][50272, 1]cuda:0" = torch.ops.aten.slice.Tensor(addmm_default, 1, 0, -7);  addmm_default = None
        view_3: "bf16[8, 1024, 50265][51478528, 50272, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_1, [8, 1024, 50265]);  slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1635 in torch_dynamo_resume_in_forward_at_1616, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        view_4: "bf16[8192, 50265][50272, 1]cuda:0" = torch.ops.aten.reshape.default(view_3, [-1, 50265])
        view_5: "i64[8192][1]cuda:0" = torch.ops.aten.reshape.default(primals_8, [-1])
        convert_element_type_15: "f32[8192, 50265][50265, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        amax: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_15, [1], True)
        sub_1: "f32[8192, 50265][50265, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_15, amax);  convert_element_type_15 = None
        exp: "f32[8192, 50265][50265, 1]cuda:0" = torch.ops.aten.exp.default(sub_1)
        sum_1: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_2: "f32[8192, 50265][50265, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1, log);  sub_1 = None
        convert_element_type_16: "bf16[8192, 50265][50265, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_2, torch.bfloat16);  sub_2 = None
        convert_element_type_17: "f32[8192, 50265][50265, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_16, torch.float32);  convert_element_type_16 = None
        ne: "b8[8192][1]cuda:0" = torch.ops.aten.ne.Scalar(view_5, -100)
        full_default: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[8192][1]cuda:0" = torch.ops.aten.where.self(ne, view_5, full_default);  view_5 = full_default = None
        unsqueeze: "i64[8192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.gather.default(convert_element_type_17, 1, unsqueeze);  convert_element_type_17 = unsqueeze = None
        squeeze: "f32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[8192][1]cuda:0" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_1: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[8192][1]cuda:0" = torch.ops.aten.where.self(ne, neg, full_default_1);  neg = full_default_1 = None
        sum_2: "i64[][]cuda:0" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type_18: "f32[][]cuda:0" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3: "f32[][]cuda:0" = torch.ops.aten.sum.default(where_1);  where_1 = None
        div: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_18);  sum_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1281 in forward, code: x = self.decoder(x)
        permute_2: "bf16[50265, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1276 in forward, code: x = self.dense(features)
        permute_6: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (div, view_3, primals_4, primals_8, view, addmm, getitem_1, rsqrt, view_2, view_3, amax, log, convert_element_type_18, permute_2, permute_6)
