class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[8008, 2560]", primals_4: "i64[16, 128]", view: "f32[2048, 2560]", add: "f32[16, 128, 8008]", amax: "f32[2048, 1]", log: "f32[2048, 1]", convert_element_type: "f32[]", tangents_1: "f32[]", tangents_2: "f32[16, 128, 8008]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:892 in torch_dynamo_resume_in_forward_at_872, code: masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        div_1: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type);  tangents_1 = convert_element_type = None
        view_3: "i64[2048]" = torch.ops.aten.reshape.default(primals_4, [-1]);  primals_4 = None
        unsqueeze_1: "i64[2048, 1]" = torch.ops.aten.unsqueeze.default(view_3, 1);  view_3 = None
        ne_3: "b8[2048, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_1, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "i64[2048, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_1, full_default);  unsqueeze_1 = full_default = None

        # No stacktrace found for following nodes
        iota_default: "i64[8008]" = torch.ops.prims.iota.default(8008, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 8008]" = torch.ops.aten.reshape.default(iota_default, [1, 8008]);  iota_default = None
        expand_default: "i64[2048, 8008]" = torch.ops.aten.expand.default(where_2, [2048, 8008]);  where_2 = None
        eq_tensor: "b8[2048, 8008]" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:892 in torch_dynamo_resume_in_forward_at_872, code: masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        where_self: "f32[2048, 8008]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "f32[2048, 1]" = torch.ops.aten.where.self(ne_3, div_1, full_default_1);  ne_3 = div_1 = full_default_1 = None
        mul: "f32[2048, 8008]" = torch.ops.aten.mul.Tensor(where_self, where_3);  where_self = where_3 = None
        view_2: "f32[2048, 8008]" = torch.ops.aten.reshape.default(add, [-1, 8008]);  add = None
        sub: "f32[2048, 8008]" = torch.ops.aten.sub.Tensor(view_2, amax);  view_2 = amax = None
        sub_1: "f32[2048, 8008]" = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
        exp_1: "f32[2048, 8008]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_4: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(mul, [1], True)
        mul_1: "f32[2048, 8008]" = torch.ops.aten.mul.Tensor(exp_1, sum_4);  exp_1 = sum_4 = None
        sub_2: "f32[2048, 8008]" = torch.ops.aten.sub.Tensor(mul, mul_1);  mul = mul_1 = None
        view_4: "f32[16, 128, 8008]" = torch.ops.aten.reshape.default(sub_2, [16, 128, 8008]);  sub_2 = None
        add_1: "f32[16, 128, 8008]" = torch.ops.aten.add.Tensor(tangents_2, view_4);  tangents_2 = view_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:885 in torch_dynamo_resume_in_forward_at_872, code: lm_logits = self.lm_head(outputs[0])
        view_5: "f32[2048, 8008]" = torch.ops.aten.reshape.default(add_1, [2048, 8008]);  add_1 = None
        permute_1: "f32[8008, 2048]" = torch.ops.aten.permute.default(view_5, [1, 0])
        mm_1: "f32[8008, 2560]" = torch.ops.aten.mm.default(permute_1, view);  permute_1 = view = None
        permute: "f32[2560, 8008]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        permute_3: "f32[8008, 2560]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_2: "f32[2048, 2560]" = torch.ops.aten.mm.default(view_5, permute_3);  view_5 = permute_3 = None
        view_6: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(mm_2, [16, 128, 2560]);  mm_2 = None
        return (mm_1, view_6, None, None)
