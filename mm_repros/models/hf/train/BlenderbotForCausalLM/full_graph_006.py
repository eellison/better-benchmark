class GraphModule(torch.nn.Module):
    def forward(self, primals_3: "i64[32, 128][128, 1]cuda:0", view: "bf16[4096, 2560][2560, 1]cuda:0", view_1: "bf16[32, 128, 8008][1025024, 8008, 1]cuda:0", amax: "f32[4096, 1][1, 1]cuda:0", log: "f32[4096, 1][1, 1]cuda:0", convert_element_type_7: "f32[][]cuda:0", permute_3: "bf16[8008, 2560][2560, 1]cuda:0", tangents_1: "f32[][]cuda:0", tangents_2: "bf16[32, 128, 8008][1025024, 8008, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:1004 in torch_dynamo_resume_in_forward_at_984, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        div_1: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_7);  tangents_1 = convert_element_type_7 = None
        view_3: "i64[4096][1]cuda:0" = torch.ops.aten.reshape.default(primals_3, [-1]);  primals_3 = None
        unsqueeze_1: "i64[4096, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_3, 1);  view_3 = None
        ne_3: "b8[4096, 1][1, 1]cuda:0" = torch.ops.aten.ne.Scalar(unsqueeze_1, -100)
        full_default: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "i64[4096, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_3, unsqueeze_1, full_default);  unsqueeze_1 = full_default = None

        # No stacktrace found for following nodes
        iota_default: "i64[8008][1]cuda:0" = torch.ops.prims.iota.default(8008, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 8008][8008, 1]cuda:0" = torch.ops.aten.reshape.default(iota_default, [1, 8008]);  iota_default = None
        expand_default: "i64[4096, 8008][1, 0]cuda:0" = torch.ops.aten.expand.default(where_2, [4096, 8008]);  where_2 = None
        eq_tensor: "b8[4096, 8008][8008, 1]cuda:0" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:1004 in torch_dynamo_resume_in_forward_at_984, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        where_self: "f32[4096, 8008][8008, 1]cuda:0" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        full_default_1: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_3, div_1, full_default_1);  ne_3 = div_1 = full_default_1 = None
        mul: "f32[4096, 8008][8008, 1]cuda:0" = torch.ops.aten.mul.Tensor(where_self, where_3);  where_self = where_3 = None
        convert_element_type_8: "bf16[4096, 8008][8008, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        convert_element_type_9: "f32[4096, 8008][8008, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_8, torch.float32);  convert_element_type_8 = None
        view_2: "bf16[4096, 8008][8008, 1]cuda:0" = torch.ops.aten.reshape.default(view_1, [-1, 8008]);  view_1 = None
        convert_element_type_4: "f32[4096, 8008][8008, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        sub: "f32[4096, 8008][8008, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_4, amax);  convert_element_type_4 = amax = None
        sub_1: "f32[4096, 8008][8008, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
        convert_element_type_5: "bf16[4096, 8008][8008, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        convert_element_type_6: "f32[4096, 8008][8008, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_5, torch.float32);  convert_element_type_5 = None
        exp_1: "f32[4096, 8008][8008, 1]cuda:0" = torch.ops.aten.exp.default(convert_element_type_6);  convert_element_type_6 = None
        sum_4: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_9, [1], True)
        mul_1: "f32[4096, 8008][8008, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_1, sum_4);  exp_1 = sum_4 = None
        sub_2: "f32[4096, 8008][8008, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_9, mul_1);  convert_element_type_9 = mul_1 = None
        convert_element_type_11: "bf16[4096, 8008][8008, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_2, torch.bfloat16);  sub_2 = None
        view_4: "bf16[32, 128, 8008][1025024, 8008, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_11, [32, 128, 8008]);  convert_element_type_11 = None
        add: "bf16[32, 128, 8008][1025024, 8008, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_2, view_4);  tangents_2 = view_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:998 in torch_dynamo_resume_in_forward_at_984, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        view_5: "bf16[4096, 8008][8008, 1]cuda:0" = torch.ops.aten.reshape.default(add, [4096, 8008]);  add = None
        permute_1: "bf16[8008, 4096][1, 8008]cuda:0" = torch.ops.aten.permute.default(view_5, [1, 0])
        mm_1: "bf16[8008, 2560][2560, 1]cuda:0" = torch.ops.aten.mm.default(permute_1, view);  permute_1 = view = None
        mm_2: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.mm.default(view_5, permute_3);  view_5 = permute_3 = None
        view_6: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [32, 128, 2560]);  mm_2 = None
        convert_element_type_16: "f32[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None
        convert_element_type_17: "f32[8008, 2560][2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        return (convert_element_type_16, convert_element_type_17, None)
