class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[32, 128, 2560][327680, 2560, 1]cuda:0", primals_2: "f32[8008, 2560][2560, 1]cuda:0", primals_3: "i64[32, 128][128, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:998 in torch_dynamo_resume_in_forward_at_984, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        convert_element_type: "bf16[8008, 2560][2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
        convert_element_type_1: "bf16[32, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_1, torch.bfloat16);  primals_1 = None
        permute: "bf16[2560, 8008][1, 2560]cuda:0" = torch.ops.aten.permute.default(convert_element_type, [1, 0]);  convert_element_type = None
        view: "bf16[4096, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1, [4096, 2560]);  convert_element_type_1 = None
        mm: "bf16[4096, 8008][8008, 1]cuda:0" = torch.ops.aten.mm.default(view, permute)
        view_1: "bf16[32, 128, 8008][1025024, 8008, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [32, 128, 8008]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:1004 in torch_dynamo_resume_in_forward_at_984, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        view_2: "bf16[4096, 8008][8008, 1]cuda:0" = torch.ops.aten.reshape.default(view_1, [-1, 8008])
        view_3: "i64[4096][1]cuda:0" = torch.ops.aten.reshape.default(primals_3, [-1])
        convert_element_type_4: "f32[4096, 8008][8008, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        amax: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_4, [1], True)
        sub: "f32[4096, 8008][8008, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_4, amax);  convert_element_type_4 = None
        exp: "f32[4096, 8008][8008, 1]cuda:0" = torch.ops.aten.exp.default(sub)
        sum_1: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_1: "f32[4096, 8008][8008, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub, log);  sub = None
        convert_element_type_5: "bf16[4096, 8008][8008, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        convert_element_type_6: "f32[4096, 8008][8008, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_5, torch.float32);  convert_element_type_5 = None
        ne: "b8[4096][1]cuda:0" = torch.ops.aten.ne.Scalar(view_3, -100)
        full_default: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[4096][1]cuda:0" = torch.ops.aten.where.self(ne, view_3, full_default);  view_3 = full_default = None
        unsqueeze: "i64[4096, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.gather.default(convert_element_type_6, 1, unsqueeze);  convert_element_type_6 = unsqueeze = None
        squeeze: "f32[4096][1]cuda:0" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[4096][1]cuda:0" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_1: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[4096][1]cuda:0" = torch.ops.aten.where.self(ne, neg, full_default_1);  neg = full_default_1 = None
        sum_2: "i64[][]cuda:0" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type_7: "f32[][]cuda:0" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3: "f32[][]cuda:0" = torch.ops.aten.sum.default(where_1);  where_1 = None
        div: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_7);  sum_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:998 in torch_dynamo_resume_in_forward_at_984, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        permute_3: "bf16[8008, 2560][2560, 1]cuda:0" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (div, view_1, primals_3, view, view_1, amax, log, convert_element_type_7, permute_3)
