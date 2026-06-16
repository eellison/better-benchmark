class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[16, 1024, 768][786432, 768, 1]cuda:0", primals_2: "f32[50005, 768][768, 1]cuda:0", primals_3: "i64[16, 1024][1024, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:1125 in torch_dynamo_resume_in_forward_at_1111, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        convert_element_type: "bf16[50005, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
        convert_element_type_1: "bf16[16, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_1, torch.bfloat16);  primals_1 = None
        permute: "bf16[768, 50005][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type, [1, 0]);  convert_element_type = None
        view: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1, [16384, 768]);  convert_element_type_1 = None
        constant_pad_nd_default_3: "bf16[768, 50008][50008, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(permute, [0, 3, 0, 0])
        mm_default_2: "bf16[16384, 50008][50008, 1]cuda:0" = torch.ops.aten.mm.default(view, constant_pad_nd_default_3);  constant_pad_nd_default_3 = None
        slice_tensor_1: "bf16[16384, 50005][50008, 1]cuda:0" = torch.ops.aten.slice.Tensor(mm_default_2, 1, 0, -3);  mm_default_2 = None
        view_1: "bf16[16, 1024, 50005][51208192, 50008, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_1, [16, 1024, 50005]);  slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:1131 in torch_dynamo_resume_in_forward_at_1111, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        view_2: "bf16[16384, 50005][50008, 1]cuda:0" = torch.ops.aten.reshape.default(view_1, [-1, 50005])
        view_3: "i64[16384][1]cuda:0" = torch.ops.aten.reshape.default(primals_3, [-1])
        convert_element_type_4: "f32[16384, 50005][50005, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        amax: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_4, [1], True)
        sub: "f32[16384, 50005][50005, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_4, amax);  convert_element_type_4 = None
        exp: "f32[16384, 50005][50005, 1]cuda:0" = torch.ops.aten.exp.default(sub)
        sum_1: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_1: "f32[16384, 50005][50005, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub, log);  sub = None
        convert_element_type_5: "bf16[16384, 50005][50005, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        convert_element_type_6: "f32[16384, 50005][50005, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_5, torch.float32);  convert_element_type_5 = None
        ne: "b8[16384][1]cuda:0" = torch.ops.aten.ne.Scalar(view_3, -100)
        full_default: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[16384][1]cuda:0" = torch.ops.aten.where.self(ne, view_3, full_default);  view_3 = full_default = None
        unsqueeze: "i64[16384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.gather.default(convert_element_type_6, 1, unsqueeze);  convert_element_type_6 = unsqueeze = None
        squeeze: "f32[16384][1]cuda:0" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[16384][1]cuda:0" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_1: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[16384][1]cuda:0" = torch.ops.aten.where.self(ne, neg, full_default_1);  neg = full_default_1 = None
        sum_2: "i64[][]cuda:0" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type_7: "f32[][]cuda:0" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3: "f32[][]cuda:0" = torch.ops.aten.sum.default(where_1);  where_1 = None
        div: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_7);  sum_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:1125 in torch_dynamo_resume_in_forward_at_1111, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        permute_3: "bf16[50005, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (div, view_1, primals_3, view, view_1, amax, log, convert_element_type_7, permute_3)
