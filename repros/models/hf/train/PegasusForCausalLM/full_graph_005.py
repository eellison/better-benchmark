class GraphModule(torch.nn.Module):
    def forward(self, primals_3: "i64[128, 128][128, 1]cuda:0", view: "bf16[16384, 1024][1024, 1]cuda:0", view_1: "bf16[128, 128, 50265][6434816, 50272, 1]cuda:0", amax: "f32[16384, 1][1, 1]cuda:0", log: "f32[16384, 1][1, 1]cuda:0", convert_element_type_7: "f32[][]cuda:0", permute_3: "bf16[50265, 1024][1024, 1]cuda:0", tangents_1: "f32[][]cuda:0", tangents_2: "bf16[128, 128, 50265][6433920, 50265, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:1120 in torch_dynamo_resume_in_forward_at_1100, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        div_1: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_7);  tangents_1 = convert_element_type_7 = None
        view_3: "i64[16384][1]cuda:0" = torch.ops.aten.reshape.default(primals_3, [-1]);  primals_3 = None
        unsqueeze_1: "i64[16384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_3, 1);  view_3 = None
        ne_3: "b8[16384, 1][1, 1]cuda:0" = torch.ops.aten.ne.Scalar(unsqueeze_1, -100)
        full_default: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "i64[16384, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_3, unsqueeze_1, full_default);  unsqueeze_1 = full_default = None

        # No stacktrace found for following nodes
        iota_default: "i64[50265][1]cuda:0" = torch.ops.prims.iota.default(50265, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 50265][50265, 1]cuda:0" = torch.ops.aten.reshape.default(iota_default, [1, 50265]);  iota_default = None
        expand_default: "i64[16384, 50265][1, 0]cuda:0" = torch.ops.aten.expand.default(where_2, [16384, 50265]);  where_2 = None
        eq_tensor: "b8[16384, 50265][50265, 1]cuda:0" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:1120 in torch_dynamo_resume_in_forward_at_1100, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        where_self: "f32[16384, 50265][50265, 1]cuda:0" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        full_default_1: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_3, div_1, full_default_1);  ne_3 = div_1 = full_default_1 = None
        mul: "f32[16384, 50265][50265, 1]cuda:0" = torch.ops.aten.mul.Tensor(where_self, where_3);  where_self = where_3 = None
        convert_element_type_8: "bf16[16384, 50265][50265, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        convert_element_type_9: "f32[16384, 50265][50265, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_8, torch.float32);  convert_element_type_8 = None
        view_2: "bf16[16384, 50265][50272, 1]cuda:0" = torch.ops.aten.reshape.default(view_1, [-1, 50265]);  view_1 = None
        convert_element_type_4: "f32[16384, 50265][50265, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        sub: "f32[16384, 50265][50265, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_4, amax);  convert_element_type_4 = amax = None
        sub_1: "f32[16384, 50265][50265, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
        convert_element_type_5: "bf16[16384, 50265][50265, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        convert_element_type_6: "f32[16384, 50265][50265, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_5, torch.float32);  convert_element_type_5 = None
        exp_1: "f32[16384, 50265][50265, 1]cuda:0" = torch.ops.aten.exp.default(convert_element_type_6);  convert_element_type_6 = None
        sum_4: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_9, [1], True)
        mul_1: "f32[16384, 50265][50265, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_1, sum_4);  exp_1 = sum_4 = None
        sub_2: "f32[16384, 50265][50265, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_9, mul_1);  convert_element_type_9 = mul_1 = None
        convert_element_type_11: "bf16[16384, 50265][50265, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_2, torch.bfloat16);  sub_2 = None
        view_4: "bf16[128, 128, 50265][6433920, 50265, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_11, [128, 128, 50265]);  convert_element_type_11 = None
        add: "bf16[128, 128, 50265][6433920, 50265, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_2, view_4);  tangents_2 = view_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/pegasus/modeling_pegasus.py:1114 in torch_dynamo_resume_in_forward_at_1100, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        view_5: "bf16[16384, 50265][50265, 1]cuda:0" = torch.ops.aten.reshape.default(add, [16384, 50265]);  add = None
        permute_1: "bf16[50265, 16384][1, 50265]cuda:0" = torch.ops.aten.permute.default(view_5, [1, 0])
        constant_pad_nd_default_2: "bf16[50272, 16384][16384, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(permute_1, [0, 0, 0, 7]);  permute_1 = None
        mm_default_1: "bf16[50272, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(constant_pad_nd_default_2, view);  constant_pad_nd_default_2 = view = None
        slice_tensor: "bf16[50265, 1024][1024, 1]cuda:0" = torch.ops.aten.slice.Tensor(mm_default_1, 0, 0, -7);  mm_default_1 = None
        constant_pad_nd_default: "bf16[16384, 50272][50272, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_5, [0, 7, 0, 0]);  view_5 = None
        constant_pad_nd_default_1: "bf16[50272, 1024][1024, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(permute_3, [0, 0, 0, 7]);  permute_3 = None
        mm_default: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(constant_pad_nd_default, constant_pad_nd_default_1);  constant_pad_nd_default = constant_pad_nd_default_1 = None
        view_6: "bf16[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_default, [128, 128, 1024]);  mm_default = None
        convert_element_type_16: "f32[128, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None
        convert_element_type_17: "f32[50265, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(slice_tensor, torch.float32);  slice_tensor = None
        return (convert_element_type_16, convert_element_type_17, None)
