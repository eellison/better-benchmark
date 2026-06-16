class GraphModule(torch.nn.Module):
    def forward(self, view: "bf16[8192, 768][768, 1]cuda:0", view_1: "bf16[4, 2048, 50272][102957056, 50272, 1]cuda:0", constant_pad_nd: "i64[4, 2049][2064, 1]cuda:0", amax: "f32[8192, 1][1, 1]cuda:0", log: "f32[8192, 1][1, 1]cuda:0", convert_element_type_5: "f32[][]cuda:0", permute_3: "bf16[50272, 768][768, 1]cuda:0", tangents_1: "f32[][]cuda:0", tangents_2: "bf16[4, 2048, 50272][102957056, 50272, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        div_1: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_5);  tangents_1 = convert_element_type_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:61 in ForCausalLMLoss, code: shift_labels = labels[..., 1:].contiguous()
        slice_1: "i64[4, 2048][2064, 1]cuda:0" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807);  constant_pad_nd = None
        clone: "i64[4, 2048][2048, 1]cuda:0" = torch.ops.aten.clone.default(slice_1, memory_format = torch.contiguous_format);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:65 in ForCausalLMLoss, code: shift_labels = shift_labels.view(-1)
        view_3: "i64[8192][1]cuda:0" = torch.ops.aten.reshape.default(clone, [-1]);  clone = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        unsqueeze_1: "i64[8192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_3, 1);  view_3 = None
        ne_3: "b8[8192, 1][1, 1]cuda:0" = torch.ops.aten.ne.Scalar(unsqueeze_1, -100)
        full_default: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "i64[8192, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_3, unsqueeze_1, full_default);  unsqueeze_1 = full_default = None

        # No stacktrace found for following nodes
        iota_default: "i64[50272][1]cuda:0" = torch.ops.prims.iota.default(50272, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 50272][50272, 1]cuda:0" = torch.ops.aten.reshape.default(iota_default, [1, 50272]);  iota_default = None
        expand_default: "i64[8192, 50272][1, 0]cuda:0" = torch.ops.aten.expand.default(where_2, [8192, 50272]);  where_2 = None
        eq_tensor: "b8[8192, 50272][50272, 1]cuda:0" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        where_self: "f32[8192, 50272][50272, 1]cuda:0" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        full_default_1: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_3, div_1, full_default_1);  ne_3 = div_1 = full_default_1 = None
        mul: "f32[8192, 50272][50272, 1]cuda:0" = torch.ops.aten.mul.Tensor(where_self, where_3);  where_self = where_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:56 in ForCausalLMLoss, code: logits = logits.float()
        convert_element_type_4: "f32[4, 2048, 50272][102957056, 50272, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_2: "f32[8192, 50272][50272, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_4, [-1, 50272]);  convert_element_type_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        sub: "f32[8192, 50272][50272, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_2, amax);  view_2 = amax = None
        sub_1: "f32[8192, 50272][50272, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
        exp_1: "f32[8192, 50272][50272, 1]cuda:0" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_4: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul, [1], True)
        mul_1: "f32[8192, 50272][50272, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_1, sum_4);  exp_1 = sum_4 = None
        sub_2: "f32[8192, 50272][50272, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul, mul_1);  mul = mul_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_4: "f32[4, 2048, 50272][102957056, 50272, 1]cuda:0" = torch.ops.aten.reshape.default(sub_2, [4, 2048, 50272]);  sub_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:56 in ForCausalLMLoss, code: logits = logits.float()
        convert_element_type_6: "bf16[4, 2048, 50272][102957056, 50272, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_4, torch.bfloat16);  view_4 = None
        add: "bf16[4, 2048, 50272][102957056, 50272, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_2, convert_element_type_6);  tangents_2 = convert_element_type_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:512 in torch_dynamo_resume_in_forward_at_499, code: logits = self.lm_head(hidden_states[:, slice_indices, :]).contiguous()
        view_5: "bf16[8192, 50272][50272, 1]cuda:0" = torch.ops.aten.reshape.default(add, [8192, 50272]);  add = None
        permute_1: "bf16[50272, 8192][1, 50272]cuda:0" = torch.ops.aten.permute.default(view_5, [1, 0])
        mm_1: "bf16[50272, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1, view);  permute_1 = view = None
        mm_2: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_5, permute_3);  view_5 = permute_3 = None
        view_6: "bf16[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [4, 2048, 768]);  mm_2 = None
        convert_element_type_11: "f32[4, 2048, 768][1572864, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None
        convert_element_type_12: "f32[50272, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        return (convert_element_type_11, convert_element_type_12, None)
