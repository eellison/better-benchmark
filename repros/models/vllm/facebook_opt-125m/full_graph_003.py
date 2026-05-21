class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f16[4, 512, 768]", arg1_1: "f16[50272, 768]", arg2_1: "i64[4, 512]"):
        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "i64[4, 513]" = torch.ops.aten.constant_pad_nd.default(arg2_1, [0, 1], -100.0);  arg2_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:61 in ForCausalLMLoss, code: shift_labels = labels[..., 1:].contiguous()
        slice_1: "i64[4, 512]" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807);  constant_pad_nd = None
        clone: "i64[4, 512]" = torch.ops.aten.clone.default(slice_1, memory_format = torch.contiguous_format);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:65 in ForCausalLMLoss, code: shift_labels = shift_labels.view(-1)
        view_3: "i64[2048]" = torch.ops.aten.reshape.default(clone, [-1]);  clone = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        ne_1: "b8[2048]" = torch.ops.aten.ne.Scalar(view_3, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:512 in torch_dynamo_resume_in_forward_at_499, code: logits = self.lm_head(hidden_states[:, slice_indices, :]).contiguous()
        view: "f16[2048, 768]" = torch.ops.aten.reshape.default(arg0_1, [2048, 768]);  arg0_1 = None
        permute: "f16[768, 50272]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        mm: "f16[2048, 50272]" = torch.ops.aten.mm.default(view, permute);  view = permute = None
        view_1: "f16[4, 512, 50272]" = torch.ops.aten.reshape.default(mm, [4, 512, 50272]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:56 in ForCausalLMLoss, code: logits = logits.float()
        convert_element_type_2: "f32[4, 512, 50272]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_2: "f32[2048, 50272]" = torch.ops.aten.reshape.default(convert_element_type_2, [-1, 50272]);  convert_element_type_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        amax: "f32[2048, 1]" = torch.ops.aten.amax.default(view_2, [1], True)
        sub: "f32[2048, 50272]" = torch.ops.aten.sub.Tensor(view_2, amax);  view_2 = amax = None
        exp: "f32[2048, 50272]" = torch.ops.aten.exp.default(sub)
        sum_1: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[2048, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_1: "f32[2048, 50272]" = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
        ne: "b8[2048]" = torch.ops.aten.ne.Scalar(view_3, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[2048]" = torch.ops.aten.where.self(ne, view_3, full_default);  ne = full_default = None
        unsqueeze: "i64[2048, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[2048, 1]" = torch.ops.aten.gather.default(sub_1, 1, unsqueeze);  sub_1 = unsqueeze = None
        squeeze: "f32[2048]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[2048]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[2048]" = torch.ops.aten.where.self(ne_1, neg, full_default_1);  ne_1 = neg = full_default_1 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
        ne_2: "b8[2048]" = torch.ops.aten.ne.Scalar(view_3, -100);  view_3 = None
        sum_2: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_3: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_3);  sum_3 = convert_element_type_3 = None
        return (div, view_1)
