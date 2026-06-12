class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[32, 128, 1024][131072, 1024, 1]cuda:0", primals_2: "f32[256008, 1024][1024, 1]cuda:0", primals_3: "i64[32, 128][128, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:556 in torch_dynamo_resume_in_forward_at_541, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        convert_element_type: "bf16[256008, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
        convert_element_type_1: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_1, torch.bfloat16);  primals_1 = None
        permute: "bf16[1024, 256008][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type, [1, 0]);  convert_element_type = None
        view: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1, [4096, 1024]);  convert_element_type_1 = None
        mm: "bf16[4096, 256008][256008, 1]cuda:0" = torch.ops.aten.mm.default(view, permute)
        view_1: "bf16[32, 128, 256008][32769024, 256008, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [32, 128, 256008]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:56 in ForCausalLMLoss, code: logits = logits.float()
        convert_element_type_4: "f32[32, 128, 256008][32769024, 256008, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1, torch.float32)

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "i64[32, 129][129, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(primals_3, [0, 1], -100.0);  primals_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:61 in ForCausalLMLoss, code: shift_labels = labels[..., 1:].contiguous()
        slice_1: "i64[32, 128][129, 1]cuda:0" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807)
        clone: "i64[32, 128][128, 1]cuda:0" = torch.ops.aten.clone.default(slice_1, memory_format = torch.contiguous_format);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_2: "f32[4096, 256008][256008, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_4, [-1, 256008]);  convert_element_type_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:65 in ForCausalLMLoss, code: shift_labels = shift_labels.view(-1)
        view_3: "i64[4096][1]cuda:0" = torch.ops.aten.reshape.default(clone, [-1]);  clone = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        amax: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(view_2, [1], True)
        sub: "f32[4096, 256008][256008, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_2, amax);  view_2 = None
        exp: "f32[4096, 256008][256008, 1]cuda:0" = torch.ops.aten.exp.default(sub)
        sum_1: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_1: "f32[4096, 256008][256008, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub, log);  sub = None
        ne: "b8[4096][1]cuda:0" = torch.ops.aten.ne.Scalar(view_3, -100)
        full_default: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[4096][1]cuda:0" = torch.ops.aten.where.self(ne, view_3, full_default);  view_3 = full_default = None
        unsqueeze: "i64[4096, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.gather.default(sub_1, 1, unsqueeze);  sub_1 = unsqueeze = None
        squeeze: "f32[4096][1]cuda:0" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[4096][1]cuda:0" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_1: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[4096][1]cuda:0" = torch.ops.aten.where.self(ne, neg, full_default_1);  neg = full_default_1 = None
        sum_2: "i64[][]cuda:0" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type_5: "f32[][]cuda:0" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3: "f32[][]cuda:0" = torch.ops.aten.sum.default(where_1);  where_1 = None
        div: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_5);  sum_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:556 in torch_dynamo_resume_in_forward_at_541, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        permute_3: "bf16[256008, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (div, view_1, view, view_1, constant_pad_nd, amax, log, convert_element_type_5, permute_3)
