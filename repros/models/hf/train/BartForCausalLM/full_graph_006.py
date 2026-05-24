import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "f32[50265, 1024]", primals_3: "i64[8, 1024]", view: "f32[8192, 1024]", view_1: "f32[8, 1024, 50265]", amax: "f32[8192, 1]", log: "f32[8192, 1]", convert_element_type: "f32[]", tangents_1: "f32[]", tangents_2: "f32[8, 1024, 50265]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:1300 in torch_dynamo_resume_in_forward_at_1280, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        div_1: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type);  tangents_1 = convert_element_type = None
        view_3: "i64[8192]" = torch.ops.aten.reshape.default(primals_3, [-1]);  primals_3 = None
        unsqueeze_1: "i64[8192, 1]" = torch.ops.aten.unsqueeze.default(view_3, 1);  view_3 = None
        ne_3: "b8[8192, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_1, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "i64[8192, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_1, full_default);  unsqueeze_1 = full_default = None

        # No stacktrace found for following nodes
        iota_default: "i64[50265]" = torch.ops.prims.iota.default(50265, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 50265]" = torch.ops.aten.reshape.default(iota_default, [1, 50265]);  iota_default = None
        expand_default: "i64[8192, 50265]" = torch.ops.aten.expand.default(where_2, [8192, 50265]);  where_2 = None
        eq_tensor: "b8[8192, 50265]" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:1300 in torch_dynamo_resume_in_forward_at_1280, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        where_self: "f32[8192, 50265]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "f32[8192, 1]" = torch.ops.aten.where.self(ne_3, div_1, full_default_1);  ne_3 = div_1 = full_default_1 = None
        mul: "f32[8192, 50265]" = torch.ops.aten.mul.Tensor(where_self, where_3);  where_self = where_3 = None
        view_2: "f32[8192, 50265]" = torch.ops.aten.reshape.default(view_1, [-1, 50265]);  view_1 = None
        sub: "f32[8192, 50265]" = torch.ops.aten.sub.Tensor(view_2, amax);  view_2 = amax = None
        sub_1: "f32[8192, 50265]" = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
        exp_1: "f32[8192, 50265]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_4: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(mul, [1], True)
        mul_1: "f32[8192, 50265]" = torch.ops.aten.mul.Tensor(exp_1, sum_4);  exp_1 = sum_4 = None
        sub_2: "f32[8192, 50265]" = torch.ops.aten.sub.Tensor(mul, mul_1);  mul = mul_1 = None
        view_4: "f32[8, 1024, 50265]" = torch.ops.aten.reshape.default(sub_2, [8, 1024, 50265]);  sub_2 = None
        add: "f32[8, 1024, 50265]" = torch.ops.aten.add.Tensor(tangents_2, view_4);  tangents_2 = view_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:1294 in torch_dynamo_resume_in_forward_at_1280, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        view_5: "f32[8192, 50265]" = torch.ops.aten.reshape.default(add, [8192, 50265]);  add = None
        permute_1: "f32[50265, 8192]" = torch.ops.aten.permute.default(view_5, [1, 0])
        constant_pad_nd_default_2: "f32[50268, 8192]" = torch.ops.aten.constant_pad_nd.default(permute_1, [0, 0, 0, 3]);  permute_1 = None
        mm_default_1: "f32[50268, 1024]" = torch.ops.aten.mm.default(constant_pad_nd_default_2, view);  constant_pad_nd_default_2 = view = None
        slice_tensor: "f32[50265, 1024]" = torch.ops.aten.slice.Tensor(mm_default_1, 0, 0, -3);  mm_default_1 = None
        permute: "f32[1024, 50265]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        permute_3: "f32[50265, 1024]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        constant_pad_nd_default: "f32[8192, 50268]" = torch.ops.aten.constant_pad_nd.default(view_5, [0, 3, 0, 0]);  view_5 = None
        constant_pad_nd_default_1: "f32[50268, 1024]" = torch.ops.aten.constant_pad_nd.default(permute_3, [0, 0, 0, 3]);  permute_3 = None
        mm_default: "f32[8192, 1024]" = torch.ops.aten.mm.default(constant_pad_nd_default, constant_pad_nd_default_1);  constant_pad_nd_default = constant_pad_nd_default_1 = None
        view_6: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(mm_default, [8, 1024, 1024]);  mm_default = None
        return (view_6, slice_tensor, None)
