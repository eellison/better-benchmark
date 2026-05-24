import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[8, 1024, 1024]", primals_2: "f32[50265, 1024]", primals_3: "i64[8, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:1294 in torch_dynamo_resume_in_forward_at_1280, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        permute: "f32[1024, 50265]" = torch.ops.aten.permute.default(primals_2, [1, 0])
        view: "f32[8192, 1024]" = torch.ops.aten.reshape.default(primals_1, [8192, 1024]);  primals_1 = None
        constant_pad_nd_default_3: "f32[1024, 50268]" = torch.ops.aten.constant_pad_nd.default(permute, [0, 3, 0, 0]);  permute = None
        mm_default_2: "f32[8192, 50268]" = torch.ops.aten.mm.default(view, constant_pad_nd_default_3);  constant_pad_nd_default_3 = None
        slice_tensor_1: "f32[8192, 50265]" = torch.ops.aten.slice.Tensor(mm_default_2, 1, 0, -3);  mm_default_2 = None
        view_1: "f32[8, 1024, 50265]" = torch.ops.aten.reshape.default(slice_tensor_1, [8, 1024, 50265]);  slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:1300 in torch_dynamo_resume_in_forward_at_1280, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        view_2: "f32[8192, 50265]" = torch.ops.aten.reshape.default(view_1, [-1, 50265])
        view_3: "i64[8192]" = torch.ops.aten.reshape.default(primals_3, [-1])
        amax: "f32[8192, 1]" = torch.ops.aten.amax.default(view_2, [1], True)
        sub: "f32[8192, 50265]" = torch.ops.aten.sub.Tensor(view_2, amax);  view_2 = None
        exp: "f32[8192, 50265]" = torch.ops.aten.exp.default(sub)
        sum_1: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[8192, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_1: "f32[8192, 50265]" = torch.ops.aten.sub.Tensor(sub, log);  sub = None
        ne: "b8[8192]" = torch.ops.aten.ne.Scalar(view_3, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[8192]" = torch.ops.aten.where.self(ne, view_3, full_default);  view_3 = full_default = None
        unsqueeze: "i64[8192, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[8192, 1]" = torch.ops.aten.gather.default(sub_1, 1, unsqueeze);  sub_1 = unsqueeze = None
        squeeze: "f32[8192]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[8192]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[8192]" = torch.ops.aten.where.self(ne, neg, full_default_1);  neg = full_default_1 = None
        sum_2: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
        div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type);  sum_3 = None
        return (div, view_1, primals_2, primals_3, view, view_1, amax, log, convert_element_type)
