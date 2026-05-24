import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f16[1, 64, 64, 64]", arg1_1: "f16[1, 64, 64, 192]", arg2_1: "i64[1, 4096]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:270 in torch_dynamo_resume_in_forward_at_266, code: position_encodings = torch.cat(
        cat: "f16[1, 64, 64, 256]" = torch.ops.aten.cat.default([arg0_1, arg1_1], -1);  arg0_1 = arg1_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:273 in torch_dynamo_resume_in_forward_at_266, code: position_encodings = torch.reshape(position_encodings, (batch_size, -1, position_encodings.shape[-1]))
        view: "f16[1, 4096, 256]" = torch.ops.aten.reshape.default(cat, [1, -1, 256]);  cat = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:278 in torch_dynamo_resume_in_forward_at_266, code: torch.index_select(position_encodings[i], 0, position_ids[i]).unsqueeze(0)
        select: "f16[4096, 256]" = torch.ops.aten.select.int(view, 0, 0);  view = None
        select_1: "i64[4096]" = torch.ops.aten.select.int(arg2_1, 0, 0);  arg2_1 = None
        index: "f16[4096, 256]" = torch.ops.aten.index.Tensor(select, [select_1]);  select = select_1 = None
        unsqueeze: "f16[1, 4096, 256]" = torch.ops.aten.unsqueeze.default(index, 0);  index = None
        return (unsqueeze,)
