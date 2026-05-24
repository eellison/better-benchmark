import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[256, 384]", primals_2: "f32[256]", primals_3: "f32[8, 1500, 384]", primals_4: "f32[2, 256]", primals_5: "f32[2]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:1333 in torch_dynamo_resume_in_forward_at_1314, code: hidden_states = self.projector(hidden_states)
        view: "f32[12000, 384]" = torch.ops.aten.reshape.default(primals_3, [12000, 384]);  primals_3 = None
        permute: "f32[384, 256]" = torch.ops.aten.permute.default(primals_1, [1, 0])
        addmm: "f32[12000, 256]" = torch.ops.aten.addmm.default(primals_2, view, permute);  primals_2 = permute = None
        view_1: "f32[8, 1500, 256]" = torch.ops.aten.reshape.default(addmm, [8, 1500, 256]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:1334 in torch_dynamo_resume_in_forward_at_1314, code: pooled_output = hidden_states.mean(dim=1)
        mean: "f32[8, 256]" = torch.ops.aten.mean.dim(view_1, [1]);  view_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:1336 in torch_dynamo_resume_in_forward_at_1314, code: logits = self.classifier(pooled_output)
        permute_1: "f32[256, 2]" = torch.ops.aten.permute.default(primals_4, [1, 0])
        addmm_1: "f32[8, 2]" = torch.ops.aten.addmm.default(primals_5, mean, permute_1);  primals_5 = permute_1 = None
        return (addmm_1, primals_1, primals_4, view, mean)
