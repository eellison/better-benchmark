class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[256, 384]", primals_4: "f32[2, 256]", view: "f32[12000, 384]", mean: "f32[8, 256]", tangents_1: "f32[8, 2]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:1336 in torch_dynamo_resume_in_forward_at_1314, code: logits = self.classifier(pooled_output)
        permute_1: "f32[256, 2]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_2: "f32[2, 256]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        mm: "f32[8, 256]" = torch.ops.aten.mm.default(tangents_1, permute_2);  permute_2 = None
        permute_3: "f32[2, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "f32[2, 256]" = torch.ops.aten.mm.default(permute_3, mean);  permute_3 = mean = None
        sum_1: "f32[1, 2]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view_2: "f32[2]" = torch.ops.aten.reshape.default(sum_1, [2]);  sum_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:1334 in torch_dynamo_resume_in_forward_at_1314, code: pooled_output = hidden_states.mean(dim=1)
        unsqueeze: "f32[8, 1, 256]" = torch.ops.aten.unsqueeze.default(mm, 1);  mm = None
        expand: "f32[8, 1500, 256]" = torch.ops.aten.expand.default(unsqueeze, [8, 1500, 256]);  unsqueeze = None
        div: "f32[8, 1500, 256]" = torch.ops.aten.div.Scalar(expand, 1500);  expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:1333 in torch_dynamo_resume_in_forward_at_1314, code: hidden_states = self.projector(hidden_states)
        view_3: "f32[12000, 256]" = torch.ops.aten.reshape.default(div, [12000, 256]);  div = None
        permute: "f32[384, 256]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        permute_6: "f32[256, 384]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_2: "f32[12000, 384]" = torch.ops.aten.mm.default(view_3, permute_6);  permute_6 = None
        permute_7: "f32[256, 12000]" = torch.ops.aten.permute.default(view_3, [1, 0])
        mm_3: "f32[256, 384]" = torch.ops.aten.mm.default(permute_7, view);  permute_7 = view = None
        sum_2: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_3, [0], True);  view_3 = None
        view_4: "f32[256]" = torch.ops.aten.reshape.default(sum_2, [256]);  sum_2 = None
        view_5: "f32[8, 1500, 384]" = torch.ops.aten.reshape.default(mm_2, [8, 1500, 384]);  mm_2 = None
        return (mm_3, view_4, view_5, mm_1, view_2)
