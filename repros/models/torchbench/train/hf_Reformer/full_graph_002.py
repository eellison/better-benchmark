class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[8, 4096, 256]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1780 in forward, code: hidden_states = torch.cat([hidden_states, hidden_states], dim=-1)
        unsqueeze: "f32[8, 4096, 1, 256]" = torch.ops.aten.unsqueeze.default(primals_1, 2);  primals_1 = None
        expand: "f32[8, 4096, 2, 256]" = torch.ops.aten.expand.default(unsqueeze, [8, 4096, 2, 256]);  unsqueeze = None
        clone: "f32[8, 4096, 2, 256]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view: "f32[8, 4096, 512]" = torch.ops.aten.reshape.default(clone, [8, 4096, 512]);  clone = None
        return (view,)
