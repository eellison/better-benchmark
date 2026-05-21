class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[8, 4096, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1780 in forward, code: hidden_states = torch.cat([hidden_states, hidden_states], dim=-1)
        slice_1: "f32[8, 4096, 256]" = torch.ops.aten.slice.Tensor(tangents_1, 2, 0, 256)
        slice_2: "f32[8, 4096, 256]" = torch.ops.aten.slice.Tensor(tangents_1, 2, 256, 512);  tangents_1 = None
        add: "f32[8, 4096, 256]" = torch.ops.aten.add.Tensor(slice_1, slice_2);  slice_1 = slice_2 = None
        return (add,)
