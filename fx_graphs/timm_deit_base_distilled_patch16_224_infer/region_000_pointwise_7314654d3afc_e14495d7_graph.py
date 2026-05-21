class GraphModule(torch.nn.Module):
    def forward(self, addmm_48: "f32[128, 1000]", addmm_49: "f32[128, 1000]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:124 in forward_head, code: return (x + x_dist) / 2
        add_tensor: "f32[128, 1000]" = torch.ops.aten.add.Tensor(addmm_48, addmm_49);  addmm_48 = addmm_49 = None
        div_tensor: "f32[128, 1000]" = torch.ops.aten.div.Tensor(add_tensor, 2);  add_tensor = None
        return div_tensor
