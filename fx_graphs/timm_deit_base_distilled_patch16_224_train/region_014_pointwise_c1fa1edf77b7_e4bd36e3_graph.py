class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[128, 1000]", primals_155: "f32[1000, 768]", primals_153: "f32[1000, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:124 in forward_head, code: return (x + x_dist) / 2
        div_tensor: "f32[128, 1000]" = torch.ops.aten.div.Tensor(tangents_1, 2);  tangents_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:118 in forward_head, code: x_dist = self.head_dist(x_dist)
        permute_default: "f32[768, 1000]" = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
        permute_default_1: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:117 in forward_head, code: x = self.head(x)
        permute_default_2: "f32[768, 1000]" = torch.ops.aten.permute.default(primals_153, [1, 0]);  primals_153 = None
        permute_default_3: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_default_2, [1, 0]);  permute_default_2 = None
        return (div_tensor, permute_default_1, permute_default_3)
