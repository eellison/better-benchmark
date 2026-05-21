class GraphModule(torch.nn.Module):
    def forward(self, addmm_51: "f32[128, 768]", _shape_param_0, view_129: "f32[128, 1, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default: "f32[128, 1, 768]" = torch.ops.aten.reshape.default(addmm_51, _shape_param_0);  addmm_51 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:122 in forward, code: x = x + self.mlp(self.norm(x))
        add_tensor: "f32[128, 1, 768]" = torch.ops.aten.add.Tensor(view_129, reshape_default);  view_129 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:126 in forward, code: x = x[:, 0]
        select_int: "f32[128, 768]" = torch.ops.aten.select.int(add_tensor, 1, 0);  add_tensor = None
        return select_int
