class GraphModule(torch.nn.Module):
    def forward(self, convolution_77: "f32[128, 48, 1, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        neg_default: "f32[128, 48, 1, 1]" = torch.ops.aten.neg.default(convolution_77)
        exp_default: "f32[128, 48, 1, 1]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[128, 48, 1, 1]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[128, 48, 1, 1]" = torch.ops.aten.div.Tensor(convolution_77, add_tensor);  convolution_77 = add_tensor = None
        return div_tensor
