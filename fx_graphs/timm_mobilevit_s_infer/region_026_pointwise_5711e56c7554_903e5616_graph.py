class GraphModule(torch.nn.Module):
    def forward(self, addmm_6: "f32[131072, 288]", _shape_param_0, _shape_param_1, arg109_1: "f32[144, 288]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        reshape_default: "f32[512, 256, 288]" = torch.ops.aten.reshape.default(addmm_6, _shape_param_0);  addmm_6 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        neg_default: "f32[512, 256, 288]" = torch.ops.aten.neg.default(reshape_default)
        exp_default: "f32[512, 256, 288]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[512, 256, 288]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[512, 256, 288]" = torch.ops.aten.div.Tensor(reshape_default, add_tensor);  reshape_default = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_1: "f32[131072, 288]" = torch.ops.aten.reshape.default(div_tensor, _shape_param_1);  div_tensor = _shape_param_1 = None
        permute_default: "f32[288, 144]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        return (reshape_default_1, permute_default)
