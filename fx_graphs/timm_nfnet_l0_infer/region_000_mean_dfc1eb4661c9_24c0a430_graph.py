class GraphModule(torch.nn.Module):
    def forward(self, convolution_80: "f32[128, 2304, 7, 7]", arg220_1: "f32[1000, 2304]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:569 in forward_features, code: x = self.final_act(x)
        neg_default: "f32[128, 2304, 7, 7]" = torch.ops.aten.neg.default(convolution_80)
        exp_default: "f32[128, 2304, 7, 7]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[128, 2304, 7, 7]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[128, 2304, 7, 7]" = torch.ops.aten.div.Tensor(convolution_80, add_tensor);  convolution_80 = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_dim: "f32[128, 2304, 1, 1]" = torch.ops.aten.mean.dim(div_tensor, [-1, -2], True);  div_tensor = None
        as_strided_default: "f32[128, 2304, 1, 1]" = torch.ops.aten.as_strided.default(mean_dim, [128, 2304, 1, 1], [2304, 1, 2304, 2304]);  mean_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        reshape_default: "f32[128, 2304]" = torch.ops.aten.reshape.default(as_strided_default, _shape_param_0);  as_strided_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:141 in forward, code: x = self.fc(x)
        permute_default: "f32[2304, 1000]" = torch.ops.aten.permute.default(arg220_1, [1, 0]);  arg220_1 = None
        return (reshape_default, permute_default)
