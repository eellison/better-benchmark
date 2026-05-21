class GraphModule(torch.nn.Module):
    def forward(self, div_64: "f32[128, 32, 49, 49]", _shape_param_0, _shape_param_1, bmm_53: "f32[4096, 49, 49]", _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_default: "f32[128, 32, 49, 49]" = torch.ops.aten.expand.default(div_64, _shape_param_0);  _shape_param_0 = None
        reshape_default: "f32[4096, 49, 49]" = torch.ops.aten.reshape.default(expand_default, _shape_param_1);  expand_default = _shape_param_1 = None
        permute_default: "f32[4096, 49, 49]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1]);  reshape_default = None
        reshape_default_1: "f32[128, 32, 49, 49]" = torch.ops.aten.reshape.default(bmm_53, _shape_param_2);  bmm_53 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        mul_tensor: "f32[128, 32, 49, 49]" = torch.ops.aten.mul.Tensor(reshape_default_1, div_64);  reshape_default_1 = None
        sum_dim_int_list: "f32[128, 32, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [-1], True)
        neg_default: "f32[128, 32, 49, 49]" = torch.ops.aten.neg.default(div_64);  div_64 = None
        fma_default: "f32[128, 32, 49, 49]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor);  neg_default = sum_dim_int_list = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        reshape_default_2: "f32[4096, 49, 49]" = torch.ops.aten.reshape.default(fma_default, _shape_param_3);  fma_default = _shape_param_3 = None
        return (permute_default, reshape_default_2)
