class GraphModule(torch.nn.Module):
    def forward(self, addmm_51: "f32[128, 768]", view_129: "f32[128, 1, 768]", rsqrt_24: "f32[128, 256, 1]", rsqrt_23: "f32[128, 256, 1]", rsqrt_22: "f32[128, 256, 1]", rsqrt_21: "f32[128, 256, 1]", rsqrt_20: "f32[128, 256, 1]", rsqrt_19: "f32[128, 256, 1]", rsqrt_18: "f32[128, 256, 1]", rsqrt_17: "f32[128, 256, 1]", rsqrt_16: "f32[128, 256, 1]", rsqrt_15: "f32[128, 256, 1]", rsqrt_14: "f32[128, 256, 1]", rsqrt_13: "f32[128, 256, 1]", rsqrt_12: "f32[128, 256, 1]", rsqrt_11: "f32[128, 256, 1]", rsqrt_10: "f32[128, 256, 1]", rsqrt_9: "f32[128, 256, 1]", rsqrt_8: "f32[128, 256, 1]", rsqrt_7: "f32[128, 256, 1]", rsqrt_6: "f32[128, 256, 1]", rsqrt_5: "f32[128, 256, 1]", rsqrt_4: "f32[128, 256, 1]", rsqrt_3: "f32[128, 256, 1]", rsqrt_2: "f32[128, 256, 1]", rsqrt_1: "f32[128, 256, 1]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default: "f32[128, 1, 768]" = torch.ops.aten.reshape.default(addmm_51, _shape_param_0);  addmm_51 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:122 in forward, code: x = x + self.mlp(self.norm(x))
        add_tensor: "f32[128, 1, 768]" = torch.ops.aten.add.Tensor(view_129, reshape_default);  view_129 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:126 in forward, code: x = x[:, 0]
        select_int: "f32[128, 768]" = torch.ops.aten.select.int(add_tensor, 1, 0);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        div_tensor: "f32[128, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
        div_tensor_1: "f32[128, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
        div_tensor_2: "f32[128, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
        div_tensor_3: "f32[128, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
        div_tensor_4: "f32[128, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
        div_tensor_5: "f32[128, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
        div_tensor_6: "f32[128, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
        div_tensor_7: "f32[128, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
        div_tensor_8: "f32[128, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
        div_tensor_9: "f32[128, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
        div_tensor_10: "f32[128, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
        div_tensor_11: "f32[128, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
        div_tensor_12: "f32[128, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
        div_tensor_13: "f32[128, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
        div_tensor_14: "f32[128, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
        div_tensor_15: "f32[128, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
        div_tensor_16: "f32[128, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
        div_tensor_17: "f32[128, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
        div_tensor_18: "f32[128, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
        div_tensor_19: "f32[128, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
        div_tensor_20: "f32[128, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
        div_tensor_21: "f32[128, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
        div_tensor_22: "f32[128, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
        div_tensor_23: "f32[128, 256, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
        return (select_int, div_tensor, div_tensor_1, div_tensor_2, div_tensor_3, div_tensor_4, div_tensor_5, div_tensor_6, div_tensor_7, div_tensor_8, div_tensor_9, div_tensor_10, div_tensor_11, div_tensor_12, div_tensor_13, div_tensor_14, div_tensor_15, div_tensor_16, div_tensor_17, div_tensor_18, div_tensor_19, div_tensor_20, div_tensor_21, div_tensor_22, div_tensor_23)
