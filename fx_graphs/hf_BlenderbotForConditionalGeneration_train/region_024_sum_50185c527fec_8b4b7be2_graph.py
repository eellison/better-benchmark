class GraphModule(torch.nn.Module):
    def forward(self, bmm_3: "f32[512, 128, 128]", where_2: "f32[16, 32, 128, 128]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        reshape_default: "f32[16, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_3, _shape_param_0);  bmm_3 = _shape_param_0 = None
        mul_tensor: "f32[16, 32, 128, 128]" = torch.ops.aten.mul.Tensor(reshape_default, where_2);  reshape_default = None
        sum_dim_int_list: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [-1], True)
        neg_default: "f32[16, 32, 128, 128]" = torch.ops.aten.neg.default(where_2);  where_2 = None
        fma_default: "f32[16, 32, 128, 128]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor);  neg_default = sum_dim_int_list = mul_tensor = None
        reshape_default_1: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(fma_default, _shape_param_1);  fma_default = _shape_param_1 = None
        return reshape_default_1
