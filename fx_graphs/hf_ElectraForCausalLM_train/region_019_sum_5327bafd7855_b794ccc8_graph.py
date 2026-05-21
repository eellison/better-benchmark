class GraphModule(torch.nn.Module):
    def forward(self, bmm_65: "f32[256, 512, 512]", gt_4: "b8[64, 4, 512, 512]", where_3: "f32[64, 4, 512, 512]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        reshape_default: "f32[64, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_65, _shape_param_0);  bmm_65 = _shape_param_0 = None
        convert_element_type_default: "f32[64, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_4, torch.float32);  gt_4 = None
        mul_tensor: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  reshape_default = mul_tensor = None
        mul_tensor_2: "f32[64, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, where_3);  mul_tensor_1 = None
        sum_dim_int_list: "f32[64, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [-1], True)
        neg_default: "f32[64, 4, 512, 512]" = torch.ops.aten.neg.default(where_3);  where_3 = None
        fma_default: "f32[64, 4, 512, 512]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_2);  neg_default = sum_dim_int_list = mul_tensor_2 = None
        reshape_default_1: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(fma_default, _shape_param_1);  fma_default = _shape_param_1 = None
        return reshape_default_1
