"""
Standalone repro captured via capture_hook.
Label: hf_DistilBertForMaskedLM_train
Pattern hash: 971a6a67ccc7
Shape hash: ae0a395f
"""
_shapes_config = "(T([32768, 768], f32), T([256, 128, 768], f32), T([32768, 768], f32), T([32768, 768], f32), T([768], f32), T([256, 128, 768], f32), T([256, 128, 1], f32), T([256, 128, 768], b8), T([768, 3072], f32), S([256, 128, 768]), S([256, 128, 768]), S([256, 128, 768]), S([32768, 768]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_58: "f32[32768, 768]", mul_236: "f32[256, 128, 768]", mm_60: "f32[32768, 768]", mm_62: "f32[32768, 768]", primals_21: "f32[768]", mul_15: "f32[256, 128, 768]", div_19: "f32[256, 128, 1]", gt_2: "b8[256, 128, 768]", primals_19: "f32[768, 3072]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:189 in forward, code: value_layer = self.v_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_58, _shape_param_0);  mm_58 = _shape_param_0 = None
        add_tensor: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_236, reshape_default);  mul_236 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:188 in forward, code: key_layer = self.k_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_1: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_60, _shape_param_1);  mm_60 = _shape_param_1 = None
        add_tensor_1: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_1);  add_tensor = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:187 in forward, code: query_layer = self.q_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_2: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_62, _shape_param_2);  mm_62 = _shape_param_2 = None
        add_tensor_2: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_2);  add_tensor_1 = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        mul_tensor: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_tensor_2, primals_21);  add_tensor_2 = primals_21 = None
        mul_tensor_1: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_15);  mul_tensor = None
        sum_dim_int_list_1: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_15, sum_dim_int_list_1);  mul_15 = sum_dim_int_list_1 = None
        sub_tensor: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(div_19, sub_tensor_1);  div_19 = sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:227 in ff_chunk, code: x = self.dropout(x)
        convert_element_type_default: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_2, torch.float32);  gt_2 = None
        mul_tensor_5: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_6: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:226 in ff_chunk, code: x = self.lin2(x)
        reshape_default_3: "f32[32768, 768]" = torch.ops.aten.reshape.default(mul_tensor_6, _shape_param_3);  mul_tensor_6 = _shape_param_3 = None
        permute_default: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
        permute_default_1: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_3, permute_default_1)



def make_inputs():
    return [
    torch.randn([32768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([256, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([256, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([256, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [256, 128, 768], dtype=torch.bool, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    [256, 128, 768],  # _shape_param_0
    [256, 128, 768],  # _shape_param_1
    [256, 128, 768],  # _shape_param_2
    [32768, 768],  # _shape_param_3
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
